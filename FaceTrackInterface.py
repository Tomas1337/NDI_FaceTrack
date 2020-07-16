from PyQt5.QtCore import QTextStream, QFile, QDateTime, QSize, Qt, QTimer,QRect, QThread, QObject, pyqtSignal,pyqtSlot, QRunnable
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QBoxLayout,
        QProgressBar, QPushButton, QButtonGroup,
        QSlider, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget, QAbstractButton, QMainWindow, QAction, QMenu,
        QStyleOptionSlider, QStyle, QSpacerItem, QSizePolicy)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QBrush, QFont, QPen, QPalette, QColor, QIcon
from face_tracking.objcenter import *
from ndi_camera import ndi_camera
import numpy as np
import NDIlib as ndi
import cv2, dlib, time, styling, sys
from tool.custom_widgets import *
from face_tracking.camera_control import *
import keyboard


class MainWindow(QMainWindow):
    signalStatus = pyqtSignal(str)
    face_track_signal = pyqtSignal(list)
    face_home_signal = pyqtSignal(list)

    def __init__(self, parent = None):
        super(MainWindow, self).__init__(parent)
        
        #Initialize the GUI object
        #Create a new worker thread
        self.gui = WindowGUI(self)
        self.setCentralWidget(self.gui) 
        self.createWorkerThread()
        self.createMenuBar()

        #Make any cross object connections
        self._connectSignals()
        title = "NDI FaceTrack"
        self.setWindowTitle(title) 
        self.gui.show()
        self.setFixedSize(700, 750)

    def _connectSignals(self):
        self.signalStatus.connect(self.gui.updateStatus)
        self.sources.aboutToShow.connect(self.worker.findSources)
        self.sources.aboutToShow.connect(self.worker_thread.start)
        #self.aboutToQuit.connect(self.forceWorkerQuit)

    def createMenuBar(self):
        bar = self.menuBar()
        self.sources = bar.addMenu("Sources")

    @pyqtSlot(list)
    def populateSources(self, _list):
        self.sources.clear()
        for i, item in enumerate(_list):
            entry = self.sources.addAction(item)
            self.sources.addAction(entry)
            #Lamda function to connect the menu item with it's index
            entry.triggered.connect(lambda e, x=i: self.worker.connect_to_camera(x))
            entry.triggered.connect(self.vid_worker_thread.start)

    ### SIGNALS
    def createWorkerThread(self):
        self.worker = WorkerObject()
        self.worker_thread = QThread()  
        self.worker.moveToThread(self.worker_thread)
        
        self.vid_worker = Video_Object()
        self.vid_worker_thread = QThread()
        self.vid_worker.moveToThread(self.vid_worker_thread)

        self.face_detector = FaceDetectionWidget()
        self.face_detector.moveToThread(self.vid_worker_thread)

        #Connect any worker signals
        self.worker.signalStatus.connect(self.gui.updateStatus)
        self.worker.ptz_object_signal.connect(self.vid_worker.read_video)
        self.worker.ptz_list_signal.connect(self.populateSources)
        self.worker.info_status.connect(self.gui.updateInfo)
        self.worker.enable_controls_signal.connect(self.gui.enable_controls)
        
        self.vid_worker.FaceFrameSignal.connect(self.face_track_signal_handler)
        self.vid_worker.DisplayNormalVideoSignal.connect(self.gui.setImage)

        self.face_detector.CameraZoomControlSignal.connect(self.vid_worker.zoom_camera_control)
        self.face_detector.DisplayVideoSignal.connect(self.gui.setImage)
        self.face_detector.CameraControlSignal.connect(self.vid_worker.camera_control)
        self.face_detector.info_status.connect(self.gui.updateInfo)
        self.face_detector.signalStatus.connect(self.gui.updateStatus)

        self.face_track_signal.connect(self.face_detector.main_track)

        self.gui.reset_track_button.clicked.connect(self.face_detector.reset_tracker)
        self.gui.azoom_lost_face_button.clicked.connect(self.face_detector.detect_autozoom_state)
        self.gui.face_lock_button.clicked.connect(self.face_detector.detect_face_lock_state)
        self.gui.y_enable_button.clicked.connect(self.face_detector.detect_ytrack_state)
        self.gui.gamma_slider.valueChanged.connect(self.face_detector.gamma_slider_values)
        self.gui.x_minE_slider.valueChanged.connect(self.face_detector.xmin_e_val)
        self.gui.y_minE_slider.valueChanged.connect(self.face_detector.ymin_e_val)  
        self.gui.zoom_slider.valueChanged.connect(self.vid_worker.zoom_handler)
        self.gui.face_track_button.clicked.connect(self.vid_worker.detect_face_track_state)
        self.gui.face_track_button.clicked.connect(self.gui.enable_track_buttons)
        self.gui.reset_default_button.clicked.connect(self.gui.reset_defaults_handler)
        self.gui.reset_default_button.click()

    def forceWorkerQuit(self):
        if self.worker_thread.isRunning():
            self.worker_thread.terminate()
            self.worker_thread.wait()

        if self.vid_worker_thread.isRunning():
            self.vid_worker_thread.terminate()
            self.vid_worker_thread.wait()

    @pyqtSlot(np.ndarray)
    def face_track_signal_handler(self, frame):
        #Must emit a a list containing (np.ndarray, string)
        if self.gui.center_track_button.isChecked():
            self.face_track_signal.emit([frame, 'center'])
        elif self.gui.left_track_button.isChecked():
            self.face_track_signal.emit([frame, 'left'])
        elif self.gui.right_track_button.isChecked():
            self.face_track_signal.emit([frame, 'right'])

class WindowGUI(QWidget):
    def __init__(self, parent):
        super(WindowGUI, self).__init__(parent)
        self.label_status = QLabel('Created by: JTJTi Digital Video + Radio', self)
        
        #Main Track Button
        self.face_track_button = QTrackingButton('TRACK')
        self.face_track_button.setCheckable(True)
        self.face_track_button.setDisabled(True)

        #Video Widgets
        self.video_frame = QLabel('',self)
        self.video_frame.setFixedHeight(400)
        self.video_frame.setAutoFillBackground(True)
        self.video_frame.setStyleSheet("background-color:#000000;")
        self.video_frame.setAlignment(Qt.AlignCenter)

        #Info Panel
        self.info_panel = QLabel('No Signal',self)
        self.info_panel.setFont(QFont("Arial", 24, QFont.Bold))
        self.info_panel.setAlignment(Qt.AlignCenter)
        self.info_panel.setStyleSheet("background-color:#000000;")
        self.info_panel.setMargin(10)
        
        #Tracking Buttons
        self.left_track_button = QTrackingButton('LEFT')
        self.left_track_button.setCheckable(True)
        self.left_track_button.setDisabled(True)

        self.center_track_button = QTrackingButton('CENTER')
        self.center_track_button.setCheckable(True)
        self.center_track_button.setDisabled(True)

        self.right_track_button = QTrackingButton('RIGHT')
        self.right_track_button.setCheckable(True)
        self.right_track_button.setDisabled(True)

        self.reset_track_button = QResetButton(self)
        self.reset_track_button.setDisabled(True)
        #self.reset_track_button.setFixedHeight(40)
        self.reset_track_button.setMinimumWidth(300)

        #Face Lock Button
        self.face_lock_button = QPushButton('LOCK TO FACE')
        self.face_lock_button.setCheckable(True)
        #self.face_lock_button.setFixedHeight(40)
        #self.face_lock_button.setDisabled(True)

        #Y-Axis Tracking
        self.y_enable_button = QToggleButton('Y-Axis Tracking')
        self.y_enable_button.setCheckable(True)
        self.y_enable_button.setChecked(True)
        self.y_enable_button.setFixedHeight(70)
        self.y_enable_button.setDisabled(True)

        #Lost Auto Zoom Out Buttons
        self.azoom_lost_face_button = QToggleButton('Auto-Find Lost')
        self.azoom_lost_face_button.setCheckable(True)
        self.azoom_lost_face_button.setChecked(True)
        self.azoom_lost_face_button.setFixedHeight(70)
        self.azoom_lost_face_button.setDisabled(True)

        #Tracking position buttons
        self.tracking_button_group = QButtonGroup()
        self.tracking_button_group.addButton(self.left_track_button)
        self.tracking_button_group.addButton(self.center_track_button)
        self.tracking_button_group.addButton(self.right_track_button)
        self.tracking_button_group.setExclusive(True)

        #Gamma Sliders
        gamma_label = QLabel()
        gamma_label.setText('Gamma')
        self.gamma_slider = QSlider()
        self.gamma_slider.setOrientation(Qt.Horizontal)
        self.gamma_slider.setValue(6)
        self.gamma_slider.setTickInterval(10)
        self.gamma_slider.setMinimum(1)
        self.gamma_slider.setMaximum(10)

        #Minimum Error Slider
        x_minE_label = QLabel()
        x_minE_label.setText('Minimum X-Error:')
        self.x_minE_slider = QSlider()
        self.x_minE_slider.setOrientation(Qt.Horizontal)
        self.x_minE_slider.setMinimum(1)
        self.x_minE_slider.setMaximum(10)
        self.x_minE_slider.setValue(5)
        y_minE_label = QLabel()
        y_minE_label.setText('Minimum Y-Error:')
        self.y_minE_slider = QSlider()
        self.y_minE_slider.setMinimum(1)
        self.y_minE_slider.setMaximum(10)
        self.y_minE_slider.setOrientation(Qt.Horizontal)
        self.y_minE_slider.setValue(5)

        #Zoom Slider
        zoom_slider_label = QLabel()
        zoom_slider_label.setText('Zoom')
        self.zoom_slider = QSlider()
        self.zoom_slider.setOrientation(Qt.Horizontal)
        self.zoom_slider.setValue(0)
        self.zoom_slider.setTickInterval(11)
        self.zoom_slider.setMinimum(0)
        self.zoom_slider.setMaximum(11)
        self.zoom_slider.setDisabled(True)

        #Reset To Default
        self.reset_default_button = QPushButton('Reset Defaults', self)

        ## LAYOUTING ##
        layout = QVBoxLayout(self)
        vid_layout = QVBoxLayout(self)
        controls_layout = QGridLayout(self)
        controls_layout.setSpacing(10)
        layout.addLayout(vid_layout)
        
        vid_layout.addWidget(self.info_panel)
        vid_layout.addWidget(self.video_frame)
        vid_layout.setSpacing(0)

        layout.addLayout(controls_layout)
        controls_layout.addWidget(self.face_track_button, 0,0,1,-1)
        layoutFacePosTrack = QHBoxLayout(self)
        layoutFacePosTrack.addWidget(self.left_track_button)
        layoutFacePosTrack.addWidget(self.center_track_button)
        layoutFacePosTrack.addWidget(self.right_track_button)
        layoutFacePosTrack.setSpacing(5)
        controls_layout.setAlignment(Qt.AlignCenter)
        controls_layout.addLayout(layoutFacePosTrack,1,0,1,-1)

        #Additional Options
        secondary_controls_layout = QVBoxLayout()
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(zoom_slider_label)
        zoom_layout.addWidget(self.zoom_slider)
        secondary_controls_layout.addLayout(zoom_layout)
        #secondary_controls_layout.addWidget(self.face_lock_button)
        secondary_controls_layout.addWidget(self.reset_track_button)
        secondary_controls_layout.setSpacing(5)
        
        toggle_controls_layout = QHBoxLayout()
        toggle_controls_layout.addWidget(self.azoom_lost_face_button)
        toggle_controls_layout.addWidget(self.y_enable_button)
        toggle_controls_layout.setSpacing(7)
        secondary_controls_layout.addLayout(toggle_controls_layout)
        controls_layout.addLayout(secondary_controls_layout,2,0)
        
        #Advanced Options
        adv_options_layout = QGridLayout()
        adv_options_group = QGroupBox('Advanced Controls')
        adv_options_group.setStyleSheet("QGroupBox {border-style: solid; border-width: 1px; border-color: grey; text-align: left; font-weight:bold; padding-top: 5px;} QGroupBox::title {right:100px; bottom : 6px;margin-top:4px;}")
        adv_options_group.setCheckable(True)
        adv_options_group.setChecked(False)
        adv_options_layout.setSpacing(7)
        adv_options_layout.addWidget(gamma_label, 1,1)
        adv_options_layout.addWidget(x_minE_label, 2,1)
        adv_options_layout.addWidget(y_minE_label, 3,1)
        adv_options_layout.addWidget(self.gamma_slider,1,2)
        adv_options_layout.addWidget(self.x_minE_slider,2,2)
        adv_options_layout.addWidget(self.y_minE_slider,3,2) 
        adv_options_layout.addWidget(self.reset_default_button,4,2) 
        adv_options_group.setLayout(adv_options_layout)
        controls_layout.addWidget(adv_options_group,2,1)

        layout.addStretch(1)
        layout.addWidget(self.label_status)
        
        #self.setFixedSize(700, 720)

    @pyqtSlot(str)
    def updateStatus(self, status):
        self.label_status.setText(status)

    @pyqtSlot(str)
    def updateInfo(self, status):
        self.info_panel.setText(status)
        
    @pyqtSlot(QImage)
    def setImage(self, image):
        img = image.scaled(640, 480, Qt.KeepAspectRatio)
        self.video_frame.setPixmap(QPixmap.fromImage(img))

    #Activates/Deactivates all the tracking buttons based if FaceTrack button is on/off
    @pyqtSlot(bool)
    def enable_track_buttons(self, state):
        if state:
            self.center_track_button.setEnabled(state)
            self.left_track_button.setEnabled(state)
            self.right_track_button.setEnabled(state)

            if (not self.center_track_button.isChecked() and
            not self.left_track_button.isChecked() and
            not self.right_track_button.isChecked()):
                self.center_track_button.setChecked(True)

        elif state is False:
            self.center_track_button.setChecked(state)
            self.left_track_button.setChecked(state)
            self.right_track_button.setChecked(state)
            
            self.center_track_button.setEnabled(state)
            self.left_track_button.setEnabled(state)
            self.right_track_button.setEnabled(state)

    def reset_defaults_handler(self, state):
        self.gamma_slider.setValue(6)
        self.x_minE_slider.setValue(0.5)
        self.y_minE_slider.setValue(0.5)

    def enable_controls(self):
        self.face_track_button.setEnabled(True)
        self.reset_track_button.setEnabled(True)
        self.y_enable_button.setEnabled(True)
        self.azoom_lost_face_button.setEnabled(True)
        self.zoom_slider.setEnabled(True)

class WorkerObject(QObject):
    signalStatus = pyqtSignal(str)
    ptz_list_signal = pyqtSignal(list)
    ptz_object_signal = pyqtSignal(object)
    info_status = pyqtSignal(str)
    enable_controls_signal = pyqtSignal()

    def __init__(self, parent=None):
        super(self.__class__, self).__init__(parent)
    
    @pyqtSlot()
    def findSources(self):
        self.ndi_cam = ndi_camera()
        self.signalStatus.emit('Searching for PTZ cameras')
        (ptz_list, sources) = self.ndi_cam.find_ptz()
        self.ptz_names = [sources[i].ndi_name for i in ptz_list]
        self.signalStatus.emit('Idle')
        self.ptz_list_signal.emit(self.ptz_names)

    @pyqtSlot(int)
    def connect_to_camera(self, int):
        self.signalStatus.emit('Connecting to camera') 
        ndi_recv = self.ndi_cam.camera_connect(int-1)
        self.signalStatus.emit('Connected to {}'.format(self.ptz_names[int-1]))
        self.ptz_object_signal.emit(ndi_recv)
        self.info_status.emit('Signal: {}'.format(self.ptz_names[int-1]))
        self.enable_controls_signal.emit()

#Handles the reading and displayingg of video
class Video_Object(QObject):
    PixMapSignal = pyqtSignal(QImage)
    FaceFrameSignal = pyqtSignal(np.ndarray)
    DisplayNormalVideoSignal = pyqtSignal(QImage)
    
    def __init__(self,parent=None):
        super(self.__class__, self).__init__(parent)
        self.face_track_state = False
        self.frame_count = 1
        self.skip_frames = 1
        self.keypress = False

    @pyqtSlot(object)
    def read_video(self, ndi_object):
        self.ndi_recv = ndi_object
        while True:
            t,v,_,_ = ndi.recv_capture_v2(self.ndi_recv, 1)
            
            if t == ndi.FRAME_TYPE_VIDEO:
                #print('checking')
                self.frame_count += 1   
                frame = v.data
                frame = frame[:,:,:3]
                resize_factor = 1
                frame = cv2.resize(frame, (int(frame.shape[1] * resize_factor), int(frame.shape[0] * resize_factor)))
                #Code to process the GUI events before proceeding
                QApplication.processEvents()

                camera_move_speed = 0.3
                camera_zoom_speed = 0.9

                if keyboard.is_pressed('e') and not self.keypress:
                    self.zoom_camera_control(camera_zoom_speed)
                    self.keypress=True
                elif self.keypress and not keyboard.is_pressed('e'):
                    self.keypress = False
                    self.zoom_camera_control(0.0)
                elif keyboard.is_pressed('q') and not self.keypress:
                    self.zoom_camera_control(camera_zoom_speed * -1)
                    self.keypress=True
                elif self.keypress and not keyboard.is_pressed('q'):
                    self.keypress = False
                    self.zoom_camera_control(0.0)
                elif keyboard.is_pressed('w') and not self.keypress:
                    self.camera_control(0.0,camera_move_speed)
                    self.keypress=True
                elif self.keypress and not keyboard.is_pressed('w'):
                    self.keypress = False
                    self.camera_control(0.0,0.0)
                elif keyboard.is_pressed('s') and not self.keypress:
                    self.camera_control(0.0,camera_move_speed*-1)
                    self.keypress=True
                elif self.keypress and not keyboard.is_pressed('s'):
                    self.keypress = False
                    self.camera_control(0.0,0.0)
                elif keyboard.is_pressed('a') and not self.keypress:
                    self.camera_control(camera_move_speed,0.0)
                    self.keypress=True
                elif self.keypress and not keyboard.is_pressed('a'):
                    self.keypress = False
                    self.camera_control(0.0,0.0)
                elif keyboard.is_pressed('d') and not self.keypress:
                    self.camera_control(camera_move_speed * -1,0.0)
                    self.keypress=True
                elif self.keypress and not keyboard.is_pressed('d'):
                    self.keypress = False
                    self.camera_control(0.0,0.0)

                if self.face_track_state == False:
                    self.display_plain_video(frame)
                elif self.face_track_state == True:
                    if self.frame_count%self.skip_frames == 0:
                        tic1 = time.time()
                        self.FaceFrameSignal.emit(frame) 
                    else:
                        continue
                    #print(tic1 - time.time())
                ndi.recv_free_video_v2(self.ndi_recv, v)

    @pyqtSlot(bool)
    def detect_face_track_state(self, state):
        self.face_track_state = state

    def display_plain_video(self, image):
        height, width, colors = image.shape
        bytesPerLine = 3 * width
        image = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        image = image.rgbSwapped()
        self.DisplayNormalVideoSignal.emit(image)

    @pyqtSlot(float, float)
    def camera_control(self, Xspeed, Yspeed):
        #Camera Control
        for i in range(1,2):
            ndi.recv_ptz_pan_tilt_speed(self.ndi_recv, Xspeed, Yspeed)
        #ndi.recv_ptz_pan_tilt_speed(self.ndi_recv, 0, 0)

    @pyqtSlot(float)
    def zoom_camera_control(self, ZoomValue):
        ndi.recv_ptz_zoom_speed(self.ndi_recv, ZoomValue)

    @pyqtSlot(int)
    def zoom_handler(self, ZoomLevel):
        ndi.recv_ptz_zoom(self.ndi_recv, ZoomLevel/10   )

class FaceDetectionWidget(QObject):
    DisplayVideoSignal = pyqtSignal(QImage)
    CameraControlSignal = pyqtSignal(float, float)
    CameraZoomControlSignal = pyqtSignal(float)
    pass_to_face_trackSignal = pyqtSignal()
    signalStatus = pyqtSignal(str)
    info_status = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        #self.body_obj = Hog_Detector()
        self.image = QImage()
        self.last_loc = [320, 180]
        self.track_coords = []
        self.prev_X_Err = None
        self.prev_Y_Err  = None
        self.check_frames_num  = 3600 #Test this out
        #self.check_frames_num = 60
        self.matching_threshold = 0.02
        self.base_feature = None
        self.btrack_ok = False
        self.focal_length = 129

        #Logging purposes
        #self.writer = csv_save()

        #Counters
        self.frame_count = 1
        self.lost_tracking = 0
        self.f_track_count = 0
        self.overlap_counter = 0
        self.tracking_confirm = 0
        self.lost_t = 0

        #Trackers
        self.tracker = None
        self.f_tracker = None
        self.face_lock = Face_Locker()

        #Object Detectors
        self.face_obj = FastMTCNN() #TRY THIS FIRST
        self.body_obj = Yolov3()

        #Slider and button Values
        self.y_trackState = True
        self.yminE = 0.13
        self.xminE = 0.13
        self.gamma = 0.6 
        self.ZoomValue = 0.0
        self.autozoom_state = True
        self.face_lock_state = False
        self.zoom_state = 0
        self.face_coords = []

    def face_tracker(self, frame):
        #ptvsd.debug_this_thread()
        """
        Do a check every 5 seconds
            If there are detections, check to see if the new detections overlap the current face_coordinates. 
                If it does overlap, then refresh the self.face_coords with the new test_coords
                If no overlap, then refresh the tracker so that it conducts a whole frame search
            If no detections in the body frame, detect on whole frame
                If there are face detections from the whole frame, return the face with the highest confidence
                Else, empty the face tracker and return []

        Check to see if tracker is ok
            Track Current face

        If track is not ok:
            Add to the lost face count
            When face Count reaches N, it empties the face_tracker
        """
        if not self.f_tracker is None:
            tic = time.time()
            ok, position = self.f_tracker.update(frame)
            #print("Face Tracker update takes:{:.2f}s".format(time.time() - tic))
            if self.frame_count % 300== 0:
                print('Running a {}s check'.format(300/30))
                #test_coords = self.face_obj.update(frame)
                test_coords = self.face_obj.get_all_locations(frame)

                if len(test_coords) > 0: #There are detections
                    print('testing here')
                    for i, j in enumerate(test_coords):
                        if self.overlap_metric(j, self.face_coords) >= 0.75:
                            x,y,w,h = j
                            break
                    return []

                else: #No detections
                    print('testing here 2')
                    self.f_tracker = None
                    return []
            
            elif ok:
                x = int(position[0])
                y = int(position[1])
                w = int(position[2])
                h = int(position[3])
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 1)
                x,y,w,h = [0 if i < 0 else i for i in [x,y,w,h]]
                self.f_track_count = 0
                face_coords = x,y,w,h
                
            else:
                if self.f_track_count > 5:
                    print('Lost face') 
                    self.info_status.emit('Refreshing Face Detector')     
                    self.f_tracker = None
                    return []
                else:
                    self.f_track_count += 1
                    return []

        elif self.f_tracker is None:
            """
            Detect a face inside the body Coordinates
            If no detections in the body frame, detect on whole frame which gets the face detection with the strongest confidence
            """
            if self.frame_count % 30 == 0:
                face_coords = self.face_obj.update(frame)
            else:
                face_coords = []
                
            if len(face_coords) > 0:
                x,y,w,h = face_coords
            else:
               return []

            #Start a face tracker
            self.f_tracker = cv2.TrackerCSRT_create()
            self.f_tracker.init(frame, (x,y,w,h))
            print('Initiating a face tracker')

        self.face_coords = x,y,w,h
        return x,y,w,h

    def body_tracker(self, frame, centerX, centerY):
        #ptvsd.debug_this_thread()
        if self.tracker is None:
            #Detect Objects using YOLO every 1 second if No Body Tracker    
            boxes = []
            if self.frame_count % 15 == 0:
                (idxs, boxes, _, _, classIDs, confidences) = self.body_obj.update(frame, (centerX, centerY))
                print('Running a YOLO')

            if len(boxes) <= 0:
                return []

            elif len(boxes) == 1:
                x,y,w,h = boxes[np.argmax(confidences)]
                x,y,w,h = [0 if i < 0 else int(i) for i in [x,y,w,h]]

            elif len(boxes) > 1 and len(self.face_coords) >= 1:
                for i, g in enumerate(boxes):
                    if self.overlap_metric(g, self.face_coords) >= 0.5:
                        x,y,w,h = [0 if i < 0 else int(i) for i in [x,y,w,h]]
                        x,y,w,h = [int(p) for p in boxes[i]]
                        break

            #Start the body tracker for the given xywh
            self.tracker = cv2.TrackerKCF_create()
            try:
                self.tracker.init(frame, (x,y,w,h))
            except UnboundLocalError:
                return []
            return x,y,w,h

        #If theres a tracker already            
        elif not self.tracker is None:
            tic = time.time()       
            self.btrack_ok, position = self.tracker.update(frame)
            #print("Body Tracker update takes:{:.2f}s".format(time.time() - tic))
            if self.btrack_ok:
                x = int(position[0])
                y = int(position[1])
                w = int(position[2])
                h = int(position[3])
                x,y,w,h = [0 if i < 0 else i for i in [x,y,w,h]]

            else:
                print('Tracking Fail')
                return []
        
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,255), 2)
            return x,y,w,h

    @pyqtSlot(list)
    def main_track(self, _list):
        #ptvsd.debug_this_thread()
        self.track_coords = []
        start = time.time()
        frame = _list[0]
        try:
            center_slot = _list[1]
        except IndexError:
            center_slot = 'center'

        (H, W) = frame.shape[:2]
        centerX= W // 2
        centerY = H // 2
        objX = centerX
        objY = centerY

        #Adjust Center X, here  
        if center_slot == 'left':
            centerX = W//4
        elif center_slot == 'right':
            centerX = int(W//1.33)
        elif center_slot == 'center':
            pass
        cv2.circle(frame,(centerX, centerY), 3, (255,255,255), 2)
        
        #Trackers
        face_coords = self.face_tracker(frame)
        if len(face_coords) <= 0:
            body_coords = self.body_tracker(frame, centerX, centerY)
    
        #Face Locking
        """
        Check if the face_lock button is activated
        If activated, check if we have something locked on already
            If yes, check to see if current face matches any of the known_face_encodings
            If it does not match:
                Scan the whole frame for faces and put into a list the detected face_coords
                Loop on each face coord:
                    Extract encodings of face coord
                    Check to see if looped encoding have matches in the known_face_encodings
                This will output the matches list which contains:
                    Results = [False, False, Index]
                    False means no match of current face to known face encodings
                    Index means best index match of current faces among the known faces
                Get the first non False Value
                Use that to change to the new face_coords

            If not locked on; This assumes that you want to register a new face
                Try to register new face
                If able to register new face, change the face_locked_on state to True

        """
        self.face_lock_state = False
        if self.face_lock_state is True and self.frame_count%30==0:        
            if self.face_lock.face_locked_on is True:
                #Check to see if detected face matches the list of known faces
                current_face_encoding = self.face_lock.face_encode(frame, face_coords)
                if self.face_lock.does_it_match(current_face_encoding) is not False:
                    print("Current face Matches")
                    pass
                else:
                    print('Current Face does not match')
                    temp_face_coords = self.face_obj.get_all_locations(frame)
                    results = []
                    for i in temp_face_coords:
                        current_face_encoding = self.face_lock.face_encode(frame, [i])
                        result = self.face_lock.does_it_match(current_face_encoding)
                        results.append(result)
                    
                    if any(results) != False:
                        f = next((i for i, x in enumerate(results) if x), None) 
                        face_coords = temp_face_coords[f]

                    else:
                        face_coords = []

            elif self.face_lock.face_locked_on is False:
                ok_register = self.face_lock.register_new_face(frame, face_coords)
                self.face_lock.face_locked_on = ok_register
                
                if ok_register:
                    print('Sucessfully registerd new face')
                else:
                    print('Failed to register new face')

        #Cascade to choose between whether to home to face or body. 
        #Face tracking takes priority
        try:
            x,y,w,h = face_coords
            self.track_coords = [x,y,x+w,y+h]
            self.info_status.emit('Tracking Face')
        except (ValueError, TypeError) as e:
            try:
                x,y,w,h = body_coords
                self.track_coords = [x,y,x+w,y+h]
                self.info_status.emit('Tracking Body')
                self.overlap_counter = 0
            except (ValueError, UnboundLocalError) as e:
                pass
        
        #Normal Tracking
        if len(self.track_coords) > 0:
            [x,y,x2,y2] = self.track_coords
            objX = int(x+(x2-x)//2)
            objY = int(y+(y2-y)//3)
            self.last_loc = (objX, objY)
            cv2.circle(frame,(objX, objY), 3, (255,0,255), 2)
            self.lost_tracking = 0
            self.CameraZoomControlSignal.emit(0.0)

        #Initiate Lost Tracking sub-routine
        else:
            if self.lost_tracking > 100 and self.lost_tracking < 500 and self.autozoom_state:
                self.info_status.emit("Lost Tracking Sequence Secondary")
                self.CameraZoomControlSignal.emit(-0.7)

            elif self.lost_tracking < 20:
                objX = self.last_loc[0]
                objY = self.last_loc[1]
                self.lost_tracking += 1
                print('Lost tracking. Going to last known location of object')
                self.info_status.emit("Lost Tracking Sequence Initial")
                #self.CameraZoomControlSignal.emit(0.0)

            else:
                objX = centerX
                objY = centerY
                print('Lost object. Centering')
                self.info_status.emit("Lost Object. Recenter subject")
                self.lost_tracking += 1
                #self.CameraZoomControlSignal.emit(0.0)
                if self.lost_tracking < 500 and self.lost_tracking%100:
                    self.tracker = None

        #Convert to qimage then send to display to GUI
        self.image = self.get_qimage(frame)
        self.DisplayVideoSignal.emit(self.image)

        ## CAMERA CONTROL
        x_controller = PTZ_Controller_Novel(self.focal_length, self.gamma)
        x_speed = x_controller.omega_tur_plus1(objX, centerX, RMin = 0.1)

        y_controller = PTZ_Controller_Novel(self.focal_length, self.gamma)
        y_speed = y_controller.omega_tur_plus1(objY, centerY, RMin = 0.1, RMax=7.5)
        #y_speed = 0
        if self.y_trackState is False:
            y_speed = 0

        self.CameraControlSignal.emit(x_speed, y_speed)
        #self.writer.update(Xerror, Yerror,Xspeed,Yspeed, objX, objY, self.frame_count)
        self.frame_count += 1

    def get_qimage(self, image):
        height, width, _ = image.shape
        bytesPerLine = 3 * width
        image = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        image = image.rgbSwapped()
        return image

    def overlap_metric(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        # BoxA and BoxB usually comes in x,y,w,h
        # Tranlate into x,y,x1,y1
        def _translate(box):
            x,y,w,h = box
            x1 = x+w
            y1 = y+h
            return [x,y,x1,y1]

        boxA = _translate(boxA)
        boxB = _translate(boxB)

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        if interArea == 0:
            return 0
        boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
        #iou = interArea / float(boxAArea + boxBArea - interArea)
        iou = interArea/boxBArea
        #print(iou)
        return iou

    @pyqtSlot(int)
    def zoom_handler(self, ZoomLevel):
        """
        Range is 129mm to 4.3mm
        When Zoom Level is 0, it is = 129mm
        When Zoom level is 10, it is = 4.3mm
        Values in between might not be linear so adjust from there
        """
        zoom_dict = {0:129, 1:116.53, 2:104.06, 3:91.59, 4: 79.12,5:66.65, 6:54.18, 7:41.71, 8:29.24, 9:16.77, 10:4.3}
        self.focal_length = zoom_dict.get(ZoomLevel)

    @pyqtSlot(int)
    def gamma_slider_values(self, gamma):
        self.gamma = gamma / 10

    @pyqtSlot(int)
    def xmin_e_val(self, xminE):
        self.xminE = xminE / 10

    @pyqtSlot(int)
    def ymin_e_val(self, yminE):
        self.yminE = yminE / 10

    @pyqtSlot()
    def reset_tracker(self):
        self.base_feature = None
        self.tracker = None
        self.f_tracker = None

    @pyqtSlot(bool)
    def detect_autozoom_state(self, state):
        self.autozoom_state = state

    @pyqtSlot(bool)
    def detect_face_lock_state(self, state):
        self.face_lock_state = state

    @pyqtSlot(int)
    def detect_zoom_state(self, state):
        self.zoom_state = state

    @pyqtSlot(bool)
    def detect_ytrack_state(self, state):
        self.y_trackState = state

def main():
    app = QApplication(sys.argv)
    style_file = QFile("styling/dark.qss")
    style_file.open(QFile.ReadOnly | QFile.Text)
    stream = QTextStream(style_file)
    app.setStyleSheet(stream.readAll())
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
