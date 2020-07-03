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
import ptvsd
from deepsort import deepsort_rbc
from deep_sort.deep_sort.nn_matching import _cosine_distance as cosine_distance
from tool.qrangeslider import QRangeSlider
from tool.qslider_text import QCustomSlider
from tool.custom_widgets import *

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
        self.gui.show()

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

        self.face_track_signal.connect(self.face_detector.face_track)

        self.gui.reset_track_button.clicked.connect(self.face_detector.reset_tracker)
        self.gui.azoom_lost_face_button.clicked.connect(self.face_detector.detect_autozoom_state)
        self.gui.y_enable_button.clicked.connect(self.face_detector.detect_ytrack_state)
        self.gui.x_speed_slider.startValueChanged.connect(self.face_detector.xspeed_min_slider_values)
        self.gui.x_speed_slider.endValueChanged.connect(self.face_detector.xspeed_max_slider_values)
        self.gui.y_speed_slider.startValueChanged.connect(self.face_detector.yspeed_min_slider_values)
        self.gui.y_speed_slider.endValueChanged.connect(self.face_detector.yspeed_max_slider_values)
        self.gui.x_minE_slider.valueChanged.connect(self.face_detector.xmin_e_val)
        self.gui.y_minE_slider.valueChanged.connect(self.face_detector.ymin_e_val)
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
        self.label_status = QLabel('Created by: __________', self)
        
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
        self.reset_track_button.setFixedHeight(40)

        #Y-Axis Tracking
        self.y_enable_button = QToggleButton('Y-Axis Tracking')
        self.y_enable_button.setCheckable(True)
        self.y_enable_button.setChecked(True)
        self.y_enable_button.setFixedHeight(100)
        self.y_enable_button.setDisabled(True)

        #Lost Auto Zoom Out Buttons
        self.azoom_lost_face_button = QToggleButton('Auto-Find Lost')
        self.azoom_lost_face_button.setCheckable(True)
        self.azoom_lost_face_button.setChecked(True)
        self.azoom_lost_face_button.setFixedHeight(100)
        self.azoom_lost_face_button.setDisabled(True)

        #Tracking position buttons
        self.tracking_button_group = QButtonGroup()
        self.tracking_button_group.addButton(self.left_track_button)
        self.tracking_button_group.addButton(self.center_track_button)
        self.tracking_button_group.addButton(self.right_track_button)
        self.tracking_button_group.setExclusive(True)

        #Speed Sliders
        x_speed_label = QLabel()
        x_speed_label.setText('X Min/Max Speed:')
        self.x_speed_slider = QRangeSlider()
        self.x_speed_slider.show()
        self.x_speed_slider.setRange(15, 25)
        self.x_speed_slider.setDrawValues(True)
        
        y_speed_label = QLabel()
        y_speed_label.setText('Y Min/Max Speed:')
        self.y_speed_slider = QRangeSlider()
        self.y_speed_slider.show()
        self.y_speed_slider.setRange(10,20)
        self.y_speed_slider.setDrawValues(True)

        #Minimum Error Slider
        x_minE_label = QLabel()
        x_minE_label.setText('Minimum X-Error:')
        self.x_minE_slider = QSlider()
        self.x_minE_slider.setOrientation(Qt.Horizontal)
        self.x_minE_slider.setValue(13)
        y_minE_label = QLabel()
        y_minE_label.setText('Minimum Y-Error:')
        self.y_minE_slider = QSlider()
        self.y_minE_slider.setOrientation(Qt.Horizontal)
        self.y_minE_slider.setValue(10)

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
        secondary_controls_layout.addWidget(self.reset_track_button)
        secondary_controls_layout.setSpacing(5)
        self.reset_track_button.setMinimumWidth(300)
        toggle_controls_layout = QHBoxLayout()
        toggle_controls_layout.addWidget(self.azoom_lost_face_button)
        toggle_controls_layout.addWidget(self.y_enable_button)
        toggle_controls_layout.setSpacing(7)
        #toggle_controls_layout.setContentsMargins(0,10,0,0)
        secondary_controls_layout.addLayout(toggle_controls_layout)
        controls_layout.addLayout(secondary_controls_layout,2,0)
        
        #Advanced Options
        adv_options_layout = QGridLayout()
        adv_options_group = QGroupBox('Advanced Controls')
        adv_options_group.setStyleSheet("QGroupBox {border-style: solid; border-width: 1px; border-color: grey; text-align: left; font-weight:bold; padding-top: 5px;} QGroupBox::title {right:120px; bottom : 6px;margin-top:4px;}")
        adv_options_group.setCheckable(True)
        adv_options_group.setChecked(False)
        adv_options_layout.setSpacing(7)
        adv_options_layout.addWidget(x_speed_label, 1,1)
        adv_options_layout.addWidget(y_speed_label, 2,1)
        adv_options_layout.addWidget(x_minE_label, 3,1)
        adv_options_layout.addWidget(y_minE_label, 4,1)
        adv_options_layout.addWidget(self.x_speed_slider,1,2)
        adv_options_layout.addWidget(self.y_speed_slider,2,2)
        adv_options_layout.addWidget(self.x_minE_slider,3,2)
        adv_options_layout.addWidget(self.y_minE_slider,4,2) 
        adv_options_layout.addWidget(self.reset_default_button,5,2) 
        adv_options_group.setLayout(adv_options_layout)
        controls_layout.addWidget(adv_options_group,2,1)

        layout.addStretch(1)
        layout.addWidget(self.label_status)
        
        self.setFixedSize(700, 750)

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
        self.x_speed_slider.setStart(15)
        self.x_speed_slider.setEnd(25)
        self.y_speed_slider.setStart(10)
        self.y_speed_slider.setEnd(20)
        self.x_minE_slider.setValue(13)
        self.y_minE_slider.setValue(13)

    def enable_controls(self):
        self.face_track_button.setEnabled(True)
        self.reset_track_button.setEnabled(True)
        self.y_enable_button.setEnabled(True)
        self.azoom_lost_face_button.setEnabled(True)

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

class csv_save(object):
    def __init__(self,parent = None):
        import csv
        self.csvFile = open('sample.csv', 'w')
        self.field_names = ['X_Error', 'Y_Error','X_Speed','Y_Speed', 'ObjX', 'ObjY', 'Frame_Count'] 
        self.writer = csv.writer(self.csvFile)

    def update(self, X_Error, Y_Error,X_Speed,Y_Speed, ObjX, ObjY, Frame_Count):
        self.writer.writerow([X_Error, Y_Error,X_Speed,Y_Speed, ObjX, ObjY, Frame_Count])
        self.writer.writerow([])
        self.csvFile.flush()

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
        for i in range(1,3):
            ndi.recv_ptz_pan_tilt_speed(self.ndi_recv, Xspeed, Yspeed)
        #ndi.recv_ptz_pan_tilt_speed(self.ndi_recv, 0, 0)

    @pyqtSlot(float)
    def zoom_camera_control(self, ZoomValue):
        ndi.recv_ptz_zoom_speed(self.ndi_recv, ZoomValue)

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
        self.check_frames_num  = 60
        self.matching_threshold = 0.02
        self.base_feature = None
        self.btrack_ok = False
        self.deepsort = deepsort_rbc()

        #Logging purposes
        self.writer = csv_save()

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

        #Object Detectors
        self.face_obj = FastMTCNN()
        self.body_obj = Yolov3()

        #Slider and button Values
        self.y_trackState = True
        self.yminE = 0.13
        self.xminE = 0.13
        self.yspeed_min = 0.10 
        self.yspeed_max = 0.20
        self.xspeed_min = 0.15 
        self.xspeed_max = 0.25
        self.ZoomValue = 0.0
        self.autozoom_state = True
        self.zoom_state = 0


    def face_tracker(self, frame, body_coords):
        ptvsd.debug_this_thread()

        if not self.f_tracker is None:
            tic = time.time()
            ok, position = self.f_tracker.update(frame)

            #Refresh face detector every N Frames
            if self.frame_count % 240 == 0 and self.btrack_ok is True:
                print('Refreshing Face Detector ')
                self.f_tracker = None
                return []
            
            elif self.frame_count % 600 == 0 and self.btrack_ok is False:
                self.f_tracker = None

            elif ok:
                x = int(position[0])
                y = int(position[1])
                w = int(position[2])
                h = int(position[3])
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 1)
                x,y,w,h = [0 if i < 0 else i for i in [x,y,w,h]]
                self.f_track_count = 0
                self.face_coords = x,y,w,h
                return x,y,w,h
                
            else:
                if self.f_track_count > 5:
                    print('Lost face') 
                    self.info_status.emit('Refreshing Face Detector')     
                    self.f_tracker = None
                    return []
                else:
                    self.f_track_count += 1
                    return []

        elif len(body_coords) <= 0 or body_coords == None:
            return []

        elif self.f_tracker is None:
            #Detect a face
            self.face_coords = self.face_obj.update(frame, body_coords)

            if len(self.face_coords) > 0:
                x,y,w,h = self.face_coords
            else:
                return []

            #Start a face tracker
            self.f_tracker = cv2.TrackerCSRT_create()
            self.f_tracker.init(frame, (x,y,w,h))
            print('Initiating a face tracker')
            return x,y,w,h

    def body_tracker(self, frame, centerX, centerY):
        ptvsd.debug_this_thread()
        
        #Check if there is a tracker already. If None:
        if self.tracker is None:
            #Detect Objects YOLO
            (idxs, boxes, _, _, classIDs, confidences) = self.body_obj.update(frame, (centerX, centerY))
            print('Running a YOLO')

            if len(boxes) == 0:
                print('No detections. Returning')
                return []
    
            #Asks if there is base_feature(that came from the previous tracker), hence to focus on on the strongest detection
            elif self.base_feature is None:
                print('here')
                winner = confidences.index(max(confidences))
                x,y,w,h = [int(i) for i in boxes[winner]]
                x,y,w,h = [0 if i < 0 else i for i in [x,y,w,h]]
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            
                #Extract features of the object
                #self.base_feature, _ = self.deepsort.extract_features_only(frame,(x,y,w,h))
            
            #Ask if there is a tracked face inside the predicted bounding box, if there is, choose that
            elif not self.f_tracker is None:
                #Check the overlaps.
                for i, h in enumerate(boxes):
                    if self.overlap_metric(h, self.face_coords) >= 0.75:
                        x,y,w,h = [int(p) for p in boxes[i]]
                        x,y,w,h = [0 if p < 0 else p for p in [x,y,w,h]]
                        self.tracker = cv2.TrackerMedianFlow_create()
                        self.tracker.init(frame, (x,y,w,h))
                        return x,y,w,h

                #If no overlaps, return the first
                x,y,w,h = boxes[0]
                return x,y,w,h
            
            #If there is a present base feature already, you have to try and reID the person
            else:
            #Extract the features of the all the crops
                try:
                    boxes_n = [boxes[i] for i in idxs[0].tolist()]            #First filter out the detections that have a NMS overlap which is indicated by idxs
                except NameError:
                    boxes_n = boxes
                
                test_features_ls = []

                #Extract features of each detection box and append to (test_features_ls)
                for i in boxes_n:
                    _x,_y,_w,_h = i
                    _x,_y,_w,_h  = [0 if p < 0 else p for p in i]
                    #print("Coords: {}".format(_x,_y,_w,_h))
                    test_features, _ = self.deepsort.extract_features_only(frame,(_x,_y,_w,_h))
                    test_features = test_features.reshape(1,2048)
                    test_features_ls.append(test_features)
                
                #create the cost matrix between the base_feature and the features that come from the detecion boxes
                t_f = np.array(test_features_ls).reshape(len(test_features_ls),2048)
                cost_matrix = cosine_distance(self.base_feature, t_f)

                #print('Cosine Matrix of RE-ID-ing:{}'.format(cost_matrix))
                if np.amin(cost_matrix) <= self.matching_threshold*0.75 and self.tracking_confirm <= 5:
                    self.tracking_confirm += 1

                #Get the index of the lowest cost(best matching) if the lowest is less thant the matching threshold
                elif np.amin(cost_matrix) <= self.matching_threshold*0.75 and self.tracking_confirm >= 5:
                    winner = divmod(cost_matrix.argmin(), cost_matrix.shape[1])[1]
                    self.tracking_confirm = 0
                else:
                    return []

                #If you cannot find a suitable match
                if not 'winner' in locals():
                    return []
                elif winner is None:
                    print("Cannot find match")
                    return []
                
                x,y,w,h = boxes_n[winner]
            
            #Start the body tracker for the given xywh
            self.tracker = cv2.TrackerMedianFlow_create()
            self.tracker.init(frame, (x,y,w,h))
            return x,y,w,h

        #If theres a tracker already            
        elif not self.tracker is None:
            tic = time.time()

            self.btrack_ok, position = self.tracker.update(frame)

            if self.btrack_ok:
                x = int(position[0])
                y = int(position[1])
                w = int(position[2])
                h = int(position[3])
                x,y,w,h = [0 if i < 0 else i for i in [x,y,w,h]]
                #print("b_tracker coords: {}".format((x,y,w,h)))
                # boxes = np.multiply(boxes,(1/resize))
                # [x,y,w,h] = boxes.astype(int)

            else:
                print('Tracking Fail')
                return []
        
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,255), 2)

            #After X frames, run a check if that object is still the same
            if self.frame_count%self.check_frames_num == 0:
                self.tracker = None
                # active_target_feature, _ = self.deepsort.extract_features_only(frame, (x,y,w,h))
                # cost_matrix = cosine_distance(self.base_feature, active_target_feature)
                # #Measure the cosine distance between hthe original feature, and the features of the current object
                # min_val = np.amin(cost_matrix)
                # if min_val <= self.matching_threshold:
                #     print("Checking present tracker: Match Success with Cost of: {}".format(min_val))
                #     self.lost_t = 0
                #     #Add to the pool of base features if within adding threshold(=0.5 * matching_threshold)
                #     if (min_val > self.matching_threshold*0.5) & (min_val < self.matching_threshold):
                #         self.base_feature = np.concatenate((self.base_feature, active_target_feature))
                #         self.base_feature = self.base_feature[:3] #Save only 3 features
                #         print("Adding to base features")
                    
                # else:
                #     print("Current feature is not a Match. The cost matrix is {}".format(min_val))
                #     if self.lost_t >= 2:
                #         #self.base_feature = None
                #         self.tracker = None
                #         self.lost_t = 0
                #         print('Cannot find match. Resetting Body Feature')
                #     else:
                #         self.lost_t +=1

            return x,y,w,h
            

    @pyqtSlot(list)
    def face_track(self, _list):
        ptvsd.debug_this_thread()
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

        #Trackers
        tic1 = time.time()
        body_coords = self.body_tracker(frame, centerX, centerY)
        tic2 = time.time()
        face_coords = self.face_tracker(frame, body_coords)
        print("Body Coordinates took: {:.3f}s. Face Coordinates took {:.3f}s".format(tic2-tic1, time.time() - tic2))
        self.auto_zoom_handler(self.zoom_state, frame, face_coords, body_coords)
         
        #If the overlap is less than 0.5, Rerun Object detection.
        if not (len(body_coords) > 0 and len(face_coords) > 0):
            pass
        else:
            self.overlap = self.overlap_metric(body_coords, face_coords)
            if self.overlap <= 0.05:
                if self.overlap_counter > 120:
                    print('Overlap Body Reset triggered')
                    self.tracker = None
                    self.overlap_counter = 0
                elif self.overlap_counter > 60:
                    print('Overlap Face Reset triggered')
                    #self.tracker = None
                    self.f_tracker = None
                else: 
                    self.overlap_counter +=1

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
            except ValueError:
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
            if self.lost_tracking > 100 and self.lost_tracking < 500 and self.autozoom_state: #and not self.base_feature is None:
                print('AutoZoom State: {}'.format(self.autozoom_state))
                print("Initiating lost track sequence with zoom")
                self.info_status.emit("Lost Tracking Sequence Secondary")
                self.CameraZoomControlSignal.emit(-0.7)

            elif self.lost_tracking < 15:
                objX = self.last_loc[0]
                objY = self.last_loc[1]
                self.lost_tracking += 1
                print('Lost tracking. Going to last known location of object')
                self.info_status.emit("Lost Tracking Sequence Initial")
                self.CameraZoomControlSignal.emit(0.0)

            else:
                objX = centerX
                objY = centerY
                print('Lost object. Centering')
                self.info_status.emit("Lost Object. Please manually re-center subject")
                self.lost_tracking += 1
                self.CameraZoomControlSignal.emit(0.0)
                if self.lost_tracking < 500 and self.lost_tracking%100:
                    self.tracker = None

        
        cv2.circle(frame,(centerX, centerY), 3, (255,255,255), 2)
        #Convert to qimage then send to display to GUI
        self.image = self.get_qimage(frame)
        self.DisplayVideoSignal.emit(self.image)

        ## CAMERA CONTROL ##
        Xerror = (centerX - objX)/W
        Yerror = (centerY - objY)/H
        
        if self.prev_X_Err == None or self.prev_Y_Err == None:
            self.prev_X_Err = Xerror
            self.prev_Y_Err = Yerror

        Xspeed = self.error_speed_translation(Xerror, self.prev_X_Err, RMin = self.xminE, TMin = self.xspeed_min, TMax = self.xspeed_max)
        Yspeed = self.error_speed_translation(Yerror, self.prev_Y_Err, RMin = self.yminE, TMin = self.yspeed_min, TMax = self.yspeed_max, penalty = 0.50)
        self.prev_X_Err = Xerror
        self.prev_Y_Err = Yerror
        
        if self.y_trackState is False:
            Yspeed = 0
             
        self.CameraControlSignal.emit(Xspeed, Yspeed)
        #print("Camera speed: {} {}".format(Xspeed, Yspeed))
        #self.writer.update(Xerror, Yerror,Xspeed,Yspeed, objX, objY, self.frame_count)
        self.frame_count += 1

    def auto_zoom_handler(self, state, frame, face_coords, body_coords):
        ptvsd.debug_this_thread()
        W,H = frame.shape[:2]

        #Dicitionary for face/body scales
        zoom_dict={0:0, 1:0.80, 2:0.30, 3:0.50, 4: 0.60}

        if state == 1 and len(body_coords) > 0:
            w,h = body_coords[2], body_coords[3]
            curr_body_percent = 1 - (H-h)/H
            target = zoom_dict.get(state)
            error = target - curr_body_percent
            print("State: {}, Curr body Percent: {}, error {}, target {}, height{}".format(state, curr_body_percent, error, target, h))
        
        elif state >= 2 and len(face_coords) > 0:
            w,h = face_coords[2], face_coords[3]
            curr_face_percent = 1 - (H-h)/H
            print("Current f_prop: {}".format(curr_face_percent))
            target = zoom_dict.get(state)
            error = target - curr_face_percent
            print("State: {}, Curr body Percent: {}, error {}, target {}, height{}".format(state, curr_face_percent, error, target, h))
        else: return

        zoom_speed = self.error_speed_translation(error, TMin = 0.2, TMax= 0.5)
        self.CameraZoomControlSignal.emit(zoom_speed)

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

    def error_speed_translation(self, Error, prev_err = None, RMin = 0.15, RMax = 1.0, TMin = 0.05, TMax = 0.80, penalty = 0.15):
        '''
        #RMin  Minimum error for Speed Adjustment, also means that it will default to 0 if the error is < 15 %
        RMax  Maximum error #100%
        TMin  Minium Scaling #This means it has a TMin% speed
        TMax  Maximum Scaling #This means it has a max TMax% speed
        '''
        if not prev_err is None:
            penalty = 0.20 * (prev_err - Error)
            #print("Penalty: {}".format(penalty))
        else:
            penalty = 0

        if Error <= RMin -0.01 and Error >= (-1 * RMin) + 0.01:
            Speed = 0
        
        else:
            #Check if Negative:
            if Error < 0:
                Error = abs(Error)
                Speed = (((Error - RMin) / (RMax - RMin)) * (TMax - TMin)) + TMin - penalty
                Speed = Speed * -1
            else:
                Speed = (((Error - RMin) / (RMax - RMin)) * (TMax - TMin)) + TMin - penalty

        return Speed

    @pyqtSlot(int)
    def xspeed_min_slider_values(self, xspeed_min):
        self.xspeed_min = xspeed_min / 100

    @pyqtSlot(int)
    def xspeed_max_slider_values(self, xspeed_max):
        self.xspeed_max = xspeed_max / 100

    @pyqtSlot(int)
    def yspeed_min_slider_values(self, yspeed_min):
        self.yspeed_min = yspeed_min / 100

    @pyqtSlot(int)
    def yspeed_max_slider_values(self, yspeed_max):
        self.yspeed_max = yspeed_max / 100

    @pyqtSlot(int)
    def xmin_e_val(self, xminE):
        self.xminE = xminE / 100

    @pyqtSlot(int)
    def ymin_e_val(self, yminE):
        self.yminE = yminE / 100

    @pyqtSlot()
    def reset_tracker(self):
        self.deepsort.reset_tracker()
        self.base_feature = None
        self.tracker = None
        self.f_tracker = None

    @pyqtSlot(bool)
    def detect_autozoom_state(self, state):
        self.autozoom_state = state

    @pyqtSlot(int)
    def detect_zoom_state(self, state):
        self.zoom_state = state

    @pyqtSlot(bool)
    def detect_ytrack_state(self, state):
        self.y_trackState = state



if __name__ == '__main__':
    app = QApplication(sys.argv)
    style_file = QFile("styling/dark.qss")
    style_file.open(QFile.ReadOnly | QFile.Text)
    stream = QTextStream(style_file)
    app.setStyleSheet(stream.readAll())
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())