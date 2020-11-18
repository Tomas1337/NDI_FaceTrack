from PyQt5.QtCore import QTextStream, QFile, QDateTime, QSize, Qt, QTimer,QRect, QThread, QObject, pyqtSignal,pyqtSlot, QRunnable, QByteArray, QBuffer, QIODevice
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox,
        QGridLayout, QGroupBox, QHBoxLayout, QLabel, QBoxLayout,
        QPushButton, QButtonGroup, QGraphicsView,
        QSlider, QStyleFactory,
        QVBoxLayout, QWidget, QAbstractButton, QMainWindow, QAction, QMenu,
        QStyle, QSpacerItem, QSizePolicy)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QBrush, QFont, QPen, QPalette, QColor, QIcon
from face_tracking.objcenter import *
from ndi_camera import ndi_camera
import numpy as np
import NDIlib as ndi
import cv2, dlib, time, styling, sys
from tool.custom_widgets import *
from face_tracking.camera_control import *
from config import CONFIG
from PipeClient import PipeClient
import keyboard, os, struct, requests, warnings, sys
import requests
from collections import namedtuple
import struct

class MainWindow(QMainWindow):
    signalStatus = pyqtSignal(str)
    track_type_signal = pyqtSignal(int)

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
        self.setFixedSize(700, 700)

    def _connectSignals(self):
        self.signalStatus.connect(self.gui.updateStatus)
        self.track_type_signal.connect(self.face_detector.set_track_type)
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

        #Connect worker signals
        self.worker.signalStatus.connect(self.gui.updateStatus)
        self.worker.ptz_object_signal.connect(self.vid_worker.read_video)
        self.worker.ptz_list_signal.connect(self.populateSources)
        self.worker.info_status.connect(self.gui.updateInfo)
        self.worker.enable_controls_signal.connect(self.gui.enable_controls)
        
        self.vid_worker.FaceFrameSignal.connect(self.face_detector.serverTransact)
        self.vid_worker.DisplayNormalVideoSignal.connect(self.gui.setImage)
        self.vid_worker.FPSSignal.connect(self.gui.updateFPS)

        self.face_detector.CameraZoomControlSignal.connect(self.vid_worker.zoom_camera_control)
        self.face_detector.DisplayVideoSignal.connect(self.gui.setImage)
        self.face_detector.CameraControlSignal.connect(self.vid_worker.camera_control)
        self.face_detector.CameraControlSignal.connect(self.gui.update_speed)
        self.face_detector.info_status.connect(self.gui.updateInfo)
        self.face_detector.signalStatus.connect(self.gui.updateStatus)

        self.gui.reset_track_button.clicked.connect(self.face_detector.reset_tracker)
        self.gui.azoom_lost_face_button.clicked.connect(self.face_detector.detect_autozoom_state)
        self.gui.y_enable_button.clicked.connect(self.face_detector.detect_ytrack_state)
        self.gui.gamma_slider.valueChanged.connect(self.face_detector.gamma_slider_values)
        self.gui.x_minE_slider.valueChanged.connect(self.face_detector.xmin_e_val)
        self.gui.y_minE_slider.valueChanged.connect(self.face_detector.ymin_e_val)  
        self.gui.zoom_slider.valueChanged.connect(self.vid_worker.zoom_handler)
        
        self.gui.face_track_button.clicked.connect(self.vid_worker.detect_face_track_state)
        self.gui.face_track_button.toggled.connect(self.face_detector.pipeStart)

        self.gui.reset_default_button.clicked.connect(self.gui.reset_defaults_handler)
        self.gui.home_pos.mouseReleaseSignal.connect(self.face_detector.getTrackPosition)

        
    def forceWorkerQuit(self):
        if self.worker_thread.isRunning():
            self.worker_thread.terminate()
            self.worker_thread.wait()

        if self.vid_worker_thread.isRunning():
            self.vid_worker_thread.terminate()
            self.vid_worker_thread.wait()

    def keyPressEvent(self, event):
        super(MainWindow, self).keyPressEvent(event)
        print('pressed from MainWindow: ', event.key())
        key_dict = {49:0, 50:1, 51:2}
        try:
            self.track_type_signal.emit(key_dict[event.key()])
        except KeyError:
            pass
        
    
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
        self.video_frame.setFixedHeight(360)
        self.video_frame.setMinimumWidth(682)
        self.video_frame.setAutoFillBackground(True)
        self.video_frame.setStyleSheet("background-color:#000000;")
        self.video_frame.setAlignment(Qt.AlignCenter)
        self.video_frame.setMargin(10)

        #Home Position Draggable
        self.home_pos = GraphicView(self.video_frame)

        #Info Panel
        self.info_panel = QLabel('No Signal',self)
        self.info_panel.setFont(QFont("Arial", 24, QFont.Bold))
        self.info_panel.setAlignment(Qt.AlignCenter)
        self.info_panel.setStyleSheet("background-color:#000000;")
        self.info_panel.setMargin(10)

        self.reset_track_button = QResetButton(self)
        self.reset_track_button.setDisabled(True)
        self.reset_track_button.setMinimumWidth(300)

        #Y-Axis Tracking
        self.y_enable_button = QToggleButton('Vertical Tracking')
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

        #Gamma Sliders
        gamma_label = QLabel()
        gamma_label.setText('Speed Sensitivity:')
        self.gamma_slider = QSlider()
        self.gamma_slider.setOrientation(Qt.Horizontal)
        self.gamma_slider.setValue((CONFIG.getint('camera_control', 'gamma_default')))
        self.gamma_slider.setTickInterval(10)
        self.gamma_slider.setMinimum((CONFIG.getint('camera_control', 'gamma_minimum')))
        self.gamma_slider.setMaximum((CONFIG.getint('camera_control', 'gamma_maximum')))

        #Minimum Error Threshold Slider
        x_minE_label = QLabel()
        x_minE_label.setText('Horizontal Threshold:')
        self.x_minE_slider = QSlider()
        self.x_minE_slider.setOrientation(Qt.Horizontal)
        self.x_minE_slider.setMinimum((CONFIG.getint('camera_control', 'horizontal_error_minimum')))
        self.x_minE_slider.setMaximum((CONFIG.getint('camera_control', 'horizontal_error_maximum')))
        self.x_minE_slider.setValue((CONFIG.getint('camera_control', 'horizontal_error_default')))
        y_minE_label = QLabel()
        y_minE_label.setText('Vertical Threshold:')
        self.y_minE_slider = QSlider()
        self.y_minE_slider.setMinimum((CONFIG.getint('camera_control', 'vertical_error_minimum')))
        self.y_minE_slider.setMaximum((CONFIG.getint('camera_control', 'vertical_error_maximum')))
        self.y_minE_slider.setOrientation(Qt.Horizontal)
        self.y_minE_slider.setValue((CONFIG.getint('camera_control', 'vertical_error_default')))

        #Zoom Slider
        zoom_slider_label = QLabel()
        zoom_slider_label.setText('ZOOM:')
        zoom_slider_label.setFont(QFont("Arial", 16))
        self.zoom_slider = QSlider()
        self.zoom_slider.setOrientation(Qt.Horizontal)
        self.zoom_slider.setValue(0)
        self.zoom_slider.setTickInterval(11)
        self.zoom_slider.setMinimum(0)
        self.zoom_slider.setMaximum(11)
        self.zoom_slider.setDisabled(True)

        #Reset To Default Button
        self.reset_default_button = QPushButton('Reset Defaults', self)

        #FPS Label
        self.fps_label = QLabel(self)
        self.fps_label.setAlignment(Qt.AlignRight)

        #Speed Vector Label
        self.speed_label = QLabel(self)
        self.speed_label.setAlignment(Qt.AlignCenter)

        ## LAYOUTING ##
        layout = QVBoxLayout(self)
        vid_layout = QVBoxLayout(self)
        
        #vid_layout.setContentsMargins(0,0,0,0)
        controls_layout = QGridLayout(self)
        controls_layout.setSpacing(10)
        layout.addLayout(vid_layout)
        layout.setAlignment(Qt.AlignCenter)

        vid_layout.addWidget(self.info_panel)
        vid_layout.addWidget(self.video_frame)
        vid_layout.setAlignment(self.video_frame, Qt.AlignCenter)
        
        vid_layout.setSpacing(0)

        layout.addLayout(controls_layout)
        controls_layout.addWidget(self.face_track_button, 0,0,1,-1)
        layoutFacePosTrack = QHBoxLayout(self)
        layoutFacePosTrack.setSpacing(5)
        controls_layout.setAlignment(Qt.AlignCenter)
        controls_layout.addLayout(layoutFacePosTrack,1,0,1,-1)

        #Additional Options
        secondary_controls_layout = QVBoxLayout()
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(zoom_slider_label)
        zoom_layout.addWidget(self.zoom_slider)
        secondary_controls_layout.addLayout(zoom_layout)
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

        bottom_info_layout = QHBoxLayout()
        bottom_info_layout.addWidget(self.label_status)
        bottom_info_layout.addWidget(self.speed_label)
        bottom_info_layout.addWidget(self.fps_label)
        layout.addLayout(bottom_info_layout)

    @pyqtSlot(str)
    def updateStatus(self, status):
        self.label_status.setText(status)

    @pyqtSlot(float)
    def updateFPS(self, fps):
        if fps < 12:
            self.fps_label.setStyleSheet('color:red')
        else:
            self.fps_label.setStyleSheet('color:white')
        self.fps_label.setText(f'{int(round(fps))} FPS')

    @pyqtSlot(float, float)
    def update_speed(self, xVel, yVel):
        self.speed_label.setText(f'X:{round(xVel,2)} Y:{round(yVel, 2)}')

    @pyqtSlot(str)
    def updateInfo(self, status):
        self.info_panel.setText(status)
        
    @pyqtSlot(QImage)
    def setImage(self, image):
        img = image.scaled(640, 360, Qt.KeepAspectRatio)
        self.video_frame.setPixmap(QPixmap.fromImage(img))

    def reset_defaults_handler(self, state):
        self.gamma_slider.setValue((CONFIG.getint('camera_control', 'gamma_default')))
        self.x_minE_slider.setValue((CONFIG.getint('camera_control', 'horizontal_error_default')))
        self.y_minE_slider.setValue((CONFIG.getint('camera_control', 'vertical_error_default')))

    def enable_controls(self):
        self.face_track_button.setEnabled(True)
        self.reset_track_button.setEnabled(True)
        self.y_enable_button.setEnabled(True)
        self.azoom_lost_face_button.setEnabled(True)
        self.zoom_slider.setEnabled(True)

class WorkerObject(QObject):
    """
    Class to handle the finding and connecting of NDI Sources/Cameras
    """
    signalStatus = pyqtSignal(str)
    ptz_list_signal = pyqtSignal(list)
    ptz_object_signal = pyqtSignal(object)
    info_status = pyqtSignal(str)
    enable_controls_signal = pyqtSignal()

    def __init__(self, parent=None):
        super(self.__class__, self).__init__(parent)
        self.ndi_cam = None
    
    @pyqtSlot()
    def findSources(self):
        self.ndi_cam = ndi_camera()
        self.signalStatus.emit('Searching for PTZ cameras')
        (ptz_list, sources) = self.ndi_cam.find_ptz()
        print("PTZ List: {}".format(ptz_list))
        self.ptz_names = [sources[i].ndi_name for i in ptz_list]
        self.signalStatus.emit('Idle')
        self.ptz_list_signal.emit(self.ptz_names)

    @pyqtSlot(int)
    def connect_to_camera(self, cam_num):
        self.signalStatus.emit('Connecting to camera') 
        ndi_recv = self.ndi_cam.camera_connect(cam_num)
        self.signalStatus.emit('Connected to {}'.format(self.ptz_names[cam_num]))
        self.ptz_object_signal.emit(ndi_recv)
        self.info_status.emit('Signal: {}'.format(self.ptz_names[cam_num]))
        self.enable_controls_signal.emit()

class Video_Object(QObject):
    """
    Handles the reading and displaying of video.
    Since we want the video object to be in-sync with the camera signals, 
    we put the under the same class therefore on the same thread.
    """
    PixMapSignal = pyqtSignal(QImage)
    FaceFrameSignal = pyqtSignal(np.ndarray)
    DisplayNormalVideoSignal = pyqtSignal(QImage)
    FPSSignal = pyqtSignal(float)
    
    def __init__(self,parent=None):
        super(self.__class__, self).__init__(parent)
        self.face_track_state = False
        self.frame_count = 1
        self.skip_frames = 1
        self.keypress = False

    @pyqtSlot(object)
    def read_video(self, ndi_object):
        FRAME_WIDTH = 640
        FRAME_HEIGHT = 360
        self.ndi_recv = ndi_object
        fps_start_time = time.time()
        diplsay_time_counter = 1
        fps_counter = 0
        

        while True:
            t,v,_,_ = ndi.recv_capture_v2(self.ndi_recv, 1)
            
            if t == ndi.FRAME_TYPE_VIDEO:
                self.frame_count += 1   
                frame = v.data
                frame = frame[:,:,:3]
                if (frame.shape[0] != FRAME_HEIGHT) or (frame.shape[1] != FRAME_WIDTH):
                    warnings.warn(f'Original frame size is:{frame.shape}')
                resize_frame_shape = (640,360)
                frame = cv2.resize(frame, resize_frame_shape)

                #Code to process the GUI events before proceeding. This is for writing the bitmap to the picturebox
                QApplication.processEvents()

                camera_move_speed = CONFIG.getfloat('camera_control', 'camera_move_speed')
                camera_zoom_speed = CONFIG.getfloat('camera_control', 'camera_zoom_speed')

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
                    self.FaceFrameSignal.emit(frame) 
                ndi.recv_free_video_v2(self.ndi_recv, v)

                fps_counter += 1
                if (time.time() - fps_start_time) > diplsay_time_counter:
                    fps = fps_counter/ (time.time()-fps_start_time)
                    self.FPSSignal.emit(fps)
                    fps_counter = 0
                    fps_start_time = time.time()
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
        """
        Function to send out the X-Y Vectors to the camera directly
        The loop helps the control of how many times the vectors are sent in one call
        This provides a tuning effect for the camera
        Args:
            Xspeed (float): X-Vector to send to camera
            Yspeed (float): Y-Vector to send to camera
        """
        for i in range(1,2):
            ndi.recv_ptz_pan_tilt_speed(self.ndi_recv, Xspeed, Yspeed)
        #ndi.recv_ptz_pan_tilt_speed(self.ndi_recv, 0, 0)

    @pyqtSlot(float)
    def zoom_camera_control(self, ZoomValue):
        ndi.recv_ptz_zoom_speed(self.ndi_recv, ZoomValue)

    @pyqtSlot(int)
    def zoom_handler(self, ZoomLevel):
        ndi.recv_ptz_zoom(self.ndi_recv, ZoomLevel/10)

class FaceDetectionWidget(QObject):
    DisplayVideoSignal = pyqtSignal(QImage)
    CameraControlSignal = pyqtSignal(float, float)
    CameraZoomControlSignal = pyqtSignal(float)
    BoundingBoxSignal = pyqtSignal(int, int, int, int)
    signalStatus = pyqtSignal(str)
    info_status = pyqtSignal(str)
    

    def __init__(self, parent=None):
        super().__init__(parent)
        FRAME_WIDTH = 640
        FRAME_HEIGHT = 360
        self.pipeClient = PipeClient()
        self.gamma = float(CONFIG.getint('camera_control', 'gamma_default'))/10
        self.xMinE = float(CONFIG.getint('camera_control', 'horizontal_error_default'))/10
        self.yMinE = float(CONFIG.getint('camera_control', 'vertical_error_default'))/10
        self.y_trackState = 1
        self.autozoom_state = 1
        self.ZoomValue = 0.0
        self.center_coords = (FRAME_WIDTH//2, FRAME_HEIGHT//2)
        self.reset_trigger = False
        self.track_type = 0

    @pyqtSlot(np.ndarray)
    def serverTransact(self, frame):
        """
        Does a read and write to Server through a pipe
        Args:
          frame (np.ndarray): image frame sent to the server,
        """
        #Sending
        frame_bytes =  frame.tobytes()
        plA = struct.pack('ffffff???i', 
            self.center_coords[0], self.center_coords[1], 
            self.gamma, self.xMinE, self.yMinE, self.ZoomValue,
            self.y_trackState, self.autozoom_state, self.reset_trigger, self.track_type)
        payload =  plA + b'frame' + frame_bytes
        self.reset_trigger = False
        self.pipeClient.writeToPipe(payload = payload)

        #Receiving
        r = self.pipeClient.readFromPipe()
        if len(r) <= 0:
            print('Response from pipe is empty')
        else:
            vectors, boundingBox = r.split(b'split')
            vectors = struct.unpack('ff', vectors)

            #Response Handling
            #Camera Control Response
            self.CameraControlSignal.emit(vectors[0], vectors[1])
            try:
                bB = struct.unpack('iiii', boundingBox)
                #Reformat boundingBox from x,y,x1,y2 to x,y,w,h
                boundingBox = (bB[0],bB[1],bB[2]-bB[0],bB[3]-bB[1])
                self.BoundingBoxSignal.emit(boundingBox[0],boundingBox[1],boundingBox[2],boundingBox[3])
            except struct.error:
                boundingBox = None

            self.displayFrame(frame, boundingBox)

    
    @pyqtSlot(bool)
    def pipeStart(self, state):
        if state is True:
            self.pipeClient.pipeRequest("http://127.0.0.1:5000/api/start_pipe_server")
        else:
            self.pipeClient.pipeClose()


    def displayFrame(self, frame, boundingBox):
        if boundingBox is None:
            pass
        else:
            x,y,w,h = boundingBox
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 1)
        
        qimage = self.get_qimage(frame)
        self.DisplayVideoSignal.emit(qimage)
        
    def get_qimage(self, image):
        """
        Turns a numpy Image array into a qImage fit for displaying
        Args:
            image: image of type numpy array

        Returns:
            image: QImage
        """
        height, width, _ = image.shape
        bytesPerLine = 3 * width
        image = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        image = image.rgbSwapped()
        return image

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
        self.xMinE = xminE / 10

    @pyqtSlot(int)
    def ymin_e_val(self, yminE):
        self.yMinE = yminE / 10

    @pyqtSlot()
    def reset_tracker(self):
        self.reset_trigger = True

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

    @pyqtSlot(int, int)
    def getTrackPosition(self, xVel, yVel):
        self.center_coords = (xVel, yVel)
        print(f'coordinates are {self.center_coords}')

    @pyqtSlot(int)
    def set_track_type(self, track_type):
        self.track_type = track_type

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
