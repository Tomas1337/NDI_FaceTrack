from PySide2.QtCore import QTextStream, QFile, QDateTime, QSize, Qt, QTimer,QRect, QThread, QObject, Signal, Slot, QRunnable, QAbstractNativeEventFilter, QAbstractEventDispatcher
from PySide2.QtWidgets import (QShortcut, QApplication, QCheckBox, QComboBox, QDateTimeEdit, QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QBoxLayout, QProgressBar, QPushButton, QButtonGroup, QSlider, QStyleFactory, QTableWidget, QTabWidget, QTextEdit, QVBoxLayout, QWidget, QAbstractButton, QMainWindow, QAction, QMenu, QStyleOptionSlider, QStyle, QSpacerItem, QSizePolicy)
from PySide2.QtGui import QKeySequence, QImage, QPixmap, QPainter, QBrush, QFont, QPen, QPalette, QColor, QIcon
from face_tracking.objcenter import *
from ndi_camera import ndi_camera
from decimal import Decimal, getcontext
from functools import partial
import numpy as np
import NDIlib as ndi
import cv2, time, styling, sys, keyboard, argparse, threading, logging
from tool.custom_widgets import *
from tool.logging import add_logger
from face_tracking.camera_control import *
from config import CONFIG
from tool.utils import str2bool
from tool.pyqtkeybind import keybinder
from bg_matting import BG_Matt
import ptvsd

class WinEventFilter(QAbstractNativeEventFilter):
    def __init__(self, keybinder):
        self.keybinder = keybinder
        super().__init__()

    def nativeEventFilter(self,eventType, message):
        ret = self.keybinder.handler(eventType, message)
        return ret, 0

#Setup Logging
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
sys.excepthook = handle_exception

class MainWindow(QMainWindow):
    signalStatus = Signal(str)
    track_type_signal = Signal(int)
    face_track_signal = Signal(np.ndarray)
    preset_camera_signal = Signal()
    camera_control_signal = Signal(float, float)
    camera_control_zoom_signal = Signal(float)
    window_exit_signal = Signal()

    def __init__(self, parent = None, args = None):
        super(MainWindow, self).__init__(parent)
        
        #Initialize the GUI object
        #Create a new worker thread
        self.args = args
        self.gui = WindowGUI(self, args)
        self.setCentralWidget(self.gui) 
        self.createThreads()
        self.createMenuBar()

        #Make any cross object connections
        self._connectSignals()
        title = "NDI FaceTrack"
        self.setWindowTitle(title)
        self.setFixedSize(700, 660)
        self.gui.show()
        if args['name'] is not None:
            self.preset_camera_signal.emit()

        #GLOBAL SHORTCUT#
        keybinder.init()
        track_button_sc_prefix = CONFIG['shortcut']['track_button_sc_prefix']
        exit_button_sc_prefix =  CONFIG['shortcut']['exit_button_sc_prefix']
        reset_button_sc_prefix =  CONFIG['shortcut']['reset_button_sc_prefix']
        keybinder.register_hotkey(self.winId(),(f'{track_button_sc_prefix}+')+ str(args['id']), self.gui.face_track_button_click)
        keybinder.register_hotkey(self.winId(),(f'{exit_button_sc_prefix}')+ str(args['id']), self.close)
        keybinder.register_hotkey(self.winId(),(f'{reset_button_sc_prefix}+')+ str(args['id']), self.gui.reset_track_button_click)

    def closeEvent(self, event):
        self.camera_control_signal.emit(0.0, 0.0)
        self.window_exit_signal.emit()
        
    def _connectSignals(self):
        self.signalStatus.connect(self.gui.updateStatus)
        self.track_type_signal.connect(self.face_detector.set_track_type)
        self.sources.aboutToShow.connect(self.camera.findSources)
        self.camera_control_signal.connect(self.camera.camera_control)
        self.camera_control_zoom_signal.connect(self.camera.camera_zoom_control)

    def createMenuBar(self):
        self.bar = self.menuBar()
        self.sources = self.bar.addMenu("Sources")

    @Slot(list)
    def populateSources(self, _list):
        self.sources.clear()
        for idx, item in enumerate(_list):
            entry = self.sources.addAction(item)
            entry.triggered.connect(self.camera.connect_to_camera) 
        entry.triggered.connect(self.vid_worker.stop_read_video)
    
 
    ### SIGNALS
    def createThreads(self):
        self.camera = CameraObject(args = self.args)
        self.camera_thread = QThread()  
        self.camera.moveToThread(self.camera_thread)
        self.camera_thread.start()
        
        self.vid_worker = Video_Object()
        self.vid_worker_thread = QThread()
        self.vid_worker.moveToThread(self.vid_worker_thread)
        
        self.face_detector = DetectionWidget()
        self.face_detector.moveToThread(self.vid_worker_thread)
        self.vid_worker_thread.start()

        self.bg_matt = BG_Matt_Object()
        self.bg_matt_thread = QThread()
        self.bg_matt.moveToThread(self.bg_matt_thread)
        self.bg_matt_thread.start()

        #Connect any worker signals
        self.camera.signalStatus.connect(self.gui.updateStatus)
        self.camera.ptz_object_signal.connect(self.vid_worker.stop_read_video)
        self.camera.ptz_object_signal.connect(self.vid_worker.read_video)
        self.camera.ptz_list_signal.connect(self.populateSources)
        self.camera.info_status.connect(self.gui.updateInfo)
        self.camera.enable_controls_signal.connect(self.gui.enable_controls)
        self.camera.face_track_button_click.connect(self.gui.face_track_button_click)

        self.vid_worker.FaceFrameSignal.connect(self.face_track_signal_handler)
        self.vid_worker.DisplayNormalVideoSignal.connect(self.gui.setImage)
        self.vid_worker.FPSSignal.connect(self.gui.updateFPS)
        self.vid_worker.TrackButtonShortcutSignal.connect(self.gui.face_track_button_click)
        self.vid_worker.BGMattFrameSignal.connect(self.bg_matt.forward)

        self.face_detector.CameraZoomControlSignal.connect(self.camera.camera_zoom_control)
        self.face_detector.DisplayVideoSignal.connect(self.gui.setImage)
        self.face_detector.CameraControlSignal.connect(self.camera.camera_control)
        self.face_detector.CameraControlSignal.connect(self.gui.update_speed)
        self.face_detector.info_status.connect(self.gui.updateInfo)
        self.face_detector.signalStatus.connect(self.gui.updateStatus)

        self.gui.reset_track_button.clicked.connect(self.face_detector.reset_tracker)
        self.gui.azoom_lost_face_button.clicked.connect(self.face_detector.detect_autozoom_state)
        self.gui.y_enable_button.clicked.connect(self.face_detector.detect_ytrack_state)
        self.gui.gamma_slider.valueChanged.connect(self.face_detector.gamma_slider_values)
        self.gui.x_minE_slider.valueChanged.connect(self.face_detector.xmin_e_val)
        self.gui.y_minE_slider.valueChanged.connect(self.face_detector.ymin_e_val)  
        self.gui.zoom_slider.valueChanged.connect(self.camera.zoom_handler)
        
        self.gui.face_track_button.clicked.connect(self.vid_worker.detect_face_track_state)
        self.gui.face_track_button.clicked.connect(self.face_detector.set_track_trigger)

        self.face_track_signal.connect(self.face_detector.main_track)
        self.gui.reset_default_button.clicked.connect(self.gui.reset_defaults_handler)
        self.gui.home_pos.mouseReleaseSignal.connect(self.face_detector.getTrackPosition)

        self.preset_camera_signal.connect(self.camera.connect_to_preset_camera)
        self.gui.face_track_button.clicked.connect(lambda state: self.camera.camera_control(0.0,0.0))

    def forceWorkerQuit(self):
        if self.camera_thread.isRunning():
            self.camera_thread.terminate()
            self.camera_thread.wait()

        if self.vid_worker_thread.isRunning():
            self.vid_worker_thread.terminate()
            self.vid_worker_thread.wait()

    def keyPressEvent(self, event):
        #ptvsd.debug_this_thread()
        super(MainWindow, self).keyPressEvent(event)
        logger.debug(f'pressed from MainWindow: {event.key()}')

        key_dict_track_type = {49:0, 50:1, 51:2} #Keys 1,2 and 3
        key_dict_camera_zoom = { Qt.Key_Q: -1,
            Qt.Key_E: 1}
        key_dict_camera_moves = {Qt.Key_W:(0,1), 
            Qt.Key_A:(1,0), 
            Qt.Key_S:(0,-1),
            Qt.Key_D:(-1,0)
            }

        if event.key() in key_dict_track_type:
            self.track_type_signal.emit(key_dict_track_type[event.key()])
        elif event.key() in key_dict_camera_moves and self.args['enable_console']:
            self.camera_movement(key_dict_camera_moves[event.key()])
        elif event.key() in key_dict_camera_zoom and self.args['enable_console']:
            self.camera_zoom(key_dict_camera_zoom[event.key()])

    def keyReleaseEvent(self, event):
        super(MainWindow, self).keyPressEvent(event)

        key_dict_camera_moves = {Qt.Key_W:(0,1), 
            Qt.Key_A:(1,0), 
            Qt.Key_S:(0,-1),
            Qt.Key_D:(-1,0)
            }

        key_dict_camera_zoom = { Qt.Key_Q: -1,
            Qt.Key_E: 1}

        if event.key() in key_dict_camera_moves:
            self.camera_control_signal.emit(0,0)
        elif event.key() in key_dict_camera_zoom:
            self.camera_control_zoom_signal.emit(0)
           
    def camera_movement(self, vector: tuple):
        camera_move_speed = CONFIG.getfloat('camera_control', 'camera_move_speed')
        speed_vector = tuple([camera_move_speed*x for x in vector])
        self.camera_control_signal.emit(speed_vector[0], speed_vector[1])
    
    def camera_zoom(self, vector: float):
        camera_zoom_speed = CONFIG.getfloat('camera_control', 'camera_zoom_speed')
        self.camera_control_zoom_signal.emit(vector * camera_zoom_speed)
        
    @Slot(np.ndarray)
    def face_track_signal_handler(self, frame):
        self.face_track_signal.emit(frame)

class WindowGUI(QWidget):
    def __init__(self, parent, args):
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
        
        #Reset Button
        self.reset_track_button = QResetButton(self)
        self.reset_track_button.setDisabled(True)
        self.reset_track_button.setMinimumWidth(300)

        #Face Lock Button
        self.face_lock_button = QPushButton('LOCK TO FACE')
        self.face_lock_button.setCheckable(True)

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

        #FPS Label
        self.fps_label = QLabel(self)
        self.fps_label.setAlignment(Qt.AlignRight)

        #Speed Vector Label
        self.speed_label = QLabel(self)
        self.speed_label.setAlignment(Qt.AlignCenter)

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
        self.adv_options_group = QGroupBox('Advanced Controls')
        self.adv_options_group.setStyleSheet("QGroupBox {border-style: solid; border-width: 1px; border-color: grey; text-align: left; font-weight:bold; padding-top: 5px;} QGroupBox::title {right:100px; bottom : 6px;margin-top:4px;}")
        self.adv_options_group.setCheckable(True)
        self.adv_options_group.setChecked(False)
        adv_options_layout.setSpacing(7)
        adv_options_layout.addWidget(gamma_label, 1,1)
        adv_options_layout.addWidget(x_minE_label, 2,1)
        adv_options_layout.addWidget(y_minE_label, 3,1)
        adv_options_layout.addWidget(self.gamma_slider,1,2)
        adv_options_layout.addWidget(self.x_minE_slider,2,2)
        adv_options_layout.addWidget(self.y_minE_slider,3,2) 
        adv_options_layout.addWidget(self.reset_default_button,4,2) 
        self.adv_options_group.setLayout(adv_options_layout)
        controls_layout.addWidget(self.adv_options_group,2,1)

        layout.addStretch(1)
        
        bottom_info_layout = QHBoxLayout()
        bottom_info_layout.addWidget(self.label_status)
        bottom_info_layout.addWidget(self.speed_label)
        bottom_info_layout.addWidget(self.fps_label)
        layout.addLayout(bottom_info_layout)

    @Slot(str)
    def updateStatus(self, status):
        self.label_status.setText(status)

    @Slot(float)
    def updateFPS(self, fps):
        if fps < 12:
            self.fps_label.setStyleSheet('color:red')
        else:
            self.fps_label.setStyleSheet('color:white')
        self.fps_label.setText(f'{int(round(fps))} FPS')

    @Slot(float, float)
    def update_speed(self, xVel, yVel):
        self.speed_label.setText(f'X:{round(xVel,2)} Y:{round(yVel, 2)}')

    @Slot(str)
    def updateInfo(self, status):
        self.info_panel.setText(status)

    @Slot()
    def face_track_button_click(self):
        self.face_track_button.click()

    @Slot()
    def reset_track_button_click(self):
        self.reset_track_button.click()
        
    @Slot(QImage)
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
        
class CameraObject(QObject):
    
    """
    Class to handle the finding and connecting of NDI Sources/Cameras
    """
    signalStatus = Signal(str)
    ptz_list_signal = Signal(list)
    ptz_object_signal = Signal(object)
    info_status = Signal(str)
    enable_controls_signal = Signal()
    face_track_button_click = Signal()
    camera_control_sent_signal = Signal(float, float)

    def __init__(self, parent=None, args = None):
        super(self.__class__, self).__init__(parent)
        self.ndi_cam = None
        self.args = args
    
    @Slot()
    def findSources(self):
        self.ndi_cam = ndi_camera()
        self.signalStatus.emit('Searching for PTZ cameras')
        self.ndi_cam.find_sources()
        (ptz_list, sources) = self.ndi_cam.find_ptz()
        print("PTZ List: {}".format(ptz_list))
        self.ptz_names = [sources[i].ndi_name for i in ptz_list]
        self.signalStatus.emit('Idle')
        self.ptz_list_signal.emit(self.ptz_names)

    @Slot(int)
    def connect_to_camera(self, cam_num):
        #ptvsd.debug_this_thread()
        self.signalStatus.emit('Connecting to camera') 
        self.ndi_recv = self.ndi_cam.camera_connect(ndi_name=self.sender().text())
        self.signalStatus.emit('Connected to {}'.format(self.ptz_names[cam_num]))
        self.ptz_object_signal.emit(self.ndi_recv)
        self.info_status.emit('Signal: {}'.format(self.ptz_names[cam_num]))
        self.enable_controls_signal.emit()

    @Slot()
    def connect_to_preset_camera(self):
        self.ndi_cam = ndi_camera()
        self.signalStatus.emit('Connecting to Preset camera')
        name = self.args['name']
        self.ndi_recv = self.ndi_cam.camera_connect(ndi_name=name)
        self.ptz_object_signal.emit(self.ndi_recv)
        self.info_status.emit(f'Signal: {name}')
        self.enable_controls_signal.emit()
        self.face_track_button_click.emit()

    @Slot(float, float)
    def camera_control(self, Xspeed: float, Yspeed: float):
        #ptvsd.debug_this_thread()
        """
        Function to send out the X-Y Vectors to the camera directly
        The loop helps the control of how many times the vectors are sent in one call
        This provides a tuning effect for the camera
        Args:
            Xspeed (float): X-Vector to send to camera
            Yspeed (float): Y-Vewctor to send to camera
        """
        try:
            ndi.recv_ptz_pan_tilt_speed(self.ndi_recv, Xspeed, Yspeed)
            self.camera_control_sent_signal.emit(Xspeed, Yspeed)
            #print(f'Camera Control X:{Xspeed}  Y:{Yspeed}')

        except AttributeError:
            self.camera_control_sent_signal.emit(Xspeed, Yspeed)
            logger.warning('WASD - QE does not work since there is no connected source')
            #print(f'Camera Control X:{Xspeed}  Y:{Yspeed}')
        
    @Slot(float)
    def camera_zoom_control(self, zoom_speed: float):
        ndi.recv_ptz_zoom_speed(self.ndi_recv, zoom_speed)

    @Slot(int)
    def zoom_handler(self, ZoomLevel):
        ndi.recv_ptz_zoom(self.ndi_recv, ZoomLevel/10)


class BG_Matt_Object(QObject):
    "Handles passing on the frame to the BG_matting module"
    def __init__(self, parent = None):
        super().__init__(parent)
        self.bg_matter = BG_Matt()

    @Slot(np.ndarray)
    def forward(self, frame):
        tic = time.time()
        matted_frame = self.bg_matter.predict(frame)
        toc = time.time()
        self.bg_time = round(toc-tic, 3)
        print(f"BG Processing time: {self.bg_time}s")
        cv2.imshow('BG Matterd Frame', matted_frame)
        cv2.waitKey(0)

    def get_time(self):
        return self.bg_time

#Handles the reading and displayingg of video
class Video_Object(QObject):
    """
    Handles the reading and displaying of video.
    Since we want the video object to be in-sync with the camera signals, 
    we put the under the same class therefore on the same thread.
    """
    PixMapSignal = Signal(QImage)
    FaceFrameSignal = Signal(np.ndarray)
    BGMattFrameSignal = Signal(np.ndarray)
    DisplayNormalVideoSignal = Signal(QImage)
    TrackButtonShortcutSignal = Signal()
    FPSSignal = Signal(float)
    
    def __init__(self,parent=None):
        super(self.__class__, self).__init__(parent)
        self.face_track_state = False
        self.frame_count = 1
        self.read_video_flag = True
        self.keypress = False

    @Slot()
    def stop_read_video(self):
        self.read_video_flag = False
    
    @Slot(object)
    def read_video(self, ndi_object):
        ptvsd.debug_this_thread()
        FRAME_WIDTH = 640   
        FRAME_HEIGHT = 360
        self.ndi_recv = ndi_object
        fps_start_time = time.time()
        diplsay_time_counter = 1    
        fps_counter = 0
        self.read_video_flag = True
        while self.read_video_flag:
            t,v,_,_ = ndi.recv_capture_v2(self.ndi_recv, 0)
            if t == ndi.FRAME_TYPE_VIDEO:
                self.frame_count += 1   
                frame = v.data
                frame = frame[:,:,:3]
                if (frame.shape[0] != FRAME_HEIGHT) or (frame.shape[1] != FRAME_WIDTH):
                    logger.warning(f'Frame is not within recommended dimensions:{frame.shape}. Resizing to ({FRAME_WIDTH}, {FRAME_HEIGHT})')
                resize_frame_shape = (640,360)
                frame = cv2.resize(frame, resize_frame_shape)

                if self.face_track_state == False:
                    self.display_plain_video(frame)
                    #self.BGMattFrameSignal.emit(frame)

                elif self.face_track_state == True:
                    # Note: Since the FaceFrameSignal ultimately ends to DetectionWidget which is in the same Thread as Video_Object, 
                    # This in effect blocks the video from proceeding before a frame is processed (Synchronous)
                    self.FaceFrameSignal.emit(frame)
                ndi.recv_free_video_v2(self.ndi_recv, v)

                #Code to process the GUI events before proceeding
                QApplication.processEvents()
                
                #Measuring FPS
                fps_counter += 1
                if (time.time() - fps_start_time) > diplsay_time_counter:
                    fps = fps_counter/ (time.time()-fps_start_time)
                    self.FPSSignal.emit(fps)
                    fps_counter = 0
                    fps_start_time = time.time()    
                
                self.frame_none_count = 0
            
            else:
                QApplication.processEvents()

    @Slot(bool)
    def detect_face_track_state(self, state):
        self.face_track_state = state

    def display_plain_video(self, image):
        height, width, colors = image.shape
        bytesPerLine = 3 * width
        image = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        image = image.rgbSwapped()
        self.DisplayNormalVideoSignal.emit(image)

class DetectionWidget(QObject):
    DisplayVideoSignal = Signal(QImage)
    CameraControlSignal = Signal(float, float)
    CameraZoomControlSignal = Signal(float)
    signalStatus = Signal(str)
    info_status = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        FRAME_WIDTH = 640
        FRAME_HEIGHT = 360
        self.image = QImage()
        self.last_loc = [FRAME_WIDTH//2, FRAME_HEIGHT//2]
        self.track_coords = []
        self.reset_trigger = False
        self.btrack_ok = False
        self.focal_length = 129
        self.body_y_offset_value = CONFIG.getint('home_position', 'body_vertical_home_offset')
        self.center_coords = (FRAME_WIDTH//2, FRAME_HEIGHT//2)
        self.track_type = 0
        self.track_trigger = False

        #Counters
        self.frame_count = 1
        self.lost_tracking = 0
        self.f_track_count = 0
        self.overlap_counter = 0
        self.lost_t = 0

        #Trackers
        self.b_tracker = None
        self.f_tracker = None

        #Object Detectors
        self.face_obj = FastMTCNN()
        self.body_obj = Yolo_v4TINY()

        #Slider and button Values
        self.y_trackState = True
        self.gamma = float(CONFIG.getint('camera_control', 'gamma_default'))/10
        self.xMinE = float(CONFIG.getint('camera_control', 'horizontal_error_default'))/10
        self.yMinE = float(CONFIG.getint('camera_control', 'vertical_error_default'))/10
        self.ZoomValue = 0.0
        self.autozoom_state = True
        self.face_lock_state = False
        self.zoom_state = 0
        self.face_coords = []

        #Speed Tracker
        self.x_prev_speed = None
        self.y_prev_speed = None

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
                logger.debug(f'Running a {(300/30)}s check')
                test_coords = self.face_obj.get_all_locations(frame)

                if len(test_coords) > 0: #There are detections
                    for i, j in enumerate(test_coords):
                        if self.overlap_metric(j, self.face_coords) >= 0.75:
                            x,y,w,h = j
                            break
                    return []

                else: #No detections
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
                    logger.debug('Lost face') 
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
            if self.frame_count % 20 == 0:
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
            logger.debug('Initiating a face tracker')

        self.face_coords = x,y,w,h
        return x,y,w,h

    def body_tracker(self, frame):
        #ptvsd.debug_this_thread()
        if self.b_tracker is None:
            #Detect Objects using YOLO every 1 second if No Body Tracker    
            boxes = []
            if self.frame_count % 15 == 0:
                (idxs, boxes, _, _, classIDs, confidences) = self.body_obj.update(frame)
                logger.debug('Running a YOLO')

            if len(boxes) <= 0:
                return []

            elif len(boxes) == 1:
                x,y,w,h = boxes[np.argmax(confidences)]
                x,y,w,h = [0 if i < 0 else int(i) for i in [x,y,w,h]]

            elif len(boxes) > 1 and len(self.face_coords) >= 1:
                for i, box in enumerate(boxes):
                    if self.overlap_metric(box, self.face_coords) >= 0.5:
                        x,y,w,h = [0 if i < 0 else int(i) for i in [x,y,w,h]]
                        x,y,w,h = [int(p) for p in boxes[i]]
                        break

            #Start the body tracker for the given xywh
            self.b_tracker = cv2.TrackerKCF_create()
            try:
                max_width = 100
                if w > max_width:
                    scale = max_width/w
                    x,y,w,h = self._resizeRect(x,y,w,h,scale)
                else:
                    pass
                self.b_tracker.init(frame, (x,y,w,h))
            except UnboundLocalError:
                return []
            return x,y,w,h

        #If theres a tracker already            
        elif not self.b_tracker is None:
            tic = time.time()       
            self.btrack_ok, position = self.b_tracker.update(frame)
            #print("Body Tracker update takes:{:.2f}s".format(time.time() - tic))
            if self.btrack_ok:
                x = int(position[0])
                y = int(position[1])
                w = int(position[2])
                h = int(position[3])
                x,y,w,h = [0 if i < 0 else i for i in [x,y,w,h]]

            else:
                logger.debug('Tracking Fail')
                return []
        
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,255), 2)
            return x,y,w,h

    @Slot(np.ndarray)
    def main_track(self, frame):
        ptvsd.debug_this_thread()
        """
        Begins main tracking module.
        Handles descision between face or body Tracking
        Handles sending commands to camera

        Args:
            frame (np.ndarray): Frame to track
        """
        self.track_coords = []
        self.set_focal_length(self.ZoomValue)
        start = time.time()

        if self.reset_trigger is True:
            self.reset_tracker()
            self.reset_trigger = False

        (H, W) = frame.shape[:2]
        centerX = self.center_coords[0]
        centerY = self.center_coords[1]
        objX = centerX
        objY = centerY

        #Trackers
        if self.track_type == 0:
            #Face-> Body (If cannot detect a face, try to detect a body)
            face_coords = self.face_tracker(frame)
            if len(face_coords) <= 0:
                body_coords = self.body_tracker(frame)

        elif self.track_type == 1:
            #Face Only
            face_coords = self.face_tracker(frame)
        
        elif self.track_type == 2:
            #Body Only
            body_coords = self.body_tracker(frame)

        try:
            x,y,w,h = face_coords
            self.track_coords = [x,y,x+w,y+h]
            self.info_status.emit('Tracking Face')
        except (ValueError, TypeError, UnboundLocalError) as e:
            try:
                x,y,w,h = body_coords
                self.track_coords = [x,y,x+w,y+h]
                self.info_status.emit('Tracking Body')
                self.overlap_counter = 0
            except (ValueError, UnboundLocalError) as e:
                pass
            finally:
                face_track_flag = False
        
        offset_value = int(self.body_y_offset_value * (H/100))

        #Normal Tracking
        if len(self.track_coords) > 0:
            [x,y,x2,y2] = self.track_coords
            objX = int(x+(x2-x)//2)
            objY = int(y+(y2-y)//3) if face_track_flag else int(y + offset_value)
            self.last_loc = (objX, objY)
            self.lost_tracking = 0

        #Initiate Lost Tracking sub-routine
        else:
            if self.lost_tracking > 100 and self.lost_tracking < 500 and self.autozoom_state:
                self.info_status.emit("Lost Tracking Sequence Secondary")
                self.CameraZoomControlSignal.emit(-0.7)

            elif self.lost_tracking < 20:
                objX = self.last_loc[0]
                objY = self.last_loc[1]
                self.lost_tracking += 1
                logger.debug('Lost tracking. Going to last known location of object')
                self.info_status.emit("Lost Tracking Sequence Initial")

            else:
                objX = centerX
                objY = centerY
                logger.debug('Lost object. Centering')
                self.info_status.emit("Lost Object. Recenter subject")
                self.lost_tracking += 1
                if self.lost_tracking < 500 and self.lost_tracking%100:
                    self.b_tracker = None

        #Convert to qimage then send to display to GUI
        self.image = self.get_qimage(frame)
        self.DisplayVideoSignal.emit(self.image)

        ## CAMERA CONTROL
        x_controller = PTZ_Controller_Novel(self.focal_length, self.gamma)
        x_speed = x_controller.omega_tur_plus1(objX, centerX, RMin = 0.1)

        y_controller = PTZ_Controller_Novel(self.focal_length, self.gamma)
        y_speed = y_controller.omega_tur_plus1(objY, centerY, RMin = 0.1, RMax=7.5)

        x_speed = Decimal(self._pos_error_thresholding(W, x_speed, centerX, objX, self.xMinE)).quantize(Decimal("0.0001"))
        y_speed = Decimal(self._pos_error_thresholding(H, y_speed, centerY, objY, self.yMinE)).quantize(Decimal("0.0001"))

        if self.y_trackState is False:
            y_speed = 0

        if (self.x_prev_speed == x_speed and 
                self.y_prev_speed == y_speed and x_speed <= 0.15):
            pass

        else:
            if self.sender().gui.face_track_button.isChecked() is True:
                self.CameraControlSignal.emit(x_speed, y_speed)
            else:
                self.CameraControlSignal.emit(0.0, 0.0)
            
        self.frame_count += 1
        self.x_prev_speed = x_speed
        self.y_prev_speed = y_speed
        
    def get_qimage(self, image):
        height, width, _ = image.shape
        bytesPerLine = 3 * width
        image = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        image = image.rgbSwapped()
        return image

    def set_focal_length(self, zoom_level):
        """
        Range is 129mm to 4.3mm
        When Zoom Level is 0, it is = 129mm
        When Zoom level is 10, it is = 4.3mm
        Values in between might not be linear so adjust from there
        """
        zoom_dict = {0:129, 1:116.53, 2:104.06, 3:91.59, 4: 79.12,5:66.65, 6:54.18, 7:41.71, 8:29.24, 9:16.77, 10:4.3}
        self.focal_length = zoom_dict.get(int(zoom_level))

    def _pos_error_thresholding(self,frame_dim, calc_speed, center_coord, obj_coord, error_threshold):
        """
        Function to check if the distance between the desired position and the current position is above a threshold.
        If below threshold, return 0 as the speed
        If above threshold, do nothing
        """
        error = np.abs(center_coord - obj_coord)/(frame_dim/2)
    
        if error >= error_threshold:
            pass
        else:
            calc_speed = 0
        
        return calc_speed

    def _resizeRect(self, x,y,w,h,scale):    
        x = int(x + w * (1 - scale)/2)
        y = int(y + h * (1 - scale)/2)
        w = int(w * scale)
        h = int(h * scale)
        return (x,y,w,h)

    def overlap_metric(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        # BoxA and BoxB usually comes in x,y,w,h
        # Translate into x,y,x1,y1
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
        return iou

    @Slot(int)
    def zoom_handler(self, ZoomLevel):
        """
        Range is 129mm to 4.3mm
        When Zoom Level is 0, it is = 129mm
        When Zoom level is 10, it is = 4.3mm
        Values in between might not be linear so adjust from there
        """
        zoom_dict = {0:129, 1:116.53, 2:104.06, 3:91.59, 4: 79.12,5:66.65, 6:54.18, 7:41.71, 8:29.24, 9:16.77, 10:4.3}
        self.focal_length = zoom_dict.get(ZoomLevel)

    @Slot(int)
    def gamma_slider_values(self, gamma):
        self.gamma = gamma / 10

    @Slot(int)
    def xmin_e_val(self, xminE):
        self.xMinE = xminE / 10

    @Slot(int)
    def ymin_e_val(self, yminE):
        self.yMinE = yminE / 10

    @Slot()
    def reset_tracker(self):
        self.b_tracker = None
        self.f_tracker = None

    @Slot(bool)
    def detect_autozoom_state(self, state):
        self.autozoom_state = state

    @Slot(bool)
    def detect_face_lock_state(self, state):
        self.face_lock_state = state

    @Slot(int)
    def detect_zoom_state(self, state):
        self.zoom_state = state

    @Slot(bool)
    def detect_ytrack_state(self, state):
        self.y_trackState = state

    @Slot(int)
    def set_track_type(self, track_type):
        self.track_type = track_type

    @Slot(bool)
    def set_track_trigger(self, state):
        self.track_trigger = state

    @Slot(int, int)
    def getTrackPosition(self, xVel, yVel):
        self.center_coords = (xVel, yVel)
        print(f'coordinates are {self.center_coords}')

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Argument Parsing NDI_FaceTrack')
    parser.add_argument('-n', '--name', default = None, help = "Provide the Name of the Camera to connect to Format: NameXXX (DeviceXXX)")
    parser.add_argument('-c', '--enable_console', type = str2bool, default = True, help = "Gives option to enable/disable the UI. This is useful for allowing users to just use the base tracking module. Must have --name argument")
    parser.add_argument('-l', '--logging', default = False, type = str2bool, help = "Generates a txt file for logging")
    parser.add_argument('-i', '--id', type = int, default = 1, help="Gives the instance an numerical ID. Used for shortcut assignment e.g. CTRL+1 will disable/enable Tracking on Application with ID-1")
    return parser.parse_args(argv)

def main(args = None):
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()
    
    if args is None:
        args = parse_args(sys.argv[1:])

    if args.logging is False:
        logging.disable(logging.CRITICAL)
        
    args_dict = vars(args)
    logger.debug(args_dict)

    style_name = "styling/dark.qss"
    style_path = os.path.abspath(os.path.dirname(__file__))
    style_file = QFile(os.path.join(style_path,style_name)) 
    style_file.open(QFile.ReadOnly | QFile.Text)
    stream = QTextStream(style_file)

    app.setStyleSheet(stream.readAll())
    main = MainWindow(args = args_dict)

    #Install a native event filter to receive events from the OS
    win_event_filter = WinEventFilter(keybinder)
    event_dispatcher = QAbstractEventDispatcher.instance()
    event_dispatcher.installNativeEventFilter(win_event_filter)
    
    if args.enable_console is True:
        main.show()
    else:
        if args.name is None:
            dialog = DialogBox()
            dialog(text= "Cannot continue", info_text="Please provide --name to continue when disabling UI")
            logger.error("No name provided to connect to while UI is disabled. Exiting Application")
            exit()
        else:
            logger.debug("UI is hidden")

    sys.exit(app.exec_())
    
logger = add_logger()

if __name__ == '__main__':
    main()
