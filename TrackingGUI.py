from PySide2.QtCore import QTextStream, QFile, QDateTime, QSize, Qt, QTimer,QRect, QThread, QObject, Signal, Slot, QAbstractNativeEventFilter, QAbstractEventDispatcher
from PySide2.QtWidgets import (QShortcut, QApplication, QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QBoxLayout, QProgressBar, QPushButton, QButtonGroup, QSlider, QStyleFactory, QTableWidget, QTabWidget, QTextEdit, QVBoxLayout, QWidget, QAbstractButton, QMainWindow, QAction, QMenu, QStyleOptionSlider, QStyle, QSpacerItem, QSizePolicy)
from PySide2.QtGui import QKeySequence, QImage, QPixmap, QPainter, QBrush, QFont, QIcon
from face_tracking.objcenter import *
from face_tracking.camera_control import *
from ndi_camera import ndi_camera
from tool.logging import add_logger
import numpy as np
import NDIlib as ndi
import cv2, time, styling, sys, os, struct, requests, warnings, argparse, logging, pickle
from tool.custom_widgets import *
from config import CONFIG
from tool.pipeclient import PipeClient 
from tool.payloads import *
from tool.utils import str2bool
from tool.pyqtkeybind import keybinder
#from tool.identity_assist import IdentityAssistWindow
#from turbojpeg import TurboJPEG
from TrackingServer_FastAPI import main as app_main
from multiprocessing import Process
#import ptvsd

class WinEventFilter(QAbstractNativeEventFilter):
    def __init__(self, keybinder):
        self.keybinder = keybinder
        super().__init__()

    def nativeEventFilter(self,eventType, message):
        ret = self.keybinder.handler(eventType, message)
        return ret, 0

# #Setup Logging
# def handle_exception(exc_type, exc_value, exc_traceback):
#     if issubclass(exc_type, KeyboardInterrupt):
#         sys.__excepthook__(exc_type, exc_value, exc_traceback)
#         return
#     logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
# sys.excepthook = handle_exception

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
        self.args = args
        self.gui = WindowGUI(self, args)
        self.setCentralWidget(self.gui) 
        self.createMenuBar()
        self.createThreads()

        #Make any cross object connections
        self._connectSignals()
        title = "NDI FaceTrack"
        self.setWindowTitle(title) 
        self.gui.show()
        screen_size = QApplication.primaryScreen().size()
        #1920 1080 width height (700, 660) 
        width = int(screen_size.width() * 0.365) # This should have a maximum width to account for ultra wide screens
        width = 800 if width > 800 else width
        height = int(screen_size.height() * 0.60)
        self.setFixedSize(width, height)
        self.ida_counter = 0
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
        self.sources.aboutToShow.connect(self.camera_thread.start)
        #self.aboutToQuit.connect(self.forceWorkerQuit)

    def createMenuBar(self):
        bar = self.menuBar()
        self.sources = bar.addMenu("Sources")
        self.add_ons = bar.addMenu("Add-ons")
        self.addon_identity_assist = self.add_ons.addAction("Identity Assist")
        # self.addon_identity_assist.triggered.connect(self.identity_assist_window)

    # def identity_assist_window(self):
    #     self.id_assist_window = IdentityAssistWindow(self)
    #     self.id_assist_window.show()
    #     self.vid_worker.IdentityAssistFrameSignal.connect(self.id_assist_window.get_keyframe)
    #     #self.vid_worker.IdentityAssistFrameSignal.connect(self.test_slot)
    #     self.id_assist_window.IdentityAssistEnableSignal.connect(self.vid_worker.detect_identity_assist_state)

    @Slot(np.ndarray)
    def test_slot(self, frame):
        print(f'frame received with shape of {frame.shape}')

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
        
        self.vid_worker = Video_Object()
        self.vid_worker_thread = QThread()
        self.vid_worker.moveToThread(self.vid_worker_thread)

        self.face_detector = FaceDetectionWidget()
        self.face_detector.moveToThread(self.vid_worker_thread)
        self.vid_worker_thread.start()

        #Connect worker signals
        self.camera.signalStatus.connect(self.gui.updateStatus)
        self.camera.ptz_object_signal.connect(self.vid_worker.stop_read_video)
        self.camera.ptz_object_signal.connect(self.vid_worker.read_video)
        self.camera.ptz_list_signal.connect(self.populateSources)
        self.camera.info_status.connect(self.gui.updateInfo)
        self.camera.enable_controls_signal.connect(self.gui.enable_controls)
        self.camera.face_track_button_click.connect(self.gui.face_track_button_click)

        self.vid_worker.FaceFrameSignal.connect(self.face_detector.server_transact)
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
        
    
class WindowGUI(QWidget):
    def __init__(self, parent, args):
        super(WindowGUI, self).__init__(parent)
        self.label_status = QLabel('Created by: Tomas Lastrilla', self)
        
        #Main Track Button
        self.face_track_button = QTrackingButton('TRACK')
        self.face_track_button.setCheckable(True)
        self.face_track_button.setDisabled(True)

        #Video Widgets
        self.video_frame = QLabel('',self)
        screen_size = QApplication.primaryScreen().size()
        height = screen_size.height()
        width = screen_size.width()
        width = 800 if width > 800 else width
        self.video_frame.setFixedHeight(int(height*0.33))
        self.video_frame.setMinimumWidth(int(width*0.355))   
        self.video_frame.setAutoFillBackground(True)
        self.video_frame.setStyleSheet("background-color:#;")
        self.video_frame.setAlignment(Qt.AlignCenter)
        # self.video_frame.setMargin(10)
        # Print info about the video_frame size
        print(f'Video Frame Size: {self.video_frame.size()}')

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
        self.y_enable_button = QToggleButton('Vertical Framing')
        self.y_enable_button.setCheckable(True)
        self.y_enable_button.setChecked(True)
        self.y_enable_button.setFixedHeight(int(height*0.0648))
        self.y_enable_button.setDisabled(True)

        #Lost Auto Zoom Out Buttons
        self.azoom_lost_face_button = QToggleButton('Auto-Find Lost')
        self.azoom_lost_face_button.setCheckable(True)
        self.azoom_lost_face_button.setChecked(True)
        self.azoom_lost_face_button.setFixedHeight(int(height*0.0648))
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
        layout.addLayout(vid_layout)

        controls_layout = QGridLayout(self)
        controls_layout.setSpacing(10)
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
    info_status = Signal(str    )
    enable_controls_signal = Signal()
    face_track_button_click = Signal()
    camera_control_sent_signal = Signal(float, float)

    def __init__(self, parent=None, args = None):
        super(self.__class__, self).__init__(parent)
        self.ndi_cam = None
    
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

class Video_Object(QObject):
    """
    Handles the reading and displaying of video.
    Since we want the video object to be in-sync with the camera signals, 
    we put the under the same class therefore on the same thread.
    """
    PixMapSignal = Signal(QImage)
    FaceFrameSignal = Signal(np.ndarray)
    IdentityAssistFrameSignal = Signal(np.ndarray)
    DisplayNormalVideoSignal = Signal(QImage)
    FPSSignal = Signal(float)
    
    def __init__(self,parent=None):
        super(self.__class__, self).__init__(parent)
        self.face_track_state = False
        self.frame_count = 1
        self.skip_frames = 1
        self.keypress = False
        self.identity_assist_state = False

    @Slot()
    def stop_read_video(self):
        self.read_video_flag = False
        ndi.recv_ptz_pan_tilt_speed(self.ndi_recv, 0, 0)

    @Slot(object)
    def read_video(self, ndi_object):
        #ptvsd.debug_this_thread()
        FRAME_WIDTH = 640
        FRAME_HEIGHT = 360
        self.ndi_recv = ndi_object
        fps_start_time = time.time()
        diplsay_time_counter = 1
        fps_counter = 0
        self.read_video_flag = True
        
        
        while True:
            t,v,_,_ = ndi.recv_capture_v2(self.ndi_recv, 0)
            
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

                if self.face_track_state == False:
                    self.display_plain_video(frame)
                elif self.face_track_state == True:
                    self.FaceFrameSignal.emit(frame)

                if self.identity_assist_state is True:
                    self.IdentityAssistFrameSignal.emit(frame)
                
                ndi.recv_free_video_v2(self.ndi_recv, v)

                fps_counter += 1
                if (time.time() - fps_start_time) > diplsay_time_counter:
                    fps = fps_counter/ (time.time()-fps_start_time)
                    self.FPSSignal.emit(fps)
                    fps_counter = 0
                    fps_start_time = time.time()
    @Slot(bool)
    def detect_face_track_state(self, state):
        self.face_track_state = state

    @Slot(bool)
    def detect_identity_assist_state(self, state):
        print("Identity Signal Enable received")
        self.identity_assist_state = state

    def display_plain_video(self, image):
        height, width, colors = image.shape
        bytesPerLine = 3 * width
        image = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        image = image.rgbSwapped()
        self.DisplayNormalVideoSignal.emit(image)

    @Slot(float, float)
    def camera_control(self, Xspeed, Yspeed, repeat=2):
        """
        Function to send out the X-Y Vectors to the camera directly
        The loop helps the control of how many times the vectors are sent in one call
        This provides a tuning effect for the camera
        Args:
            Xspeed (float): X-Vector to send to camera
            Yspeed (float): Y-Vector to send to camera
            Repeat (int, optional): Number of times to send the vectors to the NDI. Defaults to 2.
        """
        for i in range(1,repeat):
            #print(f'Camera Control X:{Xspeed}  Y:{Yspeed} Repeat:{i}')
            ndi.recv_ptz_pan_tilt_speed(self.ndi_recv, Xspeed, Yspeed)
        #ndi.recv_ptz_pan_tilt_speed(self.ndi_recv, 0, 0)

    @Slot(float)
    def zoom_camera_control(self, zoom_value):
        ndi.recv_ptz_zoom_speed(self.ndi_recv, zoom_value)

    @Slot(int)
    def zoom_handler(self, ZoomLevel):
        ndi.recv_ptz_zoom(self.ndi_recv, ZoomLevel/10)

class FaceDetectionWidget(QObject):
    """
    Component of the GUI Module that handles the interaction between control layer and tracking layer (to the server)
    """
    DisplayVideoSignal = Signal(QImage)
    CameraControlSignal = Signal(float, float)
    CameraZoomControlSignal = Signal(float)
    signalStatus = Signal(str)
    info_status = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        FRAME_WIDTH = 640
        FRAME_HEIGHT = 360
        self.pipeClient = PipeClient()
        self.gamma = float(CONFIG.getint('camera_control', 'gamma_default'))/10
        self.xMinE = float(CONFIG.getint('camera_control', 'horizontal_error_default'))/10
        self.yMinE = float(CONFIG.getint('camera_control', 'vertical_error_default'))/10
        self.y_track_state = 1
        self.autozoom_state = 1
        self.zoom_value = 0.0
        self.center_coords = (FRAME_WIDTH//2, FRAME_HEIGHT//2)
        self.reset_trigger = False
        self.track_type = 0
        

    @Slot(np.ndarray)
    def server_transact(self, frame):
        """
        Does a read and write to Server through a FIFO Pipe
        Args:
        frame (np.ndarray): image frame sent to the server,
        """
        #Sending using Pydantic payloads
        parameter_payload = PipeClient_Parameter_Payload(target_coordinate_x = self.center_coords[0],
                                                        target_coordinate_y = self.center_coords[1],
                                                        track_type = self.track_type,
                                                        gamma = self.gamma,
                                                        xMinE = self.xMinE,
                                                        yMinE = self.yMinE,
                                                        zoom_value = self.zoom_value,
                                                        y_track_state = self.y_track_state,
                                                        autozoom_state = self.autozoom_state,
                                                        reset_trigger = self.reset_trigger)

        self.pipeClient.writeToPipe(payload = parameter_payload.pickle_object())
        
        #image_payload = PipeClient_Image_Payload(frame = jpeg.encode(frame))
        is_success, im_buf_arr = cv2.imencode(".jpg", frame)
        byte_im = im_buf_arr.tobytes()
        
        image_payload = PipeClient_Image_Payload(frame = byte_im)
        self.reset_trigger = False
        self.pipeClient.writeToPipe(payload = image_payload.pickle_object())

        #Receiving pickled Pydantic payloads
        response_pickled = self.pipeClient.readFromPipe()
        if response_pickled == b'' or response_pickled == None:
            response = None
        else:
            response = pickle.loads(response_pickled)
    
        #print(f'Response from pipe is {response} meme')

        #Display frame
        if response and response.x is not None:
            bB = [response.x, response.y, response.w, response.h]
            boundingBox = (bB[0],bB[1],bB[2]-bB[0],bB[3]-bB[1])
            # Emit the signal of the x_velocity and y_velocity via the CameraControlSignal
            self.CameraControlSignal.emit(response.x_velocity, response.y_velocity)
        else:
            boundingBox = None
        self.displayFrame(frame, boundingBox)

        
        

    @Slot(bool)
    def pipeStart(self, state=True):
        """Requests a pipe to connect to
        Pipe name is stored in pipeClient.pipeName
        """ 
        if state is True:
            
            host=CONFIG.get('server','host') 
            port=CONFIG.getint('server','port') 
            url = f"http://{host}:{port}/api/start_pipe_server"
            print(f'Creating pipeHandle using pipeName with url of {url}')
            pipeName = self.pipeClient.pipeRequest(url)
            print(f'Creating pipeHandle using pipeName {pipeName}')
            self.pipeClient.createPipeHandle(pipeName)
           
        else:
            self.CameraControlSignal.emit(0.0,0.0)
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
        self.reset_trigger = True

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
        self.y_track_state = state

    @Slot(int, int)
    def getTrackPosition(self, xVel, yVel):
        self.center_coords = (xVel, yVel)
        print(f'coordinates are {self.center_coords}')

    @Slot(int)
    def set_track_type(self, track_type):
        self.track_type = track_type

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Argument Parsing NDI_FaceTrack')
    parser.add_argument('-n', '--name', default = None, help = "Provide the Name of the Camera to connect to Format: NameXXX (DeviceXXX)")
    parser.add_argument('-c', '--enable_console', type = str2bool, default = True, help = "Gives option to enable/disable the UI. This is useful for allowing users to just use the base tracking module. Must have --name argument")
    parser.add_argument('-l', '--logging', default = False, type = str2bool, help = "Generates a txt file for logging")
    parser.add_argument('-i', '--id', type = int, default = 1, help="Gives the instance an numerical ID. Used for shortcut assignment e.g. CTRL+1 will disable/enable Tracking on Application with ID-1")
    return parser.parse_args(argv)
    
def check_server():
    #Check if there is a current running FastAPI Server for tracking using requests
    try:
        host=CONFIG.get('server','host') 
        port=CONFIG.getint('server','port') 
        url = f"http://{host}:{port}/api/check_tracking_server"
        signature = CONFIG.get('server', 'signature')

        response = requests.post(url)
        if response.json() == signature:
            valid = True
        else:
            valid = False
    except:
        print('no server detected')
        valid = False

    finally:
        return valid
    
def main(args = None):
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()

    if args is None:
        args = parse_args(sys.argv[1:])

    if args.logging is False:
        logging.disable(logging.CRITICAL)

    #Check if server is enabled, if not, start one:
    if check_server():
        pass
    else:
        #retVal = DialogBox()
        # exit()
        # @TODO: Start the server in the background
        print('Trying to start own Server')
        fastapi_process = Process(target = app_main)
        fastapi_process.start()

            
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
#jpeg = TurboJPEG()

if __name__ == '__main__':
    main()

