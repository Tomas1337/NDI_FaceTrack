import unittest
from pytestqt import qtbot
import os, sys, time
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from FaceTrackInterface import main
from FaceTrackInterface import MainWindow
from FaceTrackInterface import parse_args
from tool.logging import add_logger
from PySide2.QtTest import QTest
from PySide2.QtWidgets import QApplication
from PySide2.QtCore import Qt
from tool.pyqtkeybind import keybinder
from config import CONFIG
from pyautogui import press
from tool.logging import add_logger
#import ptvsd
import pytest
import win32com.client
import keyboard
from itertools import groupby


logger = add_logger()
# @pytest.fixture
# def application_startup():
#     #keybinder.init()
#     dummy_args = {'name': None, 'id': 1, 'logging': True}
#     form = MainWindow(args=dummy_args)
#     return form

@pytest.fixture
def application_startup():
    parser = parse_args(['-i','2'])
    args = vars(parser)

    form = MainWindow(args=args)
    return form

def connect_to_camera_sequence(qtbot, form):
    form.sources.aboutToShow.emit()
    with qtbot.waitSignal(form.worker.ptz_list_signal, timeout=5000) as blocker:
        pass

    form.sources.actions()[0].trigger()
    with qtbot.waitSignal(form.worker.enable_controls_signal, timeout=5000) as blocker:
        blocker.connect(form.worker.ptz_object_signal)
        blocker.connect(form.worker.enable_controls_signal)
        blocker.connect(form.vid_worker.DisplayNormalVideoSignal)
    #qtbot.mouseClick(form.gui.face_track_button, Qt.LeftButton)

def info_status_check(status):
    valid_statuses = ['Tracking Face', 'Tracking Body']
    return (status in valid_statuses)

def wasd_w_move(signalX, signalY):
    return (signalX, signalY) == (0.0,0.3)
def wasd_a_move(signalX, signalY):
    return (signalX, signalY) == (0.3,0.0)
def wasd_s_move(signalX, signalY):
    return (signalX, signalY) == (0.0,-0.3)
def wasd_d_move(signalX, signalY):
    return (signalX, signalY) == (-0.3,0.0)
def wasd_stop(signalX, signalY):
    return (signalX, signalY) == (0.0,0.0)
def camera_moving(signalX, signalY):
    if signalX > 0.0 or signalY > 0.0:
        return True
    else:
        return False

def test_wasd_keys(qtbot, application_startup):
    "Test not being reliable"
    form = application_startup
    form.show()
    qtbot.addWidget(form)

    #2 consecutive signals calling for move and stop.
    signals = [form.worker.camera_control_sent_signal, form.worker.camera_control_sent_signal]
    #W
    callbacks = [wasd_w_move, wasd_stop]
    with qtbot.waitSignals(signals, raising=False, check_params_cbs=callbacks) as blocker:
        qtbot.keyClick(form.gui, Qt.Key_W)  

    #A
    callbacks = [wasd_a_move, wasd_stop]
    with qtbot.waitSignals(signals, raising=False, check_params_cbs=callbacks) as blocker:
        qtbot.keyClick(form.gui, Qt.Key_A) 

    #S
    callbacks = [wasd_s_move, wasd_stop]
    with qtbot.waitSignals(signals, raising=False, check_params_cbs=callbacks) as blocker:
        qtbot.keyClick(form.gui, Qt.Key_S)   

    #D
    callbacks = [wasd_s_move, wasd_stop]
    with qtbot.waitSignals(signals, raising=False, check_params_cbs=callbacks) as blocker:
        qtbot.keyClick(form.gui, Qt.Key_D)   
    
    #WASD where the camera is connected
    connect_to_camera_sequence(qtbot, form)
    signals = [form.worker.camera_control_sent_signal, form.worker.camera_control_sent_signal]
    #W
    callbacks = [wasd_w_move, wasd_stop]
    with qtbot.waitSignals(signals, raising=False, check_params_cbs=callbacks) as blocker:
        qtbot.keyClick(form.gui, Qt.Key_W)  

    #A
    callbacks = [wasd_a_move, wasd_stop]
    with qtbot.waitSignals(signals, raising=False, check_params_cbs=callbacks) as blocker:
        qtbot.keyClick(form.gui, Qt.Key_A) 

    #S
    callbacks = [wasd_s_move, wasd_stop]
    with qtbot.waitSignals(signals, raising=False, check_params_cbs=callbacks) as blocker:
        qtbot.keyClick(form.gui, Qt.Key_S)   

    #D
    callbacks = [wasd_s_move, wasd_stop]
    with qtbot.waitSignals(signals, raising=False, check_params_cbs=callbacks) as blocker:
        qtbot.keyClick(form.gui, Qt.Key_D)  
    
def test_face_track_button(qtbot):
    id = 1
    parser = parse_args(['-i', str(id),"--name","DESKTOP-C16VMFB (VLC)"])
    args = vars(parser)
    form = MainWindow(args=args)
    form.show()
    qtbot.addWidget(form)
    #connect_to_camera_sequence(qtbot, form)

    # #Press the `Source` Menu and wait to populate
    # form.sources.aboutToShow.emit()
    # with qtbot.waitSignal(form.worker.ptz_list_signal, timeout=5000) as blocker:
    #     pass
    # assert len(form.worker.ptz_names) != 0, "Sources are empty"
    # assert form.gui.face_track_button.isEnabled() == False

    # #Press one of the PTZ source items
    # form.sources.actions()[0].trigger()
    # with qtbot.waitSignal(form.worker.enable_controls_signal, timeout=5000) as blocker:
    #     blocker.connect(form.worker.ptz_object_signal)
    #     blocker.connect(form.worker.enable_controls_signal)
    #     blocker.connect(form.vid_worker.DisplayNormalVideoSignal)

    #Simulate pressing the track button multiple times.
    qtbot.wait(5000)

    assert form.gui.face_track_button.isEnabled() == True
    assert form.gui.face_track_button.isChecked() == True
    assert form.vid_worker.face_track_state == True

    #OFF
    qtbot.mouseClick(form.gui.face_track_button, Qt.LeftButton)
    qtbot.wait(400)
    assert form.gui.face_track_button.isChecked() == False
    assert form.vid_worker.face_track_state == False

    #ON
    qtbot.mouseClick(form.gui.face_track_button, Qt.LeftButton)
    with qtbot.waitSignal(form.vid_worker_thread.eventDispatcher().awake) as blocker:
        pass
    qtbot.wait(400)
    assert form.gui.face_track_button.isChecked() == True
    assert form.vid_worker.face_track_state == True

    #OFF
    qtbot.mouseClick(form.gui.face_track_button, Qt.LeftButton)
    qtbot.wait(400)
    assert form.gui.face_track_button.isChecked() == False
    assert form.vid_worker.face_track_state == False


    #Detect only face using #2 Button
    #Detect only body using #3 Button
    #Verify if camera signals are being sent
    #Verify if zoom signals are working
    #Verifyif reset tracker works.
    #Verify functionality of vertical tracking enable
    
    #We need a better control test sample to test on

def test_tracker_stop_movement(qtbot, application_startup):
    """
    Test to check if camera movement stops (does not drift) when stopping the tracking.
    This is done by having a final (0.0, 0.0) signal sent 
    """
    form = application_startup
    form.show()
    qtbot.addWidget(form)
    connect_to_camera_sequence(qtbot, form)
    
    #Enable Tracking
    #Simulate pressing the track button.
    assert form.gui.face_track_button.isEnabled() == True
    assert form.gui.face_track_button.isChecked() == False
    assert form.vid_worker.face_track_state == False

    #Turn On through GUI
    with qtbot.waitSignal(form.vid_worker_thread.eventDispatcher().awake) as blocker:
        qtbot.mouseClick(form.gui.face_track_button, Qt.LeftButton)
    assert form.gui.face_track_button.isChecked() == True
    assert form.vid_worker.face_track_state == True

    #Make sure there is movement 
    with qtbot.waitSignal(form.worker.camera_control_sent_signal,raising = False, check_params_cb=camera_moving) as blocker:
        pass

    #Turn off the tracking through GUI, but must first send final 0,0
    signals = [form.vid_worker_thread.eventDispatcher().awake, form.worker.camera_control_sent_signal]
    callbacks = [None, wasd_stop]
    with qtbot.waitSignals(signals, raising=False, check_params_cbs=callbacks, order='strict' ) as blocker:
        qtbot.mouseClick(form.gui.face_track_button, Qt.LeftButton)

    #Make sure that the camera event is not emitted
    with qtbot.assertNotEmitted(form.worker.camera_control_sent_signal, wait =1000):
        pass
    assert form.gui.face_track_button.isChecked() == False
    assert form.vid_worker.face_track_state == False

def test_zero_movement(qtbot, application_startup):
    "Do not emit continous zeroes when not needed"
    form = application_startup
    form.show()
    qtbot.addWidget(form)

    connect_to_camera_sequence(qtbot, form)
    #Enable Tracking
    #Simulate pressing the track button.
    assert form.gui.face_track_button.isEnabled() == True
    assert form.gui.face_track_button.isChecked() == False
    assert form.vid_worker.face_track_state == False

    #Turn On through GUI
    with qtbot.waitSignal(form.vid_worker_thread.eventDispatcher().awake) as blocker:
        qtbot.mouseClick(form.gui.face_track_button, Qt.LeftButton)
    assert form.gui.face_track_button.isChecked() == True
    assert form.vid_worker.face_track_state == True

    X_speed = []
    Y_speed = []
    speed_vectors = []
    for i in range(1,10):
        with qtbot.waitSignal(form.worker.camera_control_sent_signal, raising = False, timeout=50) as blocker:
            pass
        try:
            X_speed.append(blocker.args[0])
            Y_speed.append(blocker.args[1])
            speed_vectors.append(blocker.args)
        except TypeError:
            pass

    assert not any(sum(1 for _ in g) > 2 for _, g in groupby(speed_vectors))

def test_reset_button_shortcut(qtbot, application_startup):
    form = application_startup
    form.show()

    qtbot.addWidget(form)
    connect_to_camera_sequence(qtbot, form)

    #Activate tracking
    with qtbot.waitSignal(form.vid_worker_thread.eventDispatcher().awake) as blocker:
        qtbot.mouseClick(form.gui.face_track_button, Qt.LeftButton)
    shell = win32com.client.Dispatch("WScript.Shell")
    #Press Reset Shortcut Button
    
    with qtbot.waitSignal(form.gui.reset_track_button.clicked) as blocker:
        shell.SendKeys("+%1")

def test_face_track_button_shortcut(qtbot):
    for id in range(1,4):
        parser = parse_args(['-i', str(id)])
        args = vars(parser)

        form = MainWindow(args=args)
        form.show()

        qtbot.addWidget(form)
        connect_to_camera_sequence(qtbot, form)

        #Install a native event filter to receive events from the OS
        from FaceTrackInterface import WinEventFilter
        from PySide2.QtCore import QAbstractEventDispatcher
        win_event_filter = WinEventFilter(keybinder)
        event_dispatcher = QAbstractEventDispatcher.instance()
        event_dispatcher.installNativeEventFilter(win_event_filter)

        assert form.gui.face_track_button.isEnabled() == True
        assert form.gui.face_track_button.isChecked() == False
        assert form.vid_worker.face_track_state == False

        shell = win32com.client.Dispatch("WScript.Shell")
        shell.SendKeys(f"^+{id}")

        qtbot.wait(100)

        assert form.gui.face_track_button.isChecked() == True
        assert form.vid_worker.face_track_state == True
        assert form.face_detector.track_type == 0

        shell.SendKeys(f"^+{id}")
        qtbot.wait(100)
         
        assert form.gui.face_track_button.isChecked() == False
        assert form.vid_worker.face_track_state == False
        assert form.face_detector.track_type == 0
        form.close()

def test_reset_button(qtbot):
    #TODO 
    pass

def test_reset_shortcut_button(qtbot):
    #TODO 
    pass


def test_close_application_shortcut(qtbot):
    for id in range(1,4):
        parser = parse_args(['-i', str(id)])
        args = vars(parser)

        form = MainWindow(args=args)
        form.show() 
        qtbot.addWidget(form)
        connect_to_camera_sequence(qtbot, form)

        #Install a native event filter to receive events from the OS
        from FaceTrackInterface import WinEventFilter
        from PySide2.QtCore import QAbstractEventDispatcher
        win_event_filter = WinEventFilter(keybinder)
        event_dispatcher = QAbstractEventDispatcher.instance()
        event_dispatcher.installNativeEventFilter(win_event_filter)

        assert form.isVisible() == True
        shell = win32com.client.Dispatch("WScript.Shell")

        function_key = f"{{F{id}}}"

        signals = [form.window_exit_signal, form.camera_control_signal]
        callback = [None, wasd_stop]

        with qtbot.waitSignals(signals, check_params_cbs=callback) as blocker:
            shell.SendKeys(f"^+{function_key}")
        
        #qtbot.wait(300)


def test_close_application_shortcut_hidden(qtbot):
    for id in range(1,4):
        parser = parse_args(['-i', str(id),'--enable_console','False'])
        args = vars(parser)

        form = MainWindow(args=args)
        qtbot.addWidget(form)
        connect_to_camera_sequence(qtbot, form)

        #Install a native event filter to receive events from the OS
        from FaceTrackInterface import WinEventFilter
        from PySide2.QtCore import QAbstractEventDispatcher
        win_event_filter = WinEventFilter(keybinder)
        event_dispatcher = QAbstractEventDispatcher.instance()
        event_dispatcher.installNativeEventFilter(win_event_filter)

        assert form.isVisible() == False
        shell = win32com.client.Dispatch("WScript.Shell")

        function_key = f"{{F{id}}}"

        signals = [form.window_exit_signal, form.camera_control_signal]
        callback = [None, wasd_stop]

        with qtbot.waitSignals(signals, check_params_cbs=callback) as blocker:
            shell.SendKeys(f"^+{function_key}")
        # with qtbot.waitSignal(form.window_exit_signal ,timeout=100) as blocker:
        #     shell.SendKeys(f"^+{function_key}")
        
        #qtbot.wait(300)






