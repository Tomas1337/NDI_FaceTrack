import unittest
from pytestqt import qtbot
import os, sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from FaceTrackInterface import main
from FaceTrackInterface import MainWindow
from tool.info_logging import add_logger
from PySide2.QtTest import QTest
from PySide2.QtWidgets import QApplication
from tool.pyqtkeybind import keybinder
from config import CONFIG
from FaceTrackInterface import parse_args

#import ptvsd

def test_defaults(qtbot):
    
    parser = parse_args(['-i', '2','--enable_console','False'])
    args = vars(parser)

    form = MainWindow(args=args)
    qtbot.addWidget(form)

    assert form.gui.zoom_slider.value() ==  0
    assert form.gui.zoom_slider.minimum() == 0
    assert form.gui.zoom_slider.maximum() == 11

    assert form.gui.gamma_slider.value() == CONFIG.getint('camera_control', 'gamma_default')
    assert form.gui.gamma_slider.minimum() == (CONFIG.getint('camera_control', 'gamma_minimum'))
    assert form.gui.gamma_slider.maximum() == (CONFIG.getint('camera_control', 'gamma_maximum'))

    assert form.gui.x_minE_slider.value() == CONFIG.getint('camera_control', 'horizontal_error_default')
    assert form.gui.x_minE_slider.minimum() == (CONFIG.getint('camera_control', 'horizontal_error_minimum'))
    assert form.gui.x_minE_slider.maximum() == (CONFIG.getint('camera_control', 'horizontal_error_maximum'))

    assert form.gui.y_minE_slider.value() == CONFIG.getint('camera_control', 'vertical_error_default')
    assert form.gui.y_minE_slider.minimum() == (CONFIG.getint('camera_control', 'vertical_error_minimum'))
    assert form.gui.y_minE_slider.maximum() == (CONFIG.getint('camera_control', 'vertical_error_maximum'))

    assert form.gui.face_track_button.isEnabled() == False
    assert form.gui.y_enable_button.isEnabled() == False
    assert form.gui.azoom_lost_face_button.isEnabled() == False
    assert form.gui.gamma_slider.isEnabled() == False
    assert form.gui.zoom_slider.isEnabled() == False
    assert form.gui.reset_track_button.isEnabled() == False
    assert form.gui.y_minE_slider.isEnabled() == False
    assert form.gui.x_minE_slider.isEnabled() == False
    assert form.gui.adv_options_group.isChecked() == False

    