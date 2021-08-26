import unittest
from pytestqt import qtbot
import os, sys, time
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from TrackingGUI import main
from TrackingGUI import MainWindow
from TrackingModule import DetectionWidget
from tool.logging import add_logger
from PySide2.QtTest import QTest
from PySide2.QtWidgets import QApplication
from PySide2.QtCore import Qt
from tool.pyqtkeybind import keybinder
from config import CONFIG
from pyautogui import press
import cv2 as cv
import pytest

@pytest.fixture
def get_video_path():
    test_path = "test"
    for file in os.listdir(test_path):
        if file.endswith(".mp4"):
            video_path = os.path.join(test_path, file)
            yield video_path

def test_face_tracker(get_video_path):
    video = cv.VideoCapture(get_video_path)
    detector = DetectionWidget()

    while video.isOpened():
        ret, frame = video.read()
        if ret:
            detector.frame_count = 20 #Frame count must be 20 to run a MTCNN.
            coords = detector.face_tracker(frame)

            if any(x<0 for x in coords):
                assert False, 'Found negative coordinate'
                break
            else:
                assert True
                break
