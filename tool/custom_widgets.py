__author__ = "Tomas Lastrilla"
__version__ = "0.1.1"


# ---------------------------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------
# TODO
# ---------------------------------------------------------------------------------------------

"""
  - Simple button with Icon

"""
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QGridLayout, QPushButton, QSizePolicy, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPainter, QColor, QFont, QIcon
import sys


class QResetButton(QPushButton):
    def __init__(self, parent = None):
        super(QResetButton, self).__init__(parent)
        self.setText('RESET TRACKER')
        self.setFont(QFont("Open Sans", 19, QFont.Bold))
        self.setStyleSheet("QPushButton {background-color:#b45f06; border-radius: 10px; color: #e6d7c8;} QPushButton:disabled {background-color:#444444;}")

    def mousePressEvent(self, event):
        super(QResetButton, self).mousePressEvent(event)
        self.setStyleSheet("background-color:#d47e24; border-radius: 10px;")

    def mouseReleaseEvent(self, event):
        super(QResetButton, self).mouseReleaseEvent(event)
        self.setStyleSheet("background-color:#d47e24; border-radius: 10px;") if self.isChecked() else self.setStyleSheet("background-color:#b45f06; border-radius: 10px;")

class QToggleButton(QPushButton,):
    def __init__(self,top_text, parent = None):
        super(QToggleButton, self).__init__(parent)
        self.setStyleSheet("QPushButton {background-color:#38761d; border-radius: 10px; color: #e6d7c8;} QPushButton:disabled {background-color:#444444;} QPushButton:checked {background-color:#6aa84f;} QPushButton:checked:disabled {background-color:#999999;}")        #self.setText('something very long here \n and here')
        layout = QVBoxLayout(self)
        top_lbl = QLabel(str(top_text))
        top_lbl.setFont(QFont("Open Sans", 12, QFont.Bold))
        top_lbl.setAlignment(Qt.AlignCenter)
        top_lbl.setStyleSheet("background: rgba(255, 255, 255, 0);")

        self.btm_lbl = QLabel('ON')
        self.btm_lbl.setFont(QFont("Open Sans", 22, QFont.Bold))
        self.btm_lbl.setAlignment(Qt.AlignCenter)
        self.btm_lbl.setStyleSheet("background: rgba(255, 255, 255, 0);")

        layout.addWidget(top_lbl)
        layout.addWidget(self.btm_lbl)
        self.setLayout(layout)
    
    def mousePressEvent(self, event):
        super(QToggleButton, self).mousePressEvent(event)
        self.btm_lbl.setText('OFF')

    def mouseReleaseEvent(self, event):
        super(QToggleButton, self).mouseReleaseEvent(event)
        self.btm_lbl.setText('ON') if self.isChecked() else self.btm_lbl.setText('OFF')


class QTrackingButton(QPushButton):
    def __init__(self, parent = None):
        super(QTrackingButton, self).__init__(parent)
        self.setFont(QFont("Open Sans", 19, QFont.Bold))
        self.setStyleSheet("QPushButton {background-color:#38761d; border-radius: 10px; color: #e6d7c8;} QPushButton:disabled {background-color:#444444;} QPushButton:checked {background-color:#6aa84f;} QPushButton:checked:disabled {background-color:#999999;}")