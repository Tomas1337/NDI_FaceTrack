__author__ = "Tomas Lastrilla"
__version__ = "0.1.1"

from PyQt5.QtCore import QSize, Qt, QMimeData
from PyQt5.QtWidgets import QGridLayout, QPushButton, QSizePolicy, QLabel, QVBoxLayout, QWidget, QApplication
from PyQt5.QtGui import QPainter, QColor, QFont, QIcon, QDrag, QPixmap
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


class DraggableLabel(QLabel):
    def __init__(self, *args, **kwargs):
        QLabel.__init__(self, *args, **kwargs)
        self.setAcceptDrops(True)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_start_position = event.pos()

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.LeftButton):
            return
        if (event.pos() - self.drag_start_position).manhattanLength() < QApplication.startDragDistance():
            return
        drag = QDrag(self)
        mimedata = QMimeData()
        mimedata.setText(self.text())
        drag.setMimeData(mimedata)
        pixmap = QPixmap(self.size())
        painter = QPainter(pixmap)
        painter.drawPixmap(self.rect(), self.grab())
        painter.end()
        drag.setPixmap(pixmap)
        drag.setHotSpot(event.pos())
        drag.exec_(Qt.CopyAction | Qt.MoveAction)

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event):
        pos = event.pos()
        text = event.mimeData().text()
        self.setText(text)
        event.acceptProposedAction()