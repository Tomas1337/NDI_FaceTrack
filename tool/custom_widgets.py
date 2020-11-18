__author__ = "Tomas Lastrilla"
__version__ = "0.1.1"

from PyQt5.QtCore import QSize, Qt, QMimeData, QPointF, pyqtSignal
from PyQt5.QtWidgets import QGraphicsTextItem, QGridLayout, QPushButton, QSizePolicy, QLabel, QVBoxLayout, QWidget, QApplication, QGraphicsView, QGraphicsScene, QGraphicsEllipseItem
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


class MovingObject(QGraphicsTextItem):
    mouseReleaseSignal = pyqtSignal(int, int)
    def __init__(self, x ,y, scene_width, scene_height):
        super().__init__()
        self.setAcceptHoverEvents(True)
        self.setPlainText("+")
        self.setDefaultTextColor(Qt.red)
        self.setFont(QFont("Arial", 35))
        self.setTextWidth(10)
        self.bounding_width = self.boundingRect().width()
        self.bounding_height = self.boundingRect().height()
        self.setfromCenter(x, y)

        self.width = scene_width
        self.height = scene_height

    def setfromCenter(self, x, y):
        x -= int(self.bounding_width/2)
        y -= int(self.bounding_height/2)
        self.setPos(x,y)

    # mouse click event
    def mousePressEvent(self, event):
        pass
 
    def mouseMoveEvent(self, event):
        orig_cursor_position = event.lastScenePos()
        updated_cursor_position = event.scenePos()
 
        orig_position = self.scenePos()

        updated_cursor_x = updated_cursor_position.x() - orig_cursor_position.x() + orig_position.x()
        updated_cursor_y = updated_cursor_position.y() - orig_cursor_position.y() + orig_position.y()
        
        #Calibrated values
        pad_val = 22
        if updated_cursor_x > self.width - pad_val + 4:
            updated_cursor_x = self.width - pad_val  + 4
        elif updated_cursor_x < 0 - pad_val+5:
            updated_cursor_x = 0 - pad_val+5
    
        if updated_cursor_y > self.height - pad_val - 30:
            updated_cursor_y = self.height - pad_val - 30
        elif updated_cursor_y < 0 - pad_val - 7:
            updated_cursor_y = 0 - pad_val - 7

        self.setPos(QPointF(updated_cursor_x, updated_cursor_y))
 
    def mouseReleaseEvent(self, event):
        self.center_x = self.pos().x() + int(self.bounding_width/2)
        self.center_y = self.pos().y() + int(self.bounding_height/2)
        print('x: {0}, y: {1}'.format(self.center_x, self.center_y))
        self.mouseReleaseSignal.emit(self.center_x, self.center_y)

    def _getPosition(self):
        self.center_x = self.pos().x() + int(self.bounding_width/2)
        self.center_y = self.pos().y() + int(self.bounding_height/2)
        return (self.center_x, self.center_y)

class GraphicView(QGraphicsView):
    mouseReleaseSignal = pyqtSignal(int, int)
    def __init__(self, parent):
        super(GraphicView, self).__init__(parent)
 
        self.scene = QGraphicsScene()
        self.setAlignment(Qt.AlignCenter)
        self.setScene(self.scene)
        
        width = parent.parent().parent().size().width()
        height = parent.frameGeometry().height() 
        center = (width/2, height/2)
        print("Center {}".format(center))

        self.setSceneRect(0, 0, width, height)
        #self.setStyleSheet("background-color: #1c2e66;")
        self.setStyleSheet("background:transparent;")
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.moveObject = MovingObject(center[0], center[1], width, height)
        self.mouseReleaseSignal.connect(self.moveObject.mouseReleaseEvent)

        self.scene.addItem(self.moveObject)
        self.move(21,21) #Manual move since it wont center properly

    def getPosition(self):
        (center_x, center_y) = self.moveObject._getPosition()
        return (center_x, center_y)
    
    def mouseReleaseEvent(self, e):
        (center_x, center_y) = self.moveObject._getPosition()
        self.mouseReleaseSignal.emit(center_x, center_y)