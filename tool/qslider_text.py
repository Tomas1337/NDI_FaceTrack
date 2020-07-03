from PyQt5 import QtCore
from PyQt5.QtWidgets import QSlider, QWidget, QVBoxLayout, QHBoxLayout, QLabel

class QCustomSlider(QSlider):
    def __init__(self, sliderOrientation=None):
        super(QCustomSlider, self).__init__()
        self._slider = QSlider(sliderOrientation)
        #self._slider = QSlider()
        self.setLayout(QVBoxLayout())

        self._labelTicksWidget = QWidget(self)
        self._labelTicksWidget.setLayout(QHBoxLayout())
        self._labelTicksWidget.layout().setContentsMargins(0, 0, 0, 0)

        self.layout().addWidget(self._slider)
        self.layout().addWidget(self._labelTicksWidget)

    def setTickLabels(self, listWithLabels):
        lengthOfList = len(listWithLabels)
        for index, label in enumerate(listWithLabels):
            label = QLabel(str(label))
            label.setContentsMargins(0, 0, 0, 0)
            if index > lengthOfList/3:
                label.setAlignment(QtCore.Qt.AlignCenter)
            if index > 2*lengthOfList/3:
                label.setAlignment(QtCore.Qt.AlignRight)
            self._labelTicksWidget.layout().addWidget(label)

    def setRange(self, mini, maxi):
        self._slider.setRange(mini, maxi)

    def setPageStep(self, value):
        self._slider.setPageStep(value)

    def setTickInterval(self, value):
        self._slider.setTickInterval(value)

    def setTickPosition(self, position):
        self._slider.setTickPosition(position)

    def setValue(self, value):
        self._slider.setValue(value)

    def onValueChangedCall(self, function):
        self._slider.valueChanged.connect(function)
