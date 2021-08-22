import cv2, os, io, sys, time, requests, json, pickle, logging, pytest, uvicorn, random
import numpy as np
from json import dumps
from PIL import Image
from contextlib import contextmanager
from fastapi.testclient import TestClient
from pathlib import Path
from PySide2 import QtCore, QtWebSockets, QtNetwork
from PySide2.QtCore import QUrl, QCoreApplication, QTimer, QDataStream, Slot, Signal, QThread
from PySide2.QtWidgets import QApplication, QMainWindow
from threading import Thread
from multiprocessing import Process
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE
from fastapi import WebSocket 

sys.path.insert(0, '.')
from TrackingServer_FastAPI import app
from tool.payloads import *
jpeg = TurboJPEG()

def get_video_path():
    paths =  list(Path("test/").glob("*.mp4"))
    for path in paths:
        yield path

class Client_WS():
    def __init__(self, video_object):
        self.video_object = video_object
        self.video_object.FrameSignal.connect(self.send_frame_qt)
        self.video_object.VideoEndSignal.connect(self.close)
        self.ws = QtWebSockets.QWebSocket("",QtWebSockets.QWebSocketProtocol.Version13,None)
        self.ws.error.connect(self.error)
        self.ws.connected.connect(self.isConnected)
        self.ws.connected.connect(self.send_paramaters_qt)
        self.ws.open(QUrl("ws://127.0.0.1:8000/qt_ws"))
        self.frame_count = 0
        self.connected_state = False

    @Slot()
    def isConnected(self):
        print('Websocket is connected')

    @Slot()
    def send_paramaters_qt(self):
        """
        Run a test by sending a simulated live video to the websocket
        """
        #Prepare Datastream for Writing
        content = QtCore.QByteArray()
        writeStream = QtCore.QDataStream(content, QtCore.QIODevice.WriteOnly)

        #Build the paramater payload
        #This is where you can set parameters for the Tracker
        parameter_payload = Parameter_Payload()
        self.parameter_payload_loading(parameter_payload, writeStream)
        
        #Send custom Parameters
        self.ws.sendBinaryMessage(content)

    #@Slot
    def send_random_paramaters_qt(self):
        """
        Run a test by sending a simulated live video to the websocket
        """
        #Prepare Datastream for Writing
        content = QtCore.QByteArray()
        writeStream = QtCore.QDataStream(content, QtCore.QIODevice.WriteOnly)

        #Build the paramater payload
        #This is where you can set parameters for the Tracker
        parameter_payload = Parameter_Payload()

        #Set Test Payload Parameters
        random.seed(42)
        parameter_payload.target_coordinate_x = random.randint(0,640)
        parameter_payload.target_coordinate_y = random.randint(0,320)
        parameter_payload.track_type = random.randint(0,2)
        parameter_payload.gamma = random.random()
        parameter_payload.xMin = random.random()
        parameter_payload.yMinE = random.random()
        parameter_payload.zoom_value = random.random()
        parameter_payload.y_track_state = bool(random.getrandbits(1))
        parameter_payload.autozoom_state = bool(random.getrandbits(1))
        parameter_payload.reset_trigger = bool(random.getrandbits(1))

        self.parameter_payload_loading(parameter_payload, writeStream)
        
        #Send custom Parameters
        self.ws.sendBinaryMessage(content)


    @Slot(np.ndarray)
    def send_frame_qt(self, frame):
        """
        Send a video frame through the QWebsocket
        """
        #Prepare Datastream for Writing
        resize_frame_shape = (640,360)
        content = QtCore.QByteArray()
        writeStream = QtCore.QDataStream(content, QtCore.QIODevice.WriteOnly)
        
        if frame.shape != resize_frame_shape:
            image = cv2.resize(frame, resize_frame_shape)
        else:
            image = frame

        #Encode
        image_bytes = jpeg.encode(image)
        writeStream.writeQString('image')
        #writeStream << 'image'
        writeStream << image_bytes

        #Send
        self.ws.sendBinaryMessage(content)
        self.frame_count += 1
        

    def error(self, error_code):
        print("error code: {}".format(error_code))
        print(self.ws.errorString())

    def close(self):
        self.ws.close()

    def parameter_payload_loading(self, parameter_payload, writeStream):
        writeStream.writeQString('parameter')
        for key, value in parameter_payload.__dict__.items():
            if isinstance(value, int):
                writeStream.writeInt32(value)
            elif isinstance(value, float):
                writeStream.writeFloat(value)
            elif isinstance(value, bool):
                writeStream.writeBool(value)
        return writeStream

class MainWindow(QMainWindow):
    """Main Window
    Main Window is where the video_player and client objects are created
    Serves as the mother Window
    """
    def __init__(self, parent = None):
        super(MainWindow, self).__init__(parent)
        self._create_threads()
        self._connect_signals()
    
    def _connect_signals(self):
        self.client_ws.ws.connected.connect(self.video_player.read_video)
        self.client_ws.ws.textMessageReceived.connect(self.print_tracking_results)
        self.video_player.VideoEndSignal.connect(self.close)

    @Slot(str)
    def print_tracking_results(self, message):
        #data = json.loads(message)
        #print(f'Tracking results received: {data}')
        print(f'Tracking message received: {message}')

    def _create_threads(self):
        self.video_player = Video_Object()
        self.client_ws = Client_WS(self.video_player)
        self.video_player_thread = QThread(self)
        self.video_player.moveToThread(self.video_player_thread)
        self.video_player_thread.start()

    def closeEvent(self, event):
        self.video_player_thread.quit()

class Video_Object(QtCore.QObject):
    """
    Handles the reading and displaying of video.
    Since we want the video object to be in-sync with the camera signals, 
    this object should be running under the same thread as the websocket client to achieve synchronousity
    """
    FrameSignal = Signal(np.ndarray)
    VideoEndSignal = Signal()

    def __init__(self,parent=None):
        super(self.__class__, self) .__init__(parent)
        self.end_frame = 1000

    @Slot()
    def test_slot(self):
        print('Test slot initiated at video player')

    @Slot()
    def read_video(self):
        self.read_video_flag = True
        video = cv2.VideoCapture(str(list(get_video_path())[0]))
        frame_count = 0
        
        while video.isOpened():
            ret, frame = video.read()
            if ret:
                pass
            else:
                print('Stopped the video')
                break

            frame_count += 1   
            resize_frame_shape = (640,360)
            frame = cv2.resize(frame, resize_frame_shape)

            #Code to process the GUI events before proceeding.
            QApplication.processEvents()
            
            #Sends out a signal with a image frame
            self.FrameSignal.emit(frame)
            
            #Informational - Display the video being sent
            cv2.imshow('Sending',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if frame_count >= self.end_frame:
                break
        
        self.VideoEndSignal.emit()


#Test Convenience Functions
def test_qt_websocket_client_handler():
    main()

def run_server():
    uvicorn.run(app)

@pytest.fixture
def server():
    proc = Process(target=run_server, args=(), daemon=True)
    proc.start() 
    yield
    proc.kill() # Cleanup after test

def main():
    q_app = QApplication(sys.argv)
    main = MainWindow()
    q_app.exec_()

if __name__=="__main__":
    main()




    