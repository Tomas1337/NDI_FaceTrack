import cv2, os, io, sys, time, requests, json, pickle, asyncio, websockets, logging
import numpy as np
from json import dumps
from PIL import Image
from fastapi.testclient import TestClient
from pathlib import Path
from PySide2.QtCore import QObject, QDataStream, QByteArray, QIODevice
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE
jpeg = TurboJPEG()
from fastapi import WebSocket
import pytest

sys.path.insert(0, '.')
from TrackingServer_FastAPI import app
from payloads import PipeClient_Parameter_Payload, Server_Payload, Image_Payload, Parameter_Payload
from pipeclient import PipeClient


def get_video_path():
    paths =  list(Path("test/").glob("*.mp4"))
    for path in paths:
        yield path 

TEST_IMAGE_ENDPOINT = "http://127.0.0.1:8000/tracking/predict_camera_move"
TEST_VIDEO_ENDPOINT = "http://127.0.0.1:8000/tracking/predict_camera_moves"
IMAGE_NAME = 'test/test_jpg.jpg'
IMAGE_HEIGHT = 360
IMAGE_WIDTH = 640
TOLERANCE = 5

class Base64Encoder(json.JSONEncoder):
    # pylint: disable=method-hidden
    def default(self, o):
        if isinstance(o, bytes):
            return b64encode(o).decode()
        return json.JSONEncoder.default(self, o)

def test_send_image():
    #read a picture
    img = img_orig = Image.open(IMAGE_NAME)

    #resize to correct dimensions 
    img = img_orig.resize((int(IMAGE_WIDTH), int(IMAGE_HEIGHT)))
    img = np.array(img)
    image_bytes = jpeg.encode(img)

    #Prepare payloads
    frame_payload = Image_Payload(frame = image_bytes)
    parameter_payload = Parameter_Payload()

    #Send parameter payload first
    client = TestClient(app)
    with client.websocket_connect("/ws") as websocket:
        
        #Send over parameter payload
        websocket.send_bytes(pickle.dumps(parameter_payload))
        response = websocket.receive_json()

        #parse response
        if response == 'None':
            pass
        assert (response == 'None')
        
        #Send over image payload
        websocket.send_bytes(pickle.dumps(frame_payload))
        response = websocket.receive_json()

        #Assert the response
        assert_server_response(response)
        
        #parse_response
        print('Response is {response}')
    return

def test_send_video(num_of_frames=100, show_video = False):
    """
    Run a test by sending a simulated live video to the websocket and expecting the proper format received back
    """
    client = TestClient(app)
    video = cv2.VideoCapture(str(list(get_video_path())[0]))
    resize_frame_shape = (IMAGE_WIDTH,IMAGE_HEIGHT)
    start = 0

    with client.websocket_connect("/ws") as websocket:

        #Send custom Parameters
        parameter_payload = Parameter_Payload()
        websocket.send_bytes(pickle.dumps(parameter_payload))
        response = websocket.receive_json()

        assert (response == 'None')

        while video.isOpened():
            #Get Video Frame
            retval, image = video.read()
            image = cv2.resize(image, resize_frame_shape)

            #Encode
            image_bytes = jpeg.encode(image)
            frame_payload = Image_Payload(frame = image_bytes)
            websocket.send_bytes(pickle.dumps(frame_payload))

            if show_video:
                cv2.imshow('frame',image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            #Receive and Decode
            response = websocket.receive_json()
            print(f'Response from server is {response}')
            
            #Assert the response
            assert_server_response(response)

            if start >= num_of_frames:
                break
            start += 1

        websocket.close()
        print('Websocket is closed')

def assert_server_response(response):

    if type(response) == str:
        response = json.loads(response)

    expected = list(Server_Payload().__dict__.keys())
    actual = list(response.keys())

    expected.sort()
    actual.sort()

    assert (len(actual) == len(expected))
    assert (all([a == b for a, b in zip(actual, expected)]))

def on_message(wsapp, message):
    print(message)


if __name__=="__main__":
    #test_send_image()
    #test_send_video()
    #test_send_video_qt()
    # asyncio.get_event_loop().run_until_complete(
    #     test_send_video_qt_2(uri = 'ws://127.0.0.1:8000/qt_ws'))

    pass