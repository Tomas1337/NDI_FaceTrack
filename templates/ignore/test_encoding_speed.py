import time, math, pickle, io, win32file, win32pipe, cv2, requests, configparser, pywintypes, struct, sys, os, cv2
import numpy as np
from PIL import Image
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE

from fastapi.testclient import TestClient
from TrackingServer_FastAPI import app
from pathlib import Path
sys.path.insert(0, 'tool/')
from payloads import Image_Payload

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360

jpeg = TurboJPEG()

def get_video_path():
    paths =  list(Path("test/").glob("*.mp4"))
    for path in paths:
        yield path 

def test_encode():
    """
    Run a test by encoding speed test
    """
    client = TestClient(app)
    video = cv2.VideoCapture(str(list(get_video_path())[0]))
    resize_frame_shape = (IMAGE_WIDTH,IMAGE_HEIGHT)
    start = 0
    num_of_frames = 100

    with client.websocket_connect("/ws") as websocket:
        while True:
            start_time = time.time()
            retval, image = video.read()
            image = cv2.resize(image, resize_frame_shape)

            #Encode
            encoded_image = jpeg.encode(image)
            image_payload = Image_Payload(frame=encoded_image)
            image_payload = pickle.dumps(image_payload)
            websocket.send_bytes(image_payload)
            #Decode
            #bgr_array = jpeg.decode(encoded_image)

            #print('testing speed')
            #data = websocket.receive_json()
            #print(f'data on Client is {data}')
            #print("Time {:5.2f}".format((time.time() - start_time)))
            
            if start >= num_of_frames:
                break
            start += 1
            
    print('Websocket is closed')

if __name__ == '__main__':
    test_encode()

