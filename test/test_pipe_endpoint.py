import sys, os
import pytest, pathlib
import struct, cv2
from pathlib import Path
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE

sys.path.insert(0, 'tool/')
from tool.pipeclient import PipeClient 
from tool.payloads import PipeClient_Parameter_Payload, Server_Payload, PipeClient_Image_Payload
from TrackingServer_FastAPI import app
from multiprocessing import Process
import uvicorn




"""The TrackingGUI.py uses namedPipes to communicate between 
the Video_Object (frame sender) and the FastAPI server.

3rd Party applications may also tap into this 'namedPipe' communication 
following the code below.
1) test_pipe_endpoint_python.py uses Python exclusive Pickling for data serialization between python programs
2) test_pipe_endpoint_protobuff uses Universal protobuff data serialization between different programming languages:
https://developers.google.com/protocol-buffers
"""

FRAME_WIDTH = 640
FRAME_HEIGHT = 360

jpeg = TurboJPEG()


def assert_server_response(response):

    if type(response) == str:
        response = json.loads(response)

    expected = list(Server_Payload().__dict__.keys())
    actual = list(response.keys())

    expected.sort()
    actual.sort()

    assert (len(actual) == len(expected))
    assert (all([a == b for a, b in zip(actual, expected)]))

#TODO: This test currently requires ther server to be ran in the background. Fix so that it can run the server automatically
def test_pipe_endpoint_python(num_of_frames=100):
    url = 'http://127.0.0.1:8000/api/start_pipe_server'
    pipeClient = PipeClient()
    pipeName = pipeClient.pipeRequest(url)
    ok = pipeClient.createPipeHandle(pipeName)
    ok = pipeClient.checkPipe()
    print(f'Checked pipe with response {ok}')

    video = cv2.VideoCapture(str(list(get_video_path())[0]))
    while video.isOpened():
        ret, frame = video.read()
        resize_frame_shape = (640,360)
        frame = cv2.resize(frame, resize_frame_shape)
        frame_bytes = jpeg.encode(frame)
        if ret:
            break
        else:
            pass
        
    # Building the Payload using PipeClientPayload pydantic structure
    # You would add control parameters here
    payload_parameters  = PipeClient_Parameter_Payload().pickle_object()
    payload_image = PipeClient_Image_Payload(frame=frame_bytes)
    payload_image = payload_image.pickle_object()
    start = 0

    #Sending over the frame to the pipe
    while True:
        if ok:
            pipeClient.writeToPipe(payload_parameters) #Send to pipe the Tracking Parameters
            pipeClient.writeToPipe(payload_image) #Send to Pipe the image
            response_pickled = pipeClient.readFromPipe() #Read from Pipe
            
            if response_pickled == b'':
                response = None
            else:
                response = PipeClient.unpickle_object(response_pickled)
                assert_server_response(response.dict())

            print(f'Response from pipe is {response}')
            if start == num_of_frames:
                print('Ending video')
                break
            start += 1


def get_video_path():
    paths =  list(Path("test/").glob("*.mp4"))
    for path in paths:
        yield path 
        
def run_server():
    print("running fastapi server")
    uvicorn.run(app)

@pytest.fixture
def server():
    proc = Process(target=run_server, args=(), daemon=True)
    proc.start() 
    yield
    proc.kill() # Cleanup after test

def main():
    test_pipe_endpoint_python()

if __name__ == '__main__':
    main()
    


    
