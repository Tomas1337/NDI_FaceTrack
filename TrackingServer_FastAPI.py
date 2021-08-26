import time, win32file, win32pipe, cv2, json, requests
import numpy as np 
import uvicorn, pywintypes, pickle
from config import CONFIG
from multiprocessing import Process
from fastapi import FastAPI, File, Body, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.testclient import TestClient
from fastapi.openapi.utils import get_openapi
from starlette.responses import StreamingResponse
from pydantic import BaseModel
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE
from threading import Thread
from TrackingModule import DetectionWidget, tracker_main
from tool.pipeclient import PipeClient 
from tool.payloads import *
from routers import websockets

BUFFERSIZE = 921654
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360

jpeg = TurboJPEG()
app = FastAPI()
app.include_router(websockets.router)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title = 'Camera Auto Tracking module',
        description = "A Windows desktop application allows Bird Dog PTZ Cameras to automatically track persons of interests, freeing up camera operators. This application uses Machine Learning and CV Techniques to Identify and track both faces and human body figures. NDI Facetracking uses open multiple open source projects to work seamlessly betwween each other.",
        version = "1.1.0",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

class StartPipeResponse(BaseModel):
    pipeName: str

@app.post('/api/start_pipe_server', response_model = StartPipeResponse)
def start_pipe_server():
    """
    Calling this enpoint initiates ands opens a Pipe Server from the Server end.
    
    We expect a Pipe Client from an external application to connect to the Pipe Server initiated using the returned 'pipeName'
    
    A handshake must be established before (initiated by the client) communication is fully established 

    The pipe will expect either a 'image' or 'paramater' pickle object as defined by 'Image_Payload' and 'Parameter_Payload'.
        
        ```py
        class Image_Payload(BaseModel):
            frame: bytes = None

        class Server_Payload(BaseModel):
            x: int = None
            y: int = None
            w: int = None
            h: int = None
            x_velocity: float = None
            y_velocity: float = None
        ```
    
    Response for each:

        ```py
        Image_Payload -> JSON Response containing a dictionary of Trackable results
        Parameter_Payload -> No Response
        Invalid Payload -> No Response
        ```

    This pipe currently(v1.1.0) only supports communication between Python applications as limited by its Pickling encoding package.
    
    For Universal communication, please use the websocket or websocket_qt endpoint between different languages.

    """
    #Starts pipe server
    pipeName, pipe_handle = init_pipe_server()
    init_response = checkPipe(pipe_handle)

    if init_response is True:
        tracking_thread = Thread(target = start_tracking_pipe, args=[pipe_handle])
        tracking_thread.start()
        response = {"pipeName" : pipeName }

    else:
        response = JSONResponse(status_code=404, content={"message": "Could not connect to pipe"})
    return response

def start_tracking_pipe(pipe_handle):
    Tracker = DetectionWidget()
    track_flag = True

    try:
        print("Waiting for client")
        win32pipe.ConnectNamedPipe(pipe_handle, None) #This operation is blocking
        print("Got client")

        res = win32pipe.SetNamedPipeHandleState(pipe_handle, win32pipe.PIPE_READMODE_MESSAGE, None, None)
        if res == 0:
            print(f"SetNamedPipeHandleState return code: {res}")
        
        while track_flag is True:   
            try:
                #Read Pipe in bytes form
                raw_data = win32file.ReadFile(pipe_handle, 1000000)

                #Parse using Python Pickle
                data = PipeClient.unpickle_object(raw_data[1])
                if data.__repr_name__() == PipeClient_Image_Payload().__repr_name__():
                    frame = jpeg.decode(data.frame)
                    output = tracker_main(Tracker, frame)

                elif data.__repr_name__() == PipeClient_Parameter_Payload().__repr_name__():
                    custom_parameters = data.dict()
                    Tracker.set_tracker_parameters(custom_parameters)
                    continue

                else:
                    print('Bytes are of unknown format')
                    continue
    
                #Pass to Track Module and get 
                output = tracker_main(Tracker, frame)
         
                #Unpack and Pickle response
                x,y,w,h,x_Velocity,y_Velocity = output['x'], output['y'], output['w'], output['h'], output['x_velocity'], output['y_velocity']
                payload = PipeServerPayload(x=x,y=y,w=w,h=h,x_velocity=x_Velocity,y_velocity=y_Velocity)       
                payload = payload.pickle_object()

                #Write Pipe
                win32file.WriteFile(pipe_handle, payload)

            except pywintypes.error as e:
                print("Could not read from pipe. shutting down the pipe. Error: {e}")
                track_flag = False
                
    except pywintypes.error as e:
        if e.args[0] == 2:
            print("no pipe, trying again in a sec")
            time.sleep(1)
        elif e.args[0] == 109:
            print("broken pipe, bye bye")
        else:
            print(e)
        print("finished now")

    finally:
        win32pipe.DisconnectNamedPipe(pipe_handle)
        win32file.CloseHandle(pipe_handle)
        print("Pipe has been closed from server")

def init_pipe_server(pipeName = 'BD_Tracking'):
    try: 
        pipeNum += 1
    except NameError:
        pipeNum = 1

    while True:
        try:
            pipeName = 'BD_Tracking' + str(pipeNum)
            pipe_handle = win32pipe.CreateNamedPipe(
                r'\\.\pipe\{}'.format(pipeName),
                win32pipe.PIPE_ACCESS_DUPLEX,
                win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
                1, 921600, 921600,
                0,
                None)
            break

        except pywintypes.error as e:
            if e.winerror == 231:
                pipeNum +=1
            else:
                print(f'Could not Start pipe due to error: {e.strerror}')
                return None

    print(f"Starting a FastAPI server Pipe with name {pipeName} with pipe_handle of {pipe_handle}")

    pipeState = checkPipe(pipe_handle)
    print(f'Checked the pipe_handle which returned {pipeState}')
    return (pipeName, pipe_handle)

def checkPipe(pipe_handle):
    """Checks if pipe at fileHandle is open
    Args:
        fileHandle : Handler for the Pipe

    Returns:
        response
            0: success
            1: Error'd out
    """
    try:
        pipeFlag, _, _, _ = win32pipe.GetNamedPipeInfo(pipe_handle)
        if pipeFlag > 0:
            pipeState = True
        else: 
            pipeState = False

    except pywintypes.error as e:
        pipeState = False
    
    return pipeState

@app.post('/api/check_tracking_server')
def check_tracking_server():
    #Check if there is a current running FastAPI Server for tracking using requests
    signature = CONFIG.get('server', 'signature')
    try:
       response = signature
    except:
        print('Cannot verify tracking server')
        response = False
        
    finally:
        return response

class Tracker_Object():
    def __init__(self):
        self.tracker = DetectionWidget()

    def track_frame(self, frame):  
        target_coordinates = (50,50)
        custom_parameters = {'gamma':0.0}
        if frame is not None:
            output = tracker_main(self.tracker, frame, target_coordinates, custom_parameters)
            bb_box = output.get('boundingBox')
            self.set_bounding_boxes(bb_box)

            return output
        else:
            print('frame is none')
            return None

    def get_bounding_boxes(self):
        return self.bounding_boxes

    def set_bounding_boxes(self, bounding_box):
        self.bounding_boxes = bounding_box

class Video_Object():
    def __init__(self):
        self.np_frame = None
        self.bounding_boxes = []
        self.tracker_object = Tracker_Object()
        
    def frame_gen(self, ndi_recv, draw = True, track = True):
        while True:
            t,v,_,_ = ndi.recv_capture_v2(ndi_recv, 0)
            if t == ndi.FRAME_TYPE_VIDEO:
                frame = v.data
                frame = frame[:,:,:3]
                frame = np.array(frame)
                self.np_frame = frame

                if track:
                    self.track()
                    if draw and len(self.bounding_boxes[0]) >= 1:
                        for (x,y,x2,y2) in self.bounding_boxes:
                            cv2.rectangle(frame, (x,y), (x2,y2), (255,0,0), 1)

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    def get_frame(self):
        return self.np_frame

    def track(self):
        output = self.tracker_object.track_frame(self.np_frame)
        if output is not None:
            print(output.get('xVelocity'), output.get('yVelocity'))
            self.bounding_boxes = self.tracker_object.get_bounding_boxes()

# @app.post('/video_feed', include_in_schema=False)
# def video_feed():
#     #http://127.0.0.1:5000//video_feed?camera_name=DESKTOP-C16VMFB (VLC) 
#     camera_name = request.args.get('camera_name')
#     return render_template("video_feed.html", camera_name=camera_name)

# @app.post('/video_camera', include_in_schema=False)
# def video_camera():
#     camera_name = request.args['camera_name']
#     video_object = Video_Object()
#     #tracker_object = Tracker_Object(video_object)
#     # video_thread = Thread(target = start_tracking_html, args=[video_object])
#     # video_thread.start()    
#     #return_value = future.result()
#     #print(return_value)
#     #print('hello')
#     #ndi_cam = ndi_camera()
#     #ndi_recv = ndi_cam.camera_connect(ndi_name=camera_name)

#     return Response(video_object.frame_gen(ndi_recv),
# 		mimetype = "multipart/x-mixed-replace; boundary=frame")


def main():
    uvicorn.run(app=app, 
        host=CONFIG.get('server','host'), 
        port=CONFIG.getint('server','port'), 
        workers=CONFIG.getint('server','workers'))

if __name__ == '__main__':
    main()
