import time, win32file, win32pipe, cv2
import numpy as np 
import uvicorn, pywintypes
from config import CONFIG
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
from threading import Thread
from BirdDog_TrackingModule import DetectionMananger, tracker_main
from tool.pipeclient import PipeClient 
from tool.payloads import *
from routers import websockets
import os

BUFFERSIZE = 921654
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360

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
    Tracker = DetectionMananger()
    track_flag = True

    try:
        print("Waiting for client")
        win32pipe.ConnectNamedPipe(pipe_handle, None) #This operation is blocking
        print("Got client")

        res = win32pipe.SetNamedPipeHandleState(pipe_handle, win32pipe.PIPE_READMODE_MESSAGE, None, None)
        if res == 0:
            print(f"SetNamedPipeHandleState return code: {res}")
        
        turbojpg_decoding = CONFIG.getboolean('server', 'turbojpeg')
        if turbojpg_decoding:
            from turbojpeg import TurboJPEG
            jpeg_decoder = TurboJPEG()
            
        while track_flag is True:   
            try:
                #Read Pipe in bytes form
                raw_data = win32file.ReadFile(pipe_handle, 1000000)

                #Parse using Python Pickle
                data = PipeClient.unpickle_object(raw_data[1])
                if data.__repr_name__() == PipeClient_Image_Payload().__repr_name__():
                    # TurboJPEG vs OpenCV has 1.6x decoding speed difference but TurboJPEG comes with installation overhead for system
                    if turbojpg_decoding:
                        frame = jpeg_decoder.decode(data.frame)
                    else:
                        nparr = np.frombuffer(data.frame, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
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
    
    except Exception as e:
        print(f"Pickle error: {e}")

    finally:
        win32pipe.DisconnectNamedPipe(pipe_handle)
        win32file.CloseHandle(pipe_handle)
        print("Pipe has been closed from server")

def init_pipe_server(pipeName = 'BD_Tracking'):
    if 'pipeNum' in locals():
        pipeNum += 1
    else:
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

def main():
    is_production = os.environ.get("PRODUCTION", False)
    uvicorn.run(app="TrackingServer_FastAPI:app", 
        host=CONFIG.get('server','host'), 
        port=CONFIG.getint('server','port'), 
        workers=CONFIG.getint('server','workers'),
        reload= not CONFIG.getboolean('server','production'),)

if __name__ == '__main__':
    main()
