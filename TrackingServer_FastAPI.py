import re
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
from torch import pca_lowrank

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

sys.path.append('deep-person-reid/')
from torchreid.utils import FeatureExtractor, re_ranking
from torchreid.losses.hard_mine_triplet_loss import TripletLoss
import torchreid.metrics as metrics

import torch

class Identity():
    def __init__(self, features, identity, pca=False):
        self.pca = pca
        self.features = features
        assert isinstance(identity,int)
        self.identity = identity
        self.images = []
        self.features = [] #Convert this to a torch tensor
        self.store_features(features) #Save the feature the identity was initiated
        
        
    def store_image(self, image, max = 5, override=False):
        # Store image in cache 
        # WARNING: Will eat up memory if you store a lot of images
        # Try to store only a maximum of 5 images at a time

        if len(self.images) < max or override:
            self.images.append(image)
        else:
            return False
        return True
    
    def store_features(self, feature, max = 5, override=False):
        # Store embeddings in cache 
        # WARNING: Will eat up memory if you store a lot of images
        # Try to store only a maximum of 5 images at a time

        if len(self.features) < max or override:
            self.features.append(feature)
            if self.pca:
                self.run_pca()
        else:
            return False
        return True

    def get_features(self, num=1):
        #Return a number of features or just one
        return self.features[:num]

    def get_images(self):
        return self.images

    def get_identity(self):
        return self.identity

    def run_pca(self):
        self.pca = pca_lowrank(self.features)

class ReId_Object():
    """
    Contains all the methods for the ReIdentification 

    Main method here is 
    """

    def __init__(self, triplet_loss_margin=0.3, pca=False, dist_metric='euclidean'):
        F_EXTRACTOR = 'osnet_x1_0'
        self.extractor = FeatureExtractor(model_name='osnet_x1_0')
        self.triplet_loss = TripletLoss(margin=triplet_loss_margin)
        #Empty numpy array of size (1,512)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.identities = None
        self.pca = pca
        self.dist_metric = dist_metric

    def load_model(self, model=None):
        if self.model is None:
            model_path =  CONFIG.get('reid', 'model')

    
    def is_new_identity(self, identity_feature, threshold=200):
        # Checks if there are any similar identities in self.identities
        # Return True if found similarity bigger than the threshold
        # Return False if no similarity found

        if self.identities is None:
            return None
        
        dist_list = []
        dist_metric='euclidean'
        
        for identity in self.identities:
            feature = identity.get_features()[0]
            try:
                dist = metrics.compute_distance_matrix(feature, identity_feature,self.dist_metric)
                dist_list.append(dist.item())
            except Exception as e:
                raise Exception(f'Could not compute distance matrix: {e}')
            
            if dist < threshold:
                # Early return without evaluating other results
                return identity.identity #return the first match

        #If none are found then return False
        return True

    def rerank(self, qf):

        gf = self.identities
          
        #TODO Implement reranking
        print('Applying person re-ranking ...')
        distmat_qq = metrics.compute_distance_matrix(qf, qf, self.dist_metric)
        distmat_gg = metrics.compute_distance_matrix(gf, gf, self.dist_metric)
        distmat = re_ranking(distmat, distmat_qq, distmat_gg)

    def merge_similar_identities(self, threshold=200, rerank=False):
        # Checks if there are any similar identities in self.identities
        # If found similarity bigger than the threshold, delete the latter identity
        pass

    def get_features(self, frame):
        # Predict the features from the frame
        if frame is not None:
            feature = self.extractor(frame)
            return feature

        else:
            print('frame is none')
            return None

    def add_features(self, feature, identity_key):
        identity = self.identities[identity_key]
        identity.store_features(feature)

    def add_identity(self, feature):
        # Adds a new Identity(class) to the list of identities.
        # Returns the identity number of the new identity

        # Note:This function does not check   similarity between its nested objects
        
        if self.identities is None:
            print("Creating new identity cache")
            key = 0
            self.identities =  [Identity(feature, key, pca=self.pca)]
            return key
        else: 
            #Associate it with a max key
            max_key = len(self.identities)
            self.identities.append(Identity(feature, max_key, pca=self.pca))
            print(f'Added new identity with key {max_key}')
            return max_key

        #     elif identity in self.identities.keys():
        #         self.identities[identity] = torch.cat((self.identities[identity], feature), dim=0)
        #         print(f'Added new feature to identity #{identity}')
        #         return identity.get_identity()
        
class Video_Object():
    def __init__(self):
        self.np_frame = None
        self.bounding_boxes = []
        self.tracker_object = Tracker_Object()
        self.reid_object = ReId_Object()
        
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
    
    def reid(self):
        pass
        

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
