import time
from fastapi import WebSocket, APIRouter
from starlette.websockets import WebSocketDisconnect
from tool.payloads import *
from BirdDog_TrackingModule import DetectionWidget, tracker_main
from PySide2.QtCore import QDataStream, QByteArray, QIODevice,QBuffer
import numpy as np
import cv2
# from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE

router = APIRouter()
# jpeg = TurboJPEG()jpeg

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f'Current active connections: {self.active_connections}')

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@router.websocket("/qt_ws")
async def websocket_qt_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    frame = None
    Tracker = DetectionWidget()
    allowedProcessingTime = .1
    timeStep = 0
    try:
        while (websocket.client_state.value == 1):
            data_array = await websocket.receive()


            if data_array['type'] == 'websocket.disconnect':
                raise WebSocketDisconnect
            elif data_array['type'] == 'websocket.receive':
                data_q_array = QByteArray(data_array['bytes'])
            
            data_stream = QDataStream(data_q_array, QIODevice.ReadOnly)
            
            #Parse the datastream using the Pydantic ClientPayload Model
            parsed_payload = parse_qt_incoming_datastream(data_stream)
            if parsed_payload is None:
                continue
            
            #Data can come in two forms: Image or Parameter payload
            if (parsed_payload.__repr_name__() == Image_Payload().__repr_name__()) and \
                ((time.time() - timeStep) > allowedProcessingTime): # Barbaric but efficient way of dumping frames to get only the latest frame.
                timeStep = time.time()

                frame = parsed_payload.frame
                output = tracker_main(Tracker, frame)
                await websocket.send_json(output)

            elif parsed_payload.__repr_name__() == Parameter_Payload().__repr_name__():
                custom_parameters = parsed_payload.dict()
                Tracker.set_tracker_parameters(custom_parameters)
                await websocket.send_json(custom_parameters)

            else:
                print("Informational: frame has been skipped.")

    
    except WebSocketDisconnect:
        manager.disconnect(websocket)

def parse_qt_incoming_datastream(data_stream:QDataStream, structure:BaseModel = Image_Payload):
    """Parse data stream from QDataStream with Python Strcuture defined in structure
    Args:
        datastream (QDataStream): datastream containing the data to be parsed
        structure (BaseModel): Pydantic model strucutre to parse with
    
    Return:
    Parsed structure with values over written
    """
    header = data_stream.readQString()
    if header in ['frame', 'image']:
        #QDataStream > QByteArray > numpy Image 
        payload = structure()
        buff = QByteArray()
        data_stream >> buff
        
        image_bytes = bytes(buff)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        payload.frame = image

    elif header in ['parameter','parameters']:
        payload = Parameter_Payload() #Use for structure's reference only
        for key, value in payload.__dict__.items():
            if isinstance(value, int):
                parsed_value = data_stream.readInt32()                                                                                                                                     
            elif isinstance(value, float):
                parsed_value = data_stream.readFloat()
            elif isinstance(value, bool):
                parsed_value = data_stream.readBool()

            payload.__setattr__(key, parsed_value)
        
    else:
        print(f'Cannot parse the received header: {header}')
        payload = None

    return payload

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    "Consume video data through websockets that are encoded in Pydantic Models and get Tracking Coordinates"
    await websocket.accept()

    Tracker = DetectionWidget()
    try:
        while True:
            #Expects a Python pickled data
            data_pickled = await websocket.receive_bytes()
            data = pickle.loads(data_pickled)

            #Data can come in two forms: Image or Parameter payload
            if data.__repr_name__() == Image_Payload().__repr_name__():
                frame = jpeg.decode(data.frame)
                output = tracker_main(Tracker, frame)
                await websocket.send_json(output)

            elif data.__repr_name__() == Parameter_Payload().__repr_name__():
                custom_parameters = data.dict()
                Tracker.set_tracker_parameters(custom_parameters)
                await websocket.send_json("None")
                continue

    except WebSocketDisconnect:
        print("WebSocketDisconnect")
        await websocket.close() 

@router.get("/ws_docs")
def ws_docs():
    """
    This header is for documentation viewing for the `/qt_ws` and `/ws` websocket endpoints
    ---

    `http://localhost:port/ws_qt`

    Consume video data through websockets that are encoded in a Python Pickle Object and get a JSON response as a Tracking Coordinates dictionary
    Calling this enpoint initiates and opens a unique websocket between the Client and Server

    We expect a websocket client to finish the initial HTTP handshake using standard HEADERS to establish connection
    This endpoint will expect either a 'image' or 'paramater' pickle object as defined by `'Image_Payload'` and `'Parameter_Payload'`.
    Being a pickle object, we can easily parse between 'image' and 'parameter' payloads using it's Pydantic `__repr_name__()` property
            
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
    This pipe currently(v1.1.0) only supports communication between Python applications as limited by its Pickling encoding package. See `ws_qt` for QT implementation
    This endpoint will continue to persist until Timeouet or a websocket close signal is received from the Client.

    ---

    `http://localhost:port/ws_qt`

    The same as `'/ws' `endpoint but encoding and decoding using the QT Ecosystem, particulary `QByteArray` & `QDataStream`
    
    **IMPORTANT**: In order to distinguish between payloads, a `QString` containing the label (`'image' `or `'paramater'`) must be appended at the beginning of the payload

    Response for each:

    ```py
    Image_Payload -> JSON Response containing a dictionary of Trackable results
    Parameter_Payload -> No Response
    Invalid Payload -> No Response
    ```
    
    """
    pass
