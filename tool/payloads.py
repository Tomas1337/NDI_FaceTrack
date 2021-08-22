from pydantic import BaseModel
from pickle import UnpicklingError
import pickle
import sys
sys.path.insert(0, '.')
from config import CONFIG

class Parameter_Payload(BaseModel):
    target_coordinate_x: int = 320
    target_coordinate_y: int = 160
    track_type: int = 0
    gamma: float = ((CONFIG.getint('camera_control', 'gamma_default'))) / 10
    xMinE: float = ((CONFIG.getint('camera_control', 'horizontal_error_default'))) / 10
    yMinE: float = ((CONFIG.getint('camera_control', 'vertical_error_default'))) / 10
    zoom_value: float = 0.0
    y_track_state: bool = True
    autozoom_state: bool = True
    reset_trigger: bool = False

class Image_Payload(BaseModel):
    frame: bytes = None

class Server_Payload(BaseModel):
    x: int = None
    y: int = None
    w: int = None
    h: int = None
    x_velocity: float = None
    y_velocity: float = None

class PipeClient_Parameter_Payload(Parameter_Payload):
    def pickle_object(self):
        return pickle.dumps(self)

    @staticmethod
    def unpickle_object(data):
        try:
            return pickle.loads(data)
        except (EOFError, UnpicklingError):
            return ('Cannot unpickle object')

class PipeClient_Image_Payload(Image_Payload):
    def pickle_object(self):
        return pickle.dumps(self)

    @staticmethod
    def unpickle_object(data):
        try:
            return pickle.loads(data)
        except (EOFError, UnpicklingError):
            return ('Cannot unpickle object')

class PipeServerPayload(Server_Payload):
    def pickle_object(self):
        return pickle.dumps(self)