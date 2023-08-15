from face_tracking.camera_control import *
from face_tracking.objcenter import *
from tool.custom_widgets import *
from config import CONFIG
import time, cv2
from tool.info_logging import add_logger
from decimal import Decimal
from collections import deque
import numpy as np

class DetectionWidget():
    
    def __init__(self):
        FRAME_WIDTH = 640
        FRAME_HEIGHT = 360
        self.last_loc = [FRAME_WIDTH//2, FRAME_HEIGHT//2]
        self.target_coordinate_x = FRAME_WIDTH//2
        self.target_coordinate_y = FRAME_HEIGHT//2
        self.track_coords = []
        self.btrack_ok = False
        self.focal_length = 129
        self.body_y_offset_value = CONFIG.getint('home_position', 'body_vertical_home_offset')

        #Counters
        self.frame_count = 1
        self.lost_tracking = 0
        self.lost_tracking_count = 0
        self.f_track_count = 0
        self.overlap_counter = 0
        self.lost_t = 0

        #Trackers
        self.b_tracker = None
        self.f_tracker = None
        self.p_tracker = None

        #Slider and button Values
        self.track_type = 0
        self.y_track_state = True
        self.gamma = float(CONFIG.getint('camera_control', 'gamma_default'))/10
        self.xMinE = float(CONFIG.getint('camera_control', 'horizontal_error_default'))/10
        self.yMinE = float(CONFIG.getint('camera_control', 'vertical_error_default'))/10
       
        self.zoom_value = 0
        self.zoom_state = 0
        self.autozoom_state = True
        self.face_lock_state = False
        self.face_coords = []
        self.reset_trigger = False
        self.parameter_list = ['target_coordinate_x', 'target_coordinate_y', 'gamma', 'xMinE', 'yMinE', 'zoom_value', 'y_track_state','autozoom_state','reset_trigger','track_type']
        
        #Speed Tracker
        self.x_prev_speed = None
        self.y_prev_speed = None
        
        # Maintain a history of recent speeds for both x_speed and y_speed.
        self.x_speed_history = deque(maxlen=60)  # Store last 60 frames
        self.y_speed_history = deque(maxlen=60)  # Store last 60 frames
        
        # Add new attributes if they don't exist
        self.tracked_id = None
        self.time_since_last_seen = None
        self.last_loc = None
        
        # Detector and Tracker 
        self.tracker = ObjectDetectionTracker(yolo_model_path='models/yolov8n_640.onnx', 
            task='detect', use_csrt=True, overlap_frames=None, device='cpu')
        
    def is_valid_coords(self, coords):
        """
        This function checks whether the coords can be unpacked to x, y, w, h.
        """
        return coords is not None and len(coords) == 4
    
    def calculate_camera_control(self, objX, objY, centerX, centerY, W, H):
        x_controller = PTZ_Controller_Novel(self.focal_length, self.gamma)
        x_speed = x_controller.omega_tur_plus1(objX, centerX, RMin=0.1)
        y_controller = PTZ_Controller_Novel(self.focal_length, self.gamma)
        y_speed = y_controller.omega_tur_plus1(objY, centerY, RMin=0.1, RMax=7.5)
        x_speed = float(Decimal(self._pos_error_thresholding(W, x_speed, centerX, objX, self.xMinE)).quantize(Decimal("0.0001")))
        y_speed = float(Decimal(self._pos_error_thresholding(H, y_speed, centerY, objY, self.yMinE)).quantize(Decimal("0.0001")))
        return x_speed, y_speed
    
    def process_speed_history(self, x_speed, y_speed):
        # Add the new speed to the history.
        self.x_speed_history.append(x_speed)
        self.y_speed_history.append(y_speed)

        # Calculate the mean and standard deviation of these histories.
        x_mean = np.mean(self.x_speed_history)
        x_std = np.std(self.x_speed_history)
        y_mean = np.mean(self.y_speed_history)
        y_std = np.std(self.y_speed_history)

        # If the current speed is more than a certain number of standard deviations away from the mean, ignore it.
        threshold = 2  # Adjust as needed.
        if abs(x_speed - x_mean) > threshold * x_std:
            x_speed = self.x_prev_speed  # Ignore the calculated speed.

        if abs(y_speed - y_mean) > threshold * y_std:
            y_speed = self.y_prev_speed  # Ignore the calculated speed.

        return x_speed, y_speed


    def main_track(self, frame, skip_frames=None, fallback_delay=5):
        """
        Takes in a list where the first element is the frame
        The second element are the target coordinates of the tracker
        """
        start = time.time()
        self.frame_count += 1
        
        # Keep track of the frame count
        if not hasattr(self, 'frame_count'):
            self.frame_count = 0
            
        #Only perform detection every 'skip_frames' frames
        if skip_frames and self.frame_count % (skip_frames+1) != 1:
            # Sped down x_speed_history approaching 0 linearly
            # x_speed = self.x_speed_history[-1] * (skip_frames+1) / self.frame_count if len(self.x_speed_history) > 0 else 0.0
            # y_speed = self.y_speed_history[-1] * (skip_frames+1) / self.frame_count if len(self.y_speed_history) > 0 else 0.0            
            self.frame_count += 1
            x_speed = self.x_prev_speed
            y_speed = self.y_prev_speed
            return (x_speed, y_speed)
        
        self.track_coords = []
        self.set_focal_length(self.zoom_value)
    
        if self.reset_trigger is True:
            self.reset_tracker()
            self.reset_trigger = False
        
        frame = np.array(frame)

        (H, W) = frame.shape[:2]
        centerX = self.target_coordinate_x
        centerY = self.target_coordinate_y
    
        results =  self.tracker.track_with_csrt(frame, device='cpu', imgsz=640)
        #print(f"Time taken for detection and tracking: {time.time() - s1_time}")
        
        face_coords = None
        body_coords = None

        for result in results:
            box = result["box"]
            body_coords = box
            face_coords = box
        if len(results) == 0:
            face_coords=None
        
        if self.is_valid_coords(face_coords):
            x, y, w, h = face_coords
            self.track_coords = [x, y, x + w, y + h]
            face_track_flag = True
        elif self.is_valid_coords(body_coords):
            x, y, w, h = body_coords
            self.track_coords = [x, y, x + w, y + h]
            self.overlap_counter = 0
            face_track_flag = False
        else:
            face_track_flag = False
        
        offset_value = int(self.body_y_offset_value * (H/100))

        #Normal Tracking
        if len(self.track_coords) > 0:
            [x,y,x2,y2] = self.track_coords
            objX = int(x+(x2-x)//2)
            objY = int(y+(y2-y)//2) if face_track_flag else int(y + offset_value)
            self.last_loc = (objX, objY)
            self.lost_tracking = 0
        else:
            objX = self.target_coordinate_x
            objY = self.target_coordinate_y

        objX = int(objX) if objX is not None else 0
        objY = int(objY) if objY is not None else 0
        
        ## CAMERA CONTROL
        x_speed, y_speed = self.calculate_camera_control(objX, objY, centerX, centerY, W, H)

        if self.y_track_state is False:
            y_speed = 0

        self.frame_count += 1
        self.x_prev_speed = x_speed
        self.y_prev_speed = y_speed

        ## Probabilistic Smoothing
        x_speed, y_speed = self.process_speed_history(x_speed, y_speed) if bool(CONFIG.getboolean('camera_control', 'speed_history_smoothing')) else (x_speed, y_speed)

        return (x_speed, y_speed)

    def get_bounding_boxes(self):
        return self.track_coords

    def set_tracker_parameters(self, custom_parameters):
        #Unpack Dictionary and set Track Parameters
        #Only set tracker parameters that are inside `custom_parameters`
        #Leaving it to {} will not set any custom_parameter in Tracker
        if custom_parameters == {}:
            return
        for key in self.parameter_list:
            setattr(self, key, custom_parameters.get(key))

    def get_tracker_parameters(self):
        temp_dict = {}
        for key in self.parameter_list:
            value = getattr(self, key)
            temp_dict[key] = value

        return temp_dict

    def set_focal_length(self, zoom_level):
        """
        Range is 129mm to 4.3mm
        When Zoom Level is 0, it is = 129mm
        When Zoom level is 10, it is = 4.3mm
        Values in between might not be linear so adjust from there
        """
        zoom_dict = {0:129, 1:116.53, 2:104.06, 3:91.59, 4: 79.12,5:66.65, 6:54.18, 7:41.71, 8:29.24, 9:16.77, 10:4.3}
        self.focal_length = zoom_dict.get(int(zoom_level or 0))
    
    def reset_tracker(self):
        print('Triggering reset')
        self.f_tracker = None
        self.b_tracker = None
        self.tracker.reset_tracker()
        

    def get_zoom_speed(self, zoom_speed = 0):
        if zoom_speed != 0:
            zoom_speed = 0
        return zoom_speed

    def info_status(self, status: str):
        print(status)

    def _pos_error_thresholding(self,frame_dim, calc_speed, center_coord, obj_coord, error_threshold):
        """
        Function to check if the distance between the desired position and the current position is above a threshold.
        If below threshold, return 0 as the speed
        If above threshold, do nothing
        """
        error = np.abs(center_coord - obj_coord)/(frame_dim/2)
    
        if error >= error_threshold:
            pass
        else:
            calc_speed = 0
        
        return calc_speed

    def _resizeRect(self, x,y,w,h,scale):    
        x = int(x + w * (1 - scale)/2)
        y = int(y + h * (1 - scale)/2)
        w = int(w * scale)
        h = int(h * scale)
        return (x,y,w,h)

    def overlap_metric(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        # BoxA and BoxB in x,y,w,h format
        
        def _translate(box):
            # Translate into x,y,x1,y1
            x,y,w,h = box
            x1 = x+w
            y1 = y+h
            return [x,y,x1,y1]

        boxA = _translate(boxA)
        boxB = _translate(boxB)

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        if interArea == 0:
            return 0
        boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
        iou = interArea/boxBArea
        return iou


def tracker_main(Tracker, frame, custom_parameters = {}):
    """
    Inputs come from the UI:
        Tracker: Object
            - An initated DetectionWidget
        frame: Numpy array
            -3 Channel with dimensions of (H, W, C); 


        custom_parameters: dict
            - A dictionary containing the below parameters
            target_coordinate_x: int(0,640)
            target_cooridinate_y: int(0,320)
                - Desired Position of the object relative to the Frame where (0,0) is on the Top Left Corner.
                - Must not exceed H and W
        
            gamma: int(0~10)
                Senstivity of camera movement
            
            xMinE: int(0~10)
                Minmium error of Horizontal movement
            
            yMinE: int(0~10)
                Minmium error of Vertical movement

            zoom_value : float (0.0 ~ 10.0)
                Adjusting the Zoom_level automatically adjusts the the focal_length variable of the algorithm. Which is used to compute for the Camera speeds

            y_track_state: bool
                0: Disabled
                1: Enabled (default)

            autozoom_state: bool
                0: Disabled
                1: Enabled (default)

            reset_trigger: bool
                If enabling reset, next frame or next few frames upon trigger must send 0 continously again.
                0: Do nothing
                1: Enable Reset
            
            track_type: int
                0: Face -> Body Tracking (default)
                1: Face Only
                2: Body Only

    Return:
        A dictionary where:
        xVelocity, yVelocity: Vectors for camera direction and speed
        boundingBox: A list of 4 points in the format [x,y,x2,y2]
    """ 
    
    if Tracker is None:
        return 0
    
    Tracker.set_tracker_parameters(custom_parameters)
    x_velocity, y_velocity = Tracker.main_track(frame, skip_frames=None)
    #print(f"Tracker has returned with x_velocity: {x_velocity}, y_velocity: {y_velocity}")
    track_coords = Tracker.get_bounding_boxes()
    
    output = {
    "x_velocity": x_velocity, 
    "y_velocity": y_velocity, 
    }

    if len(track_coords) == 4:
        output["x"] = track_coords[0]
        output["y"] = track_coords[1]
        output["w"] = track_coords[2]
        output["h"] = track_coords[3]
    else:
        #print("Cannot unpack track_coords")
        output["x"] = None
        output["y"] = None
        output["w"] = None
        output["h"] = None
    
    #print(f"Output XY: {output['x_velocity']}, {output['y_velocity']}")
    return output

logger = add_logger()

if __name__ == '__main__':
    
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 360

    Tracker = DetectionWidget()
    print('Starting new tracker?')
    args = {}
    tracker_main(*args)
