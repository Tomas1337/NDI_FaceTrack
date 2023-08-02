from face_tracking.camera_control import *
from face_tracking.objcenter import *
from tool.custom_widgets import *
from config import CONFIG
import time, cv2
from tool.info_logging import add_logger
from decimal import Decimal
from collections import deque

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

        #Toggle for face recognition
        self.f_recognition = False

        # #Object Detectors
        # #self.face_obj = FastMTCNN()
        # self.face_obj = FastDeepSORT() #TRY THIS FIRST
        # self.body_obj = Yolo_v4TINY()

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
        
    def is_valid_coords(self, coords):
        """
        This function checks whether the coords can be unpacked to x, y, w, h.
        """
        return coords is not None and len(coords) == 4

    def main_track(self, frame, skip_frames=0, fallback_delay=5):
        """
        Takes in a list where the first element is the frame
        The second element are the target coordinates of the tracker
        """
        start = time.time()
        self.frame_count += 1
        
        # Keep track of the frame count
        if not hasattr(self, 'frame_count'):
            self.frame_count = 0
            
        # Only perform detection every 'skip_frames' frames
        if self.frame_count % (skip_frames+1) != 1:
            # Sped down x_speed_history approaching 0 linearly
            x_speed = self.x_speed_history[-1] * (skip_frames+1) / self.frame_count if len(self.x_speed_history) > 0 else 0.0
            y_speed = self.y_speed_history[-1] * (skip_frames+1) / self.frame_count if len(self.y_speed_history) > 0 else 0.0
            
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
    
        # Jun262023 Trackers
        #tracker = ObjectDetectionTracker(yolo_model_path='models/yolov8n.pt')
        tracker = ObjectDetectionTracker(yolo_model_path='models/yolov8n_640.onnx', task='detect')
        s1_time = time.time()
        results = tracker.track_with_class(frame, stream=True, device='cpu', persist=True, imgsz=640)
        #results = tracker.detect(frame, stream=True, device='cpu', imgsz=640)
        print(f"Time taken for detection and tracking: {time.time() - s1_time}")
        
        face_coords = None
        body_coords = None
        tracked_result = None
        fallback_result = None
        
        for result in results:
            box = result["box"]
            class_id = result.get('class_id', 0)
            detected_id = result.get('id')
            
            if class_id == 0:  # Person class
                # Prioritize the previously Tracked ID
                if detected_id == self.tracked_id:
                    tracked_result = result
                    break
                elif not fallback_result:
                    fallback_result = result
            
            elif class_id == 1:  # Body class
                body_coords = box
                
        if tracked_result:  # We found our tracked object
            self.tracked_id = tracked_result['id']
            face_coords = tracked_result['box']
            self.time_since_last_seen = time.time()  # Update the time we last saw it
            
        elif (fallback_result and  # We have a fallback object
            (not self.time_since_last_seen or time.time() - self.time_since_last_seen >= fallback_delay)):  # And we've waited long enough to switch
            self.tracked_id = fallback_result['id']
            face_coords = fallback_result['box']
            self.time_since_last_seen = time.time()  # Update the time we last saw it

        
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
        x_controller = PTZ_Controller_Novel(self.focal_length, self.gamma)
        x_speed = x_controller.omega_tur_plus1(objX, centerX, RMin = 0.1)

        y_controller = PTZ_Controller_Novel(self.focal_length, self.gamma)
        y_speed = y_controller.omega_tur_plus1(objY, centerY, RMin = 0.1, RMax=7.5)
        
        x_speed = float(Decimal(self._pos_error_thresholding(W, x_speed, centerX, objX, self.xMinE)).quantize(Decimal("0.0001")))
        y_speed = float(Decimal(self._pos_error_thresholding(H, y_speed, centerY, objY, self.yMinE)).quantize(Decimal("0.0001")))
        
        if self.y_track_state is False:
            y_speed = 0
        self.frame_count += 1

        self.x_prev_speed = x_speed
        self.y_prev_speed = y_speed
        
        ## Lets add some probablistic smoothing
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

        #print(f"Speed has been dampened from {self.x_prev_speed} to {x_speed} and {self.y_prev_speed} to {y_speed} by proib")
        print(f"Time taken for main_track: {(time.time() - start)}. Tracking ID {self.tracked_id}")
        
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
        

    def get_zoom_speed(self, zoom_speed = 0):
        if zoom_speed != 0:
            zoom_speed = 0
        return zoom_speed

    def face_tracker(self, frame):
        #ptvsd.debug_this_thread()
        """
        Input:
        Frame: The image on where to run a face_tracker on

        Return:
        x,y,w,h: Center and box coordiantes
        []: If no Face is Found

        Do a check every 300 frames
            If there are detections, check to see if the new detections overlap the current face_coordinates. 
                If it does overlap, then refresh the self.face_coords with the new test_coords
                If no overlap, then refresh the tracker so that it conducts a whole frame search
            If no detections in the body frame, detect on whole frame
                If there are face detections from the whole frame, return the face with the highest confidence
                Else, empty the face tracker and return []

        Check to see if tracker is ok
            Track Current face

        If track is not ok:
            Add to the lost face count
            When face Count reaches N, it empties the face_tracker
        """
        if self.f_tracker:
            ok, position = self.f_tracker.update(frame)
            
            if self.frame_count % 300 == 0:
                if self.lost_tracking_count > 0: # Prevent from going negative
                    self.lost_tracking_count -= 1
                    
                logger.debug(f'Running a {(300/30)}s check')
                pred_coords = self.face_obj.get_all_locations(frame)

                if pred_coords is not None and len(pred_coords) > 0: #There are detections
                    max_overlap = 0
                    for i, j in enumerate(pred_coords):
                        overlap = self.overlap_metric(j, self.face_coords)
                        if overlap > max_overlap:
                            max_overlap = overlap
                            x, y, w, h = j
                    
                    if max_overlap >= 0.50:
                        self.face_coords = [x, y, w, h] # Update face_coords if a sufficient overlap is found
                        return x, y, w, h
                    
                    else:
                        print(f"Overlap is {max_overlap}. Refreshing Face Detector")
                        self.f_tracker = None
                        return []
                
                else:
                        print(f"No detections in the body frame. Refreshing Face Detector")
                        self.f_tracker = None
                        return []
            
            elif ok:
                x = int(position[0])
                y = int(position[1])
                w = int(position[2])
                h = int(position[3])
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 1)
                x,y,w,h = [0 if i < 0 else i for i in [x,y,w,h]]
                self.f_track_count = 0
                face_coords = x,y,w,h
                
            else:
                self.lost_tracking_count += 1
                if self.f_track_count > 5:
                    logger.debug('Lost face') 
                    self.info_status('Refreshing Face Detector from Lost Tracking')     
                    self.f_tracker = None
                    return []
                else:
                    self.f_track_count += 1
                    return []

        elif self.f_tracker is None:
            """
            Detect a face inside the body Coordinates
            If no detections in the body frame, detect on whole frame which gets the face detection with the strongest confidence
            """
            if self.frame_count % 20 == 0: #MTCNN Is slow so we have to be sparing. #TODO Change to Yolo 
                face_coords = self.face_obj.update(frame)
            else:
                face_coords = []
                
            if len(face_coords) > 0:
                x,y,w,h = face_coords
            else:
                return []

            #Start a face tracker
            #self.f_tracker = cv2.TrackerKCF_create()
            self.f_tracker = cv2.TrackerCSRT_create()
            self.f_tracker.init(frame, (x,y,w,h))
            logger.debug('Initiating a face tracker')

        self.face_coords = x,y,w,h
        return x,y,w,h

    def body_tracker(self, frame):
        #ptvsd.debug_this_thread()
        if self.b_tracker is None:
            #Detect Objects using YOLO every 1 second if No Body Tracker    
            boxes = []
            if self.frame_count % 15 == 0:
                (idxs, boxes, _, _, classIDs, confidences) = self.body_obj.update(frame)
                print('Running a YOLO')

            if len(boxes) <= 0:
                return []

            elif len(boxes) == 1:
                x,y,w,h = boxes[np.argmax(confidences)]
                x,y,w,h = [0 if i < 0 else int(i) for i in [x,y,w,h]]

            #If the body detections are more than 1 and there are present self.face_coords
            elif len(boxes) > 1 and len(self.face_coords) >= 1:
                for i, g in enumerate(boxes):
                    if self.overlap_metric(g, self.face_coords) >= 0.5:
                        x,y,w,h = [0 if i < 0 else int(i) for i in [x,y,w,h]]
                        x,y,w,h = [int(p) for p in boxes[i]]
                        break

            #Start the body tracker for the given xywh
            self.b_tracker = cv2.TrackerKCF_create()
            try:
                max_width = 100
                if w > max_width:
                    scale = max_width/w
                    x,y,w,h = self._resizeRect(x,y,w,h,scale)
                else:
                    pass
                self.b_tracker.init(frame, (x,y,w,h))
            except UnboundLocalError:
                return []
            return x,y,w,h

        #If theres a tracker already            
        elif not self.b_tracker is None:
            tic = time.time()       
            self.btrack_ok, position = self.b_tracker.update(frame)
            if self.btrack_ok:
                x = int(position[0])
                y = int(position[1])
                w = int(position[2])
                h = int(position[3])
                x,y,w,h = [0 if i < 0 else i for i in [x,y,w,h]]

            else:
                self.lost_tracking_count += 1
                logger.debug('Tracking Fail')
                return []
            

            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,255), 2)
            return x,y,w,h

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
    x_velocity, y_velocity = Tracker.main_track(frame, skip_frames=5)
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
    
    return output

logger = add_logger()

if __name__ == '__main__':
    
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 360

    Tracker = DetectionWidget()
    print('Starting new tracker?')
    args = {}
    tracker_main(*args)
