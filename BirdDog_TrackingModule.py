from face_tracking.camera_control import *
from face_tracking.objcenter import *
from tool.custom_widgets import *
from config import CONFIG
import dlib, time, styling, sys

class DetectionWidget():
    def __init__(self):
        self.last_loc = [320, 180]
        self.track_coords = []
        self.btrack_ok = False
        self.focal_length = 129
        self.body_y_offset_value = CONFIG.getint('home_position', 'body_vertical_home_offset')

        #Counters
        self.frame_count = 1
        self.lost_tracking = 0
        self.f_track_count = 0
        self.overlap_counter = 0
        self.lost_t = 0

        #Trackers
        self.b_tracker = None
        self.f_tracker = None

        #Object Detectors
        self.face_obj = FastMTCNN() #TRY THIS FIRST
        #self.body_obj = Yolov3()
        self.body_obj = Yolov4()

        #Slider and button Values
        self.track_type = 0
        self.y_trackState = True
        self.yminE = 0.13
        self.xminE = 0.13
        self.gamma = (CONFIG.getint('camera_control', 'gamma_default'))/10
        self.ZoomValue = 0
        self.autozoom_state = True
        self.face_lock_state = False
        self.face_coords = []
        self.reset_trigger = False

    def main_track(self, _list):
        """
        Takes in a list where the first element is the frame
        The second element are the target coordinates of the tracker
        """
        self.track_coords = []
        self.set_focal_length(self.ZoomValue)
        start = time.time()
        frame = _list[0]

        if self.reset_trigger is True:
            self.reset_tracker()
            self.reset_trigger = False

        try:
            center_coords = _list[1]
        except IndexError:
            (H, W) = frame.shape[:2]
            center_coords = (W//2, H//2)

        (H, W) = frame.shape[:2]
        centerX = int(center_coords[0])
        centerY = int(center_coords[1])
        objX = centerX
        objY = centerY
        
        #Trackers
        if self.track_type == 0:
            #Face-> Body (If cannot detect a face, try to detect a body)
            face_coords = self.face_tracker(frame)
            if len(face_coords) <= 0:
                body_coords = self.body_tracker(frame)

        elif self.track_type == 1:
            #Face Only
            face_coords = self.face_tracker(frame)
        
        elif self.track_type == 2:
            #Body Only
            body_coords = self.body_tracker(frame)

        try:
            x,y,w,h = face_coords
            self.track_coords = [x,y,x+w,y+h]
            #self.info_status.emit('Tracking Face')
            face_track_flag = True
        except (ValueError, TypeError, UnboundLocalError) as e:
            try:
                x,y,w,h = body_coords
                self.track_coords = [x,y,x+w,y+h]
                #self.info_status.emit('Tracking Body')
                self.overlap_counter = 0
            except (ValueError, UnboundLocalError) as e:
                pass
            finally:
                face_track_flag = False
        
        offset_value = int(self.body_y_offset_value * (H/100))

        #Normal Tracking
        if len(self.track_coords) > 0:
            [x,y,x2,y2] = self.track_coords
            objX = int(x+(x2-x)//2)
            objY = int(y+(y2-y)//2) if face_track_flag else int(y + offset_value)
            self.last_loc = (objX, objY)
            self.lost_tracking = 0

        #Initiate Lost Tracking sub-routine
        else:
            if self.lost_tracking > 100 and self.lost_tracking < 500:
                #self.info_status.emit("Lost Tracking Sequence Secondary")
                #self.CameraZoomControlSignal.emit(-0.7)
                print("Lost Tracking Sequence - Secondary")

            elif self.lost_tracking < 20:
                objX = self.last_loc[0]
                objY = self.last_loc[1]
                self.lost_tracking += 1
                print('Lost tracking. Going to last known location of object')
                #self.info_status.emit("Lost Tracking Sequence Initial")
                #self.CameraZoomControlSignal.emit(0.0)

            else:
                objX = centerX
                objY = centerY
                print('Lost object. Centering')
                #self.info_status.emit("Lost Object. Recenter subject")
                self.lost_tracking += 1
                #self.CameraZoomControlSignal.emit(0.0)
                if self.lost_tracking < 500 and self.lost_tracking%100:
                    self.b_tracker = None

        ## CAMERA CONTROL
        x_controller = PTZ_Controller_Novel(self.focal_length, self.gamma)
        x_speed = x_controller.omega_tur_plus1(objX, centerX, RMin = 0.1)

        y_controller = PTZ_Controller_Novel(self.focal_length, self.gamma)
        y_speed = y_controller.omega_tur_plus1(objY, centerY, RMin = 0.1, RMax=7.5)
        
        x_speed = self._pos_error_thresholding(W, x_speed, centerX, objX, self.xminE)
        y_speed = self._pos_error_thresholding(H, y_speed, centerY, objY, self.yminE)

        if self.y_trackState is False:
            y_speed = 0
        self.frame_count += 1
        #print("X Speed: {} Y Speed: {}".format(x_speed, y_speed))
        
        return (x_speed, y_speed)

    def get_bounding_boxes(self):
        return self.track_coords


    def set_focal_length(self, zoom_level):
        """
        Range is 129mm to 4.3mm
        When Zoom Level is 0, it is = 129mm
        When Zoom level is 10, it is = 4.3mm
        Values in between might not be linear so adjust from there
        """
        zoom_dict = {0:129, 1:116.53, 2:104.06, 3:91.59, 4: 79.12,5:66.65, 6:54.18, 7:41.71, 8:29.24, 9:16.77, 10:4.3}
        self.focal_length = zoom_dict.get(int(zoom_level))
    
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
        if not self.f_tracker is None:
            ok, position = self.f_tracker.update(frame)
            if self.frame_count % 300== 0:
                print('Running a {}s check'.format(300/30))
                test_coords = self.face_obj.get_all_locations(frame)

                if len(test_coords) > 0: #There are detections
                    for i, j in enumerate(test_coords):
                        if self.overlap_metric(j, self.face_coords) >= 0.75:
                            x,y,w,h = j
                            break
                    return []

                else: #No detections
                    self.f_tracker = None
                    return []
            
            elif ok:
                x = int(position[0])
                y = int(position[1])
                w = int(position[2])
                h = int(position[3])
                #cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 1)
                x,y,w,h = [0 if i < 0 else i for i in [x,y,w,h]]
                self.f_track_count = 0
                face_coords = x,y,w,h
                
            else:
                if self.f_track_count > 5:
                    print('Lost face') 
                    #self.info_status.emit('Refreshing Face Detector')     
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
            if self.frame_count % 1 == 0:
                face_coords = self.face_obj.update(frame)
            else:
                face_coords = []
                
            if len(face_coords) > 0:
                x,y,w,h = face_coords
            else:
               return []

            #Start a face tracker
            self.f_tracker = cv2.TrackerCSRT_create()
            self.f_tracker.init(frame, (x,y,w,h))
            print('Initiating a face tracker')

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
                        x,y,w,h = g
                        x,y,w,h = [0 if i < 0 else int(i) for i in [x,y,w,h]]
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
            print("Body Tracker update takes:{:.2f}s".format(time.time() - tic))
            if self.btrack_ok:
                x = int(position[0])
                y = int(position[1])
                w = int(position[2])
                h = int(position[3])
                x,y,w,h = [0 if i < 0 else i for i in [x,y,w,h]]

            else:
                print('Tracking Fail')
                return []
        
            #cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,255), 2)
            return x,y,w,h

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


def tracker_main(Tracker, frame, target_coordinates, custom_parameters):
    """
    Inputs come from the UI:
        Tracker: Object
            - An initated DetectionWidget
        frame: Numpy array
            -3 Channel with dimensions of (H, W, C); 

        target_coordinates: Tuple 
            - Desired Position of the object relative to the Frame where (0,0) is on the Top Left Corner.
            - Must not exceed H and W
        
        custom_parameters: dict
            - A dictionary containing the below parameters

            gamma: float(0.0~1.0)
                Senstivity of camera movement
            
            xMinE: float(0.0~1.0)
                Minmium error of Horizontal movement
            
            yMinE: float(0.0~1.0)
                Minmium error of Vertical movement

            ZoomValue : float (0.0 ~ 10.0)
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
        boundingBox: A list of 4 points in the format [x,y,x1,y2]
    """ 

    if Tracker is None:
        return 0
    
    #Unpack Dictionary and set Track Parameters
    for key in ['gamma', 'xminE', 'yminE', 'ZoomValue', 'y_trackState','autozoom_state','reset_trigger','track_type']:
        setattr(Tracker, key, custom_parameters[key])

    xVelocity, yVelocity = Tracker.main_track([frame, target_coordinates])
    track_coords = Tracker.get_bounding_boxes()
    
    output = {
    "xVelocity": xVelocity, 
    "yVelocity": yVelocity, 
    #"zoomSpeed":zoomSpeed,
    "boundingBox":[track_coords]
    }
    
    return output
  
if __name__ == '__main__':
    try:
        Tracker
    except NameError:
        Tracker = DetectionWidget()
        print('starting new tracker?')
    tracker_main(*args)
