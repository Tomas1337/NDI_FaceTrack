
import sys
import cv2
sys.path.append('.')
from face_tracking.objcenter import *
from face_tracking.camera_control import PTZ_Controller_Novel
from decimal import Decimal
class Detection():
    """
    Class object used for detecing tracking objects in the frame.
    Calling main_track(frame) will return coordinates and return which object using it's ID 
    The detection object has certain fallbacks and has 'memory' of which frames it's seen.
    Default priority tracking is in the order of the first ID to last ID in the priority_list variable
    Default behavior is 
        1. Try to track firstID object. Return object coordinates if successful
        2. If cannot detect firstID object, result to calling secondID object.
        3. Continue cascading fallback until last object has been reached. 
        4. If no objects have been detected, scan for each object sequentially.       
    """
    def __init__(self):
        FRAME_WIDTH = 640
        FRAME_HEIGHT = 360
        self.last_loc = [FRAME_WIDTH//2, FRAME_HEIGHT//2]
        self.track_coords = []
        self.focal_length = 129
        self.reset_trigger = False
        self.btrack_ok = False
        self.center_coords = (FRAME_WIDTH//2, FRAME_HEIGHT//2)
        self.track_type = 0
        self.track_trigger = False

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
        self.face_obj = FastMTCNN()
        self.body_obj = Yolo_v4TINY()

        self.ZoomValue = 0.0
        self.autozoom_state = True
        self.face_lock_state = False
        self.face_coords = []

        #Speed Tracker
        self.x_prev_speed = None
        self.y_prev_speed = None
    
    def main_track(self, frame, track_type = 0):
        """
        Begins main tracking module.
        Handles descision between face or body Tracking
        Handles sending commands to camera

        Args:
            frame (np.ndarray): Frame to track
        """
        self.track_coords = []
        self.set_focal_length(self.ZoomValue)
        start = time.time()

        if self.reset_trigger is True:
            self.reset_tracker()
            self.reset_trigger = False

        (H, W) = frame.shape[:2]
        centerX = self.center_coords[0]
        centerY = self.center_coords[1]
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
            self.info_status('Tracking Face')
        except (ValueError, TypeError, UnboundLocalError) as e:
            try:
                x,y,w,h = body_coords
                self.track_coords = [x,y,x+w,y+h]
                self.info_status('Tracking Body')
                self.overlap_counter = 0
            except (ValueError, UnboundLocalError) as e:
                pass
            finally:
                face_track_flag = False
        
        #Normal Tracking
        if len(self.track_coords) > 0:
            [x,y,x2,y2] = self.track_coords
            objX = int(x+(x2-x)//2)
            objY = int(y+(y2-y)//3)
            self.last_loc = (objX, objY)
            self.lost_tracking = 0

        #Initiate Lost Tracking sub-routine
        else:
            if self.lost_tracking > 100 and self.lost_tracking < 500 and self.autozoom_state:
                self.info_status("Lost Tracking Sequence Secondary")
                #self.CameraZoomControlSignal.emit(-0.7)

            elif self.lost_tracking < 20:
                objX = self.last_loc[0]
                objY = self.last_loc[1]
                self.lost_tracking += 1
                # logger.debug('Lost tracking. Going to last known location of object')
                # self.info_status.emit("Lost Tracking Sequence Initial")

            else:
                objX = centerX
                objY = centerY
                self.info_status("Lost Object. Recenter subject")
                self.lost_tracking += 1
                if self.lost_tracking < 500 and self.lost_tracking%100:
                    self.b_tracker = None


        self.set_tracked_coords(objX, objY)
        self.frame_count += 1

        # Convert to qimage then send to display to GUI
        self.image = self.get_qimage(frame)
        self.DisplayVideoSignal.emit(self.image)

        # CAMERA CONTROL
        x_controller = PTZ_Controller_Novel(self.focal_length, self.gamma)
        x_speed = x_controller.omega_tur_plus1(objX, centerX, RMin = 0.1)

        y_controller = PTZ_Controller_Novel(self.focal_length, self.gamma)
        y_speed = y_controller.omega_tur_plus1(objY, centerY, RMin = 0.1, RMax=7.5)

        x_speed = Decimal(self._pos_error_thresholding(W, x_speed, centerX, objX, self.xMinE)).quantize(Decimal("0.0001"))
        y_speed = Decimal(self._pos_error_thresholding(H, y_speed, centerY, objY, self.yMinE)).quantize(Decimal("0.0001"))

        if self.y_trackState is False:
            y_speed = 0

        if (self.x_prev_speed == x_speed and 
                self.y_prev_speed == y_speed and x_speed <= 0.15):
            pass

        else:
            if self.sender().gui.face_track_button.isChecked() is True:
                self.CameraControlSignal(x_speed, y_speed)
            else:
                self.CameraControlSignal(0.0, 0.0)
            
        self.x_prev_speed = x_speed
        self.y_prev_speed = y_speed
        return (objX, objY)
            
    def face_tracker(self, frame):
        if not self.f_tracker is None:
            tic = time.time()
            ok, position = self.f_tracker.update(frame)
            #print("Face Tracker update takes:{:.2f}s".format(time.time() - tic))
            if self.frame_count % 300== 0:
                #logger.debug(f'Running a {(300/30)}s check')
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
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 1)
                x,y,w,h = [0 if i < 0 else i for i in [x,y,w,h]]
                self.f_track_count = 0
                face_coords = x,y,w,h
                
            else:
                if self.f_track_count > 5:
                    #logger.debug('Lost face') 
                    self.info_status('Refreshing Face Detector')     
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
            if self.frame_count % 20 == 0:
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

        self.face_coords = x,y,w,h
        return x,y,w,h

    def body_tracker(self, frame):
        #ptvsd.debug_this_thread()
        if self.b_tracker is None:
            #Detect Objects using YOLO every 1 second if No Body Tracker    
            boxes = []
            if self.frame_count % 15 == 0:
                (idxs, boxes, _, _, classIDs, confidences) = self.body_obj.update(frame)
               

            if len(boxes) <= 0:
                return []

            elif len(boxes) == 1:
                x,y,w,h = boxes[np.argmax(confidences)]
                x,y,w,h = [0 if i < 0 else int(i) for i in [x,y,w,h]]

            elif len(boxes) > 1 and len(self.face_coords) >= 1:
                for i, box in enumerate(boxes):
                    if self.overlap_metric(box, self.face_coords) >= 0.5:
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
            #print("Body Tracker update takes:{:.2f}s".format(time.time() - tic))
            if self.btrack_ok:
                x = int(position[0])
                y = int(position[1])
                w = int(position[2])
                h = int(position[3])
                x,y,w,h = [0 if i < 0 else i for i in [x,y,w,h]]

            else:
                #logger.debug('Tracking Fail')
                return []
        
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,255), 2)
            return x,y,w,h

    def set_focal_length(self, zoom_level):
        """
        Range is 129mm to 4.3mm
        When Zoom Level is 0, it is = 129mm
        When Zoom level is 10, it is = 4.3mm
        Values in between might not be linear so adjust from there
        """
        zoom_dict = {0:129, 1:116.53, 2:104.06, 3:91.59, 4: 79.12,5:66.65, 6:54.18, 7:41.71, 8:29.24, 9:16.77, 10:4.3}
        self.focal_length = zoom_dict.get(int(zoom_level))

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
            
    def set_tracked_coords(self, objX: float, objY: float):
            self.X_tracked_coord = objX
            self.Y_tracked_coord = objY

    def get_track_coords(self):
        return (self.X_tracked_coord, self.Y_tracked_coord)

    def overlap_metric(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        # BoxA and BoxB usually comes in x,y,w,h
        # Translate into x,y,x1,y1
        def _translate(box):
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
        #iou = interArea / float(boxAArea + boxBArea - interArea)
        iou = interArea/boxBArea
        return iou

    def info_status(self, status: str):
        print(status)

    def CameraControlSignal(self, X_speed:float, Y_speed: float):
        print(X_speed, Y_speed)
    
    def get_current_frame(self):
        return self.image

    def reset_tracker(self):
        self.b_tracker = None
        self.f_tracker = None


def main():
    detection = Detection()
    long_video = "D:\Downloads\When life is unfair trust God - Ptr. Joby Soriano-TZeCoY82wVQ.mkv"
    short_video = "test/test_videos/faces.mp4"
    cap = cv2.VideoCapture(long_video)
    ret, frame = cap.read()
    x_coord = 0
    y_coord = 0
    while(True):
        ret, frame = cap.read()

        if ret:
            height, width, layers = frame.shape
            new_h = int(height / 3)
            new_w = int(width / 3)
            frame = cv2.resize(frame, (new_w, new_h))
            #coordinates, objectId = detection.main_track(frame)
            x_coord, y_coord = detection.main_track(frame)
            cv2.circle(frame, (x_coord, y_coord), 4, (255, 0, 0), thickness = 5)
            cv2.imshow('frame',frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            pass
        
if __name__ == '__main__':
    main()


