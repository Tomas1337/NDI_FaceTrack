

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

        #Object Detectors
        #self.face_obj = FastMTCNN()
        self.face_obj = FastDeepSORT() #TRY THIS FIRST
        self.body_obj = Yolo_v4TINY()

        #Slider and button Values
        self.track_type = 0
        self.y_track_state = True
        self.gamma = float(CONFIG.getint('camera_control', 'gamma_default'))/10
        self.xMinE = float(CONFIG.getint('camera_control', 'horizontal_error_default'))/10
        self.yMinE = float(CONFIG.getint('camera_control', 'vertical_error_default'))/10
       
        self.gamma = (CONFIG.getint('camera_control', 'gamma_default'))/10
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
    
    def main_track(self, frame):
        """
        Takes in a list where the first element is the frame
        The second element are the target coordinates of the tracker
        """
        self.track_coords = []
        self.set_focal_length(self.zoom_value)
        start = time.time()

        if self.reset_trigger is True:
            self.reset_tracker()
            self.reset_trigger = False
        
        frame = np.array(frame)

        (H, W) = frame.shape[:2]
        centerX = self.target_coordinate_x
        centerY = self.target_coordinate_y
        
        #Trackers

        if self.track_type is None:
            return (None, None)
            
        elif self.track_type == 0:
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

        print(f"Speed has been dampened from {self.x_prev_speed} to {x_speed} and {self.y_prev_speed} to {y_speed} by proib")
        return (x_speed, y_speed)

    def face_tracker(self, frame):
        if self.f_tracker:
            ok, position = self.f_tracker.update(frame)
            
            if self.frame_count % 300 == 0:
                if self.lost_tracking_count > 0: # Prevent from going negative
                    self.lost_tracking_count -= 1
                    
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

            if self.frame_count % 20 == 0: 
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
