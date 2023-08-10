import cv2
import os, time
import numpy as np
from facenet_pytorch import MTCNN
from tool.utils import overlap_check

CURR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
from ultralytics.yolo.engine.model import YOLO
from config import CONFIG

class FastMTCNN(object):
    """Fast MTCNN implementation."""
    
    def __init__(self):
        # from facenet_pytorch import MTCNN
        self.resize = 0.5
        self.mtcnn = MTCNN(margin = 14, factor = 0.6, keep_all= True,post_process=True, select_largest=False,device= 'cpu')

    def update(self, frame ,minConf=0.9):
        """ 
        Checks if there are body coords.
        If None, then it assumes that you want to scan the entire frame
        If there are body_coords then it cuts out the frame with only the body detections
        """
        print('Running a scoped MTCNN')
        width_minimum = 30
        height_minimum = 40
        if self.resize != 1:
            frame = cv2.resize(frame, (int(frame.shape[1] * self.resize), int(frame.shape[0] * self.resize)))

        boxes, results = self.mtcnn.detect(frame, landmarks=False)
        if boxes is None:
            return []

        # Resize the bounding boxes
        boxes = (boxes * (1 / self.resize)).astype(int)
        
        # Filter boxes by confidence and size
        boxes = [box for box, res in zip(boxes, results) 
                if res >= minConf and (box[2] - box[0]) >= width_minimum and (box[3] - box[1]) >= height_minimum]

        # Return the box with maximum confidence (if you need all boxes, adjust this)
        if boxes:
            return boxes[np.argmax(results)]
        
        return []

    def get_all_locations(self, frame, minConf = 0.6):
        start_time = time.time()
        if self.resize != 1:
            frame = cv2.resize(frame, (int(frame.shape[1] * self.resize), int(frame.shape[0] * self.resize)))
        boxes, results = self.mtcnn.detect(frame, landmarks=False)
        print('Running a MTCNN All')
        print(f'Face Detection Inference is: {time.time()-start_time}. Return: {boxes} faces')
        if boxes is None:
            return []
        elif len(boxes) > 0:
            # Resize facebounding box to original size
            boxes = np.multiply(boxes,(1/self.resize))
            return boxes

class ObjectDetectionTracker:
    def __init__(self, yolo_model_path, device=0, use_csrt=False, track_type='person', overlap_frames=None, **kwargs):
        self.model = YOLO(yolo_model_path, **kwargs)
        self.face_detector = FastMTCNN() if track_type == 'face' else None
        self.device = device # 0,1 for GPU; 'cpu' for CPU
        self.use_csrt = use_csrt
        self.p_track_count = 0
        self.p_tracker = None
        self.lost_tracking_count = 0
        self.check_count = 0 
        self.track_type = track_type
        self.overlap_frames = overlap_frames
        self.overlap_check_count = 0
        
        
        if self.use_csrt:
            params = cv2.TrackerCSRT_Params()

            # Read the configuration from the INI file
            csrt_params = CONFIG['csrt_parameters']
  
            for param, value in csrt_params.items():
                if param in ["use_channel_weights", "use_color_names", "use_gray", "use_hog", "use_rgb", "use_segmentation"]:
                    setattr(params, param, CONFIG.getboolean('csrt_parameters', param))
                elif param == "window_function":
                    setattr(params, param, value)
                elif param in ['admm_iterations', 'background_ratio','histogram_bins','num_hog_channels_used','number_of_scales']:
                    setattr(params, param, int(value))
                else:
                    setattr(params, param, float(value))
            
            self.csrt_params = params
            self.tracking=False
    
    def track_with_csrt(self, source=None, imgsz=640, **kwargs):
        original_h, original_w = source.shape[:2]  # Save the original frame size
        scale_x = original_w / imgsz
        scale_y = original_h / imgsz
        frame = cv2.resize(source, (imgsz, imgsz))

        if self.p_tracker:
            ok, position = self.p_tracker.update(frame)
            if ok:
                x, y, w, h = map(int, position)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 1)
                x,y,w,h = [0 if i < 0 else i for i in [x,y,w,h]]
                x,y,w,h = [int(i) for i in [x,y,w,h]]
                self.p_track_count = 0
                
                # Overlap check, if needed
                if self.overlap_frames is not None and self.overlap_check_count >= self.overlap_frames:
                    results = self.model.predict(frame, **kwargs)
                    if len(results) == 0:
                        return []
                    
                    boxes = None
                    for result in results:
                        classes = result.boxes.cls.cpu().numpy().astype(int)
                        if 0 not in classes:
                            continue

                        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                        confidences = result.boxes.conf.cpu().numpy()
                        filtered_classes = [i for i, class_id in enumerate(classes) if class_id == 0]
                        filtered_confidences = [i for i in filtered_classes if confidences[i] > 0.5]
                        filtered_indices = list(set(filtered_classes).intersection(filtered_confidences))
                        boxes = [boxes[i] for i in filtered_indices]
                        confidences = [confidences[i] for i in filtered_indices]
                        
                        # sort by the confidence
                        boxes = [box for _, box in sorted(zip(confidences, boxes), key=lambda pair: pair[0], reverse=True)]
                    
                    if boxes is None or len(boxes) == 0:
                        return []

                    #scaled_box = (box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y)
                    x1, y1, x2, y2 = [int(i) for i in boxes[0]]

                    detected_box = (x1, y1, x2 - x1, y2 - y1)
                    overlap = overlap_check(self.p_coords, detected_box)
                    if overlap < 0.30:
                        print(f"Overlap is {overlap}. Resetting tracker")
                        self.p_tracker = None
                        self.overlap_check_count = 0
                        return []
                    self.overlap_check_count = 0
                else:
                    self.overlap_check_count += 1

                
            else:
                self.lost_tracking_count += 1
                if self.p_track_count > 5:
                    self.p_tracker = None
                    return []
                else:
                    self.p_track_count += 1
                    return []
                

        elif self.p_tracker is None:
            results = self.model.predict(frame, **kwargs)
            for result in results:
                classes = result.boxes.cls.cpu().numpy().astype(int)
                if 0 not in classes:
                    if result == results[-1]:
                        return []
                    else:
                        continue

                boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                filtered_indices = [i for i, class_id in enumerate(classes) if class_id == 0]
                boxes = [boxes[i] for i in filtered_indices]
                    
                if len(boxes) == 0:
                    return []
                
                for idx, box in enumerate(boxes):
                    scaled_box = (box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y)
                    scaled_box = [int(i) for i in scaled_box]
                    x1,y1,x2,y2 = scaled_box
                    x,y,w,h = x1,y1,x2-x1,y2-y1
                    

            #Start CSRT tracker
            #self.p_tracker = cv2.TrackerKCF_create()
            self.p_tracker = cv2.TrackerCSRT_create(self.csrt_params)
            # modified_h = int(0.69 * original_h) if h > int(0.69 * original_h) else h
            # modified_w = int(0.5 * w) if x + int(0.5 * w) < original_w else x
            # h = modified_h
            # w = modified_w
            self.p_tracker.init(frame, (x,y,w,h)) # We need to modify the height so that it is not too tall. make it 69% of the max height of the frame
            

        self.p_coords = x,y,w,h
        return [{
            'box': (x,y,w,h),
            'class_id': 0
        }]
        
    def reset_tracker(self):
        self.p_tracker = None







        