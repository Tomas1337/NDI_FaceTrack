import cv2
import os, time
import numpy as np
from tool.utils import overlap_check
from .onnx_yolov8 import BoxResult, Result
from config import TRACK_TYPE_DICT
CURR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
from config import CONFIG

class FastMTCNN(object):
    """Fast MTCNN implementation."""
    def __init__(self):
        from .pytorch_mtcnn.mtcnn import MTCNN
        self.resize = 0.5
        self.mtcnn = MTCNN(margin=14, factor=0.709, keep_all=False, post_process=True, select_largest=False, device='cpu')

    def update(self, frame, minConf=0.9, iou_threshold=0.45, nms_threshold=0.5):
        # print('Running a scoped MTCNN')
        width_minimum = 30
        height_minimum = 40
        original_shape = frame.shape
        resize_ratio = self.resize if self.resize != 1 else 1
        resized_frame = frame
        if resize_ratio != 1:
            resized_frame = cv2.resize(frame, (int(frame.shape[1] * resize_ratio), int(frame.shape[0] * resize_ratio)))

        boxes, results = self.mtcnn.detect(resized_frame, landmarks=False)
        if boxes is None or len(boxes) == 0:
            return []

        # Resize the bounding boxes to match the resized frame
        boxes = (boxes / resize_ratio).astype(int)

        # Filter boxes by confidence and size
        filtered_boxes = [box for box, res in zip(boxes, results)
                        if res >= minConf and (box[2] - box[0]) >= width_minimum and (box[3] - box[1]) >= height_minimum]
        
        result_items = [Result(boxes=BoxResult(xyxy=np.array([box[0], box[1], box[2], box[3]])[np.newaxis, :],
                                            cls=np.array([0]).astype(int),  # Assuming class ID 0 for faces
                                            conf=np.array([res])))
                        for box, res in zip(boxes, results) if res >= minConf]

        return result_items

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
    def __init__(self, yolo_model_path, device=0, use_csrt=False, track_type='face', 
            debug_show=False, overlap_frame_check=None, **kwargs):
        
        self.general_detector = GeneralDetector(yolo_model_path, track_type)
        self.device = device # 0,1 for GPU; 'cpu' for CPU
        self.use_csrt = use_csrt
        self.p_track_count = 0
        self.p_tracker = None
        self.lost_tracking_count = 0
        self.check_count = 0 
        self.track_type = track_type
        self.overlap_frames = overlap_frame_check
        self.overlap_check_count = 0
        self.debug_show = debug_show
        self.bounding_box_edge_padding = 100
        self.p_coords = None
        self.no_detection_count = 0
        
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

    
    def __del__(self):
        cv2.destroyAllWindows()
        
    def track_with_csrt(self, source=None, imgsz=640, **kwargs):
        self.track_type = kwargs.get('track_type', self.track_type)
        self.general_detector.track_type = TRACK_TYPE_DICT.get(self.track_type, 'face')
        frame = source

        if self.p_tracker:
            ok, position = self.p_tracker.update(frame)
            if ok:
                # x, y, x2, y2 = map(int, position)
                # w = int(x2 - x)
                # h = int(y2 - y)

                x, y, w, h = map(int, position)
                x,y,w,h = [0 if i < 0 else int(i) for i in [x,y,w,h]]
                self.p_track_count = 0
                
                if self.debug_show:
                    f_copy = frame.copy()
                    cv2.rectangle(f_copy, (x,y), (x+w,y+h), (255,0,255), 1)
                    cv2.imshow('CSRT Tracking', f_copy)
                    cv2.waitKey(1)
                
                w = int(w * (1 - 0.5)) if w > frame.shape[1]*0.5 else w
                h = int(h * (0.75)) if h > frame.shape[0]*0.75 else h
                # Overlap check, if needed
                self.overlap_check_count += 1
                if self.overlap_frames is not None and self.overlap_check_count >= self.overlap_frames:
                    overlap_result = self.check_overlap(frame, **kwargs)
                    if overlap_result == None:
                        pass
                    elif overlap_result == False:
                        self.p_tracker = None
                    elif overlap_result == True:
                        self.overlap_check_count = 0
                    
            else:
                self.lost_tracking_count += 1
                if self.p_track_count > 5:
                    self.p_tracker = None
                else:
                    self.p_track_count += 1
                return []
                

        elif self.p_tracker is None:
            results = self.general_detector.detect(frame, minConf=0.9)
            if not results:
                return []
            for result in results:
                classes =  result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy().astype(float)

                if 0 not in classes:
                    if result == results[-1]:
                        return []
                    else:
                        continue

                boxes = result.boxes.xyxy.cpu().numpy().astype(int) 
                filtered_indices = [i for i, class_id in enumerate(classes) if class_id == 0]
                boxes = [boxes[i] for i in filtered_indices]
                confidences = [confidences[i] for i in filtered_indices]
                boxes = [box for _, box in sorted(zip(confidences, boxes), key=lambda pair: pair[0], reverse=True)]
                    
            if not boxes:
                return []
            
            x1,y1,x2,y2 = [int(i) for i in boxes[0]]
            x,y,w,h = x1,y1,x2-x1,y2-y1
            
            # # Reducing the bounding box size around the center
            w = int(w * (1 - 0.5)) if w > frame.shape[1]*0.5 else w
            h = int(h * (0.75)) if h > frame.shape[0]*0.75 else h
            self.p_tracker = cv2.TrackerCSRT_create(self.csrt_params)    
            self.p_tracker.init(frame, (x,y,x+w,y+h))
            
            if self.debug_show:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 1)
                cv2.imshow('CSRT Tracker Create', frame)
                cv2.waitKey(1)
                
            x,y,w,h=map(int,(x,y,w,h))
            
        self.p_coords = x,y,w,h
        return [{
            'box': (x,y,w,h),
            'class_id': 0
        }]
        
    def reset_tracker(self):
        self.p_tracker = None
        
    def increment_lost_tracking(self):
        self.lost_tracking_count += 1
        print(f"Overlap is None. Lost tracking count is {self.lost_tracking_count}")
        if self.lost_tracking_count >= 7:
            print("Overlap is None. Resetting tracker")
            self.p_tracker = None
            self.lost_tracking_count = 0
            self.overlap_check_count = 0
            return False
        return True
            
    def check_overlap(self, frame, **kwargs):
        results = self.general_detector.detect(frame, verbose=False, minConf=0.4, **kwargs)
        
        # No detections on current frame
        if not results:
            return self.increment_lost_tracking()

        boxes = None
        for result in results:
            classes = result.boxes.cls.cpu().numpy().astype(int)
            if 0 not in classes:
                continue

            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy().astype(float)
            filtered_classes = [i for i, class_id in enumerate(classes) if class_id == 0]
            filtered_confidences = [i for i in filtered_classes if confidences[i] > 0.5]
            filtered_indices = list(set(filtered_classes).intersection(filtered_confidences))
            boxes = [boxes[i] for i in filtered_indices]
            confidences = [confidences[i] for i in filtered_indices]

            # sort by the confidence
            boxes = [box for _, box in sorted(zip(confidences, boxes), key=lambda pair: pair[0], reverse=True)]

        if not boxes:
            return None

        x1, y1, x2, y2 = [int(i) for i in boxes[0]]
        detected_box = (x1, y1, x2 - x1, y2 - y1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw detected box in red
        px, py, pw, ph = self.p_coords
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (255, 0, 0), 2)  # Draw self.p_coords in blue

        if self.debug_show:
            cv2.imshow('Overlap', frame)
            cv2.waitKey(1)
            
        overlap = overlap_check(self.p_coords, detected_box)
        if overlap < 0.10:
            return self.increment_lost_tracking()
        else:
            print(f"Overlap SUCCESS: {round(overlap*100,2)}%. Resetting lost tracking count")
            self.lost_tracking_count = 0
        return True

class GeneralDetector:
    def __init__(self, yolo_model_path, track_type='fallback'):
        if yolo_model_path.endswith('.pt'):
            from ultralytics.yolo.engine.model import YOLO
        else:
            from object_tracking.onnx_yolov8 import ONNX_YOLOv8 as YOLO
        self.yolo_model = YOLO(yolo_model_path)
        self.face_detector = FastMTCNN()
        self.track_type = track_type

    def detect(self, frame, minConf=0.6, **kwargs):
        if self.track_type == 'face':
            return self.face_detector.update(frame, minConf=0.2)
        elif self.track_type == 'body':
            return self._detect_body(frame, minConf)
        else: # "Fallback face > person"
            face_result = self.face_detector.update(frame, minConf)
            if face_result:
                return face_result
            return self._detect_body(frame, minConf)

    def _detect_body(self, frame, minConf):
        results = self.yolo_model.predict(frame)
        return results
        # if len(results) == 0:
        #     return []

        # boxes = []
        # for result in results:
        #     classes = result.boxes.cls.cpu().numpy().astype(int)
        #     if 0 not in classes:
        #         continue

        #     detected_boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        #     confidences = result.boxes.conf.cpu().numpy().astype(float)
        #     filtered_indices = [i for i, class_id in enumerate(classes) if class_id == 0 and confidences[i] >= minConf]
        #     boxes += [detected_boxes[i] for i in filtered_indices]

        # return [{'box': (x1, y1, x2 - x1, y2 - y1), 'class_id': 0} for x1, y1, x2, y2 in boxes]