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
        self.resize = 1
        self.mtcnn = MTCNN(margin = 14, factor = 0.709, keep_all= True,post_process=True, select_largest=False,device= 'cpu')
    
    def update(self, frame, minConf=0.9, iou_threshold=0.45, nms_threshold=0.5):
        print('Running a scoped MTCNN')
        width_minimum = 30
        height_minimum = 40
        if self.resize != 1:
            frame = cv2.resize(frame, (int(frame.shape[1] * self.resize), int(frame.shape[0] * self.resize)))

        boxes, results = self.mtcnn.detect(frame, landmarks=False)
        if boxes is None or len(boxes) == 0:
            return []

        # Resize the bounding boxes
        boxes = (boxes * (1 / self.resize)).astype(int)

        # Filter boxes by confidence and size
        filtered_boxes = []
        scores = []
        for box, res in zip(boxes, results):
            if res >= minConf and (box[2] - box[0]) >= width_minimum and (box[3] - box[1]) >= height_minimum:
                filtered_boxes.append(box)
                scores.append(res)

        # Apply Non-Maximum Suppression (NMS) to filter the results
        result_boxes = cv2.dnn.NMSBoxes(filtered_boxes, scores, minConf, iou_threshold, nms_threshold)
        result_items = []

        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = filtered_boxes[index]
            xyxy = np.array([
                box[0],
                box[1],
                box[2],
                box[3]
            ])
            boxes_obj = BoxResult(
                xyxy=xyxy[np.newaxis, :],
                cls=np.array([0]).astype(int),  # Assuming class ID 0 for faces
                conf=np.array([scores[index]])
            )
            result_item = Result(boxes=boxes_obj)
            result_items.append(result_item)

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
        self.track_type = kwargs.get('track_type', self.track_type)
        self.general_detector.track_type = TRACK_TYPE_DICT.get(self.track_type, 'face')
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
                
                if self.debug_show:
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,255), 1)
                    cv2.imshow('P Tracker', frame)
                    cv2.waitKey(1)
                
                ## Overlap check, if needed
                self.overlap_check_count += 1
                if self.overlap_frames is not None and self.overlap_check_count >= self.overlap_frames:
                    if not self.check_overlap(frame, **kwargs):
                        return []
                    
            else:
                self.lost_tracking_count += 1
                if self.p_track_count > 5:
                    self.p_tracker = None
                else:
                    self.p_track_count += 1
                return []
                

        elif self.p_tracker is None:
            results = self.general_detector.detect(frame, minConf=0.9)
            if len(results) == 0:
                return []
            for result in results:
                classes =  result.boxes.cls.cpu().numpy().astype(int)
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
                  
                    # Reducing the bounding box size around the center
                    w = int(w * (1 - 0.5)) if w > frame.shape[1]*0.5 else w
                    h = int(h * (1 - 0.5)) if h > frame.shape[0]*0.5 else h
                    
      
            self.p_tracker = cv2.TrackerCSRT_create(self.csrt_params)    
            self.p_tracker.init(frame, (x,y,w,h)) 
            
        self.p_coords = x,y,w,h
        return [{
            'box': (x,y,w,h),
            'class_id': 0
        }]
        
    def reset_tracker(self):
        self.p_tracker = None
        
    def check_overlap(self, frame, **kwargs):
        results = self.general_detector.detect(frame, verbose=False, minConf=0.4, **kwargs)
        
        # No detections on current frame
        if not results:
            return None

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
            self.lost_tracking_count += 1
            print(f"Overlap is {overlap}. Lost tracking count is {self.lost_tracking_count}")
            if self.lost_tracking_count >= 10:
                print(f"Overlap is {overlap}. Resetting tracker")
                self.p_tracker = None
                self.lost_tracking_count = 0
                self.overlap_check_count = 0
                return False
        else:
            self.lost_tracking_count = 0
        self.overlap_check_count = 0
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

    def detect(self, frame, minConf=0.9, **kwargs):
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
