import cv2
import os

CURR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
from ultralytics.yolo.engine.model import YOLO
from config import CONFIG

class ObjectDetectionTracker:
    def __init__(self, yolo_model_path, device=0, use_csrt=False, **kwargs):
        self.model = YOLO(yolo_model_path, **kwargs)
        self.device = device # 0,1 for GPU; 'cpu' for CPU
        self.use_csrt = use_csrt
        self.p_track_count = 0
        self.p_tracker = None
        self.lost_tracking_count = 0
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
    
    def detect(self, frame, imgsz=640, **kwargs):
        original_h, original_w = frame.shape[:2]  # Save the original frame size
        scale_x = original_w / imgsz
        scale_y = original_h / imgsz

        # Reshape the frame to 640x640
        frame_resized = cv2.resize(frame, (imgsz, imgsz))
        results = self.model.predict(frame_resized, **kwargs)

        detections = []
        for result in results:
            if self.model.model.split('.')[-1] == 'onnx':
                boxes = result.boxes.data.cpu().numpy().astype(int)
            else:
                classes = result.boxes.cls.cpu().numpy().astype(int)
                if 0 not in classes:
                    continue
                boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                filtered_indices = [i for i, class_id in enumerate(classes) if class_id == 0]
                boxes = [boxes[i] for i in filtered_indices]

            for idx, box in enumerate(boxes):
                scaled_box = (box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y)
                detection = {
                    "box": scaled_box,
                    # Additional information if needed
                }
                detections.append(detection)

        return detections
    
    
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
                    detection = {
                        "box": scaled_box,
                        # Additional information if needed
                    }
                    x,y,w,h = scaled_box

            #Start a face tracker
            #self.p_tracker = cv2.TrackerKCF_create()
            self.p_tracker = cv2.TrackerCSRT_create(self.csrt_params)
            self.p_tracker.init(frame, (x,y,w,h))

        self.p_coords = x,y,w,h
        return [{
            'box': (x,y,w,h),
            'class_id': 0
        }]
        
    def reset_tracker(self):
        self.p_tracker = None
        