import cv2
import time
import numpy as np

import os
import time
from tool.utils import timing_decorator

from pathlib import Path
import torch
CURR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
import torch
from ultralytics.yolo.engine.model import YOLO

class ObjectDetectionTracker:
    def __init__(self, yolo_model_path, device=0, **kwargs):
        self.model = YOLO(yolo_model_path, **kwargs)
        self.device = device # 0,1 for GPU; 'cpu' for CPU
    
    def detect(self, frame, **kwargs):
        # Perform detection
        frame = cv2.resize(frame, (640, 640))
        results = self.model.predict(frame, **kwargs)
        
        # if model is onnx, then results is a tuple
        if self.model.model.split('.')[-1] == 'onnx':
            detections = []
            for result in results:
                boxes = result.boxes.data.cpu().numpy().astype(int)
                # scores = result[1].cpu().numpy()
                # classes = result[2].cpu().numpy()
                
                for idx, box in enumerate(boxes):
                    detection = {
                        "box": (box[0], box[1], box[2], box[3]),
                        # "score": scores[idx],
                        # "class": classes[idx]
                    }
                    detections.append(detection)
                
        
        elif not results[0].keypoints:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            #ids = results[0].id.cpu().numpy().astype(int)
            #ids = [ None for _ in range(len(boxes))]
            detections = []
            for idx, box, in enumerate(boxes):
                detection = {
                    "box": (box[0], box[1], box[2], box[3]),
                }
                detections.append(detection)
        return detections
        
    @timing_decorator
    def track_with_class(self, source=None, stream=False, persist=False, **kwargs):
        """
        Perform object tracking on the input source using the registered trackers.

        Args:
            source (str, optional): The input source for object tracking. Can be a file path or a video stream.
            stream (bool, optional): Whether the input source is a video stream. Defaults to False.
            persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
            **kwargs (optional): Additional keyword arguments for the tracking process.

        Returns:
            (List[dict]): The tracking results, including the class ID.

        """
        start_time = time.time()
        #Reshape the frame to 640x640
        frame = cv2.resize(source, (640,640))
        results = self.model.track(source=frame, stream=stream, persist=persist, **kwargs)
        print(f"Time of Detect&Track {time.time() - start_time}")

        # Parsing out results
        start_time = time.time()
        detections = []
        for result in results:
            
                boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else [None for _ in range(len(boxes))]
                classes = result.boxes.cls.cpu().numpy().astype(int)
                
                if 0 not in classes:
                    continue

                # Filter out the results based on class value
                filtered_indices = [i for i, class_id in enumerate(classes) if class_id == 0]

                # Only loop over filtered indices
                for i in filtered_indices:
                    box, id, class_id = boxes[i], ids[i], classes[i]
                    detection = {
                        "box": (box[0], box[1], box[2], box[3]),
                        "id": id,
                        "class_id": class_id
                    }
                    detections.append(detection)
        print(f"Time of Parsing Results {time.time() - start_time}")
        return detections


class FastMTCNN(object):
    """Fast MTCNN implementation."""
    
    def __init__(self):
        from facenet_pytorch import MTCNN
        self.resize = 0.5
        self.mtcnn = MTCNN(margin = 14, factor = 0.6, keep_all= True,post_process=True, select_largest=False,device= 'cpu')

    def update(self, frame, body_coords = [], minConf = 0.9):
        """ 
        Checks if there are body coords.
        If None, then it assumes that you want to scan the entire frame
        If there are body_coords then it cuts out the frame with only the body detections
        """
        print('Running a scoped MTCNN')
        W, H = frame.shape[:2]

        width_minimum = 30
        height_minimum = 40
        if self.resize != 1:
            frame = cv2.resize(frame, (int(frame.shape[1] * self.resize), int(frame.shape[0] * self.resize)))
        
        boxes, results = self.mtcnn.detect(frame, landmarks=False)

        if boxes is None:
            return []
        
        #Filter out the results if there are faces less than minConf
        #boxes = [i for (i,j) in zip(boxes,results) if j >= minConf]

        elif len(boxes) > 0:
            #Resize facebounding box to original size
            boxes = np.multiply(boxes,(1/self.resize))
            too_small = []
            for f,b in enumerate(boxes):
                x_a,y_a,x2_a,y2_a = b
                w_a = x2_a - x_a
                h_a = y2_a - y_a

                if w_a < width_minimum or h_a < height_minimum:
                    too_small.append(f)
                else:
                    pass

            if len(too_small) > 0:
                boxes = np.delete(boxes, too_small, axis = 0)

            if len(boxes) > 0:        
                (x,y,x2,y2) = boxes[np.argmax(results)]
            else:
                return []
            
        else:
            return []

        w = x2-x
        h = y2-y
        x,y,w,h = [int(x) if x > 0 else 0 for x in [x,y,w,h]]
        boxes = [x,y,w,h]
        return boxes

    @timing_decorator
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


class Yolo_v4TINY(object):
    CONFIDENCE_THRESHOLD = 0.2
    NMS_THRESHOLD = 0.4
    COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
    
    def __init__(self):
        labelsPath = os.path.join(CURR_PATH, "models\coco.names")
        weightsPath = os.path.join(CURR_PATH, "models\yolov4-tiny.weights")
        configPath = os.path.join(CURR_PATH, "models\yolov4-tiny.cfg")


        with open(labelsPath, 'r') as f: 
            self.LABELS = f.read().strip().split("\n")
        self.net = cv2.dnn.readNet(weightsPath, configPath)
        # self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(416, 416), scale=1/255)

    def update(self, frames, minConf = 0.4, threshold = 0.5):
        """TODO
        Fix the class parser. It's ugly right now
        """
        #ptvsd.debug_this_thread()
        idxs = []
        self.boxes = []
        self.confidences = []
        self.classIDs = []
        (H, W) = frames.shape[:2]
        classIDs, scores, boxes = self.model.detect(frames, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)

        if len(classIDs) > 0:
            mask = classIDs == 0
            if True in mask.squeeze():
                curr_score = 0
                for index, p in enumerate(mask):
                    if p == True and scores[index] >= curr_score:
                        curr_score = scores[index]
                        top = index
                        self.confidences.append(curr_score)
                box = boxes[top]
                (X, Y, width, height) = box.astype("int")

                #self.classIDs.append(classIDs)
                self.boxes.append([int(X), int(Y), int(width), int(height)])
                #self.detections.append([x,y,x+width,y+height,scores])
                #idxs = cv2.dnn.NMSBoxes(self.boxes, self.confidences, minConf, threshold)
                return None, self.boxes, None, None, None,self.confidences
            else:
                return None, [], None, None, None, None
        else:
            return None, [], None, None, None, None
