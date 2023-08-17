import onnxruntime
import cv2
import numpy as np
import time
import os

class NumpyWrapper:
    def __init__(self, array):
        self.array = array

    def cpu(self):
        return self

    def numpy(self):
        return self

    def astype(self, dtype):
        return self.array.astype(dtype)

class ResultItem:
    def __init__(self, xyxy, cls, conf):
        self.boxes = self.Boxes(xyxy, cls, conf)

    class Boxes:
        def __init__(self, xyxy, cls, conf):
            self.xyxy = xyxy
            self.cls = cls
            self.conf = conf

        def cpu(self):
            # If you need to do something specific for moving to CPU, do it here.
            return self
        
        
class BoxResult:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = NumpyWrapper(xyxy)
        self.cls = NumpyWrapper(cls)
        self.conf = NumpyWrapper(conf)

    def cpu(self):
        return self

class Result:
    def __init__(self, boxes):
        self.boxes = boxes
        
CLASSES = ['person']

class ONNX_YOLOv8:
    def __init__(self, model_path=None, confidence_thres=0.75, iou_thres=0.45, nms_thres=0.5):
        if model_path is None:
            base_path = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_path, '..', 'models', 'yolov8n_640.onnx')
        opt_session = onnxruntime.SessionOptions()
        opt_session.enable_mem_pattern = False
        opt_session.enable_cpu_mem_arena = False
        opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        EP_list = ['TensorrtExecutionProvider', 'CPUExecutionProvider']
        self.ort_session = onnxruntime.InferenceSession(model_path, providers=EP_list)
        self.input_names = [input.name for input in self.ort_session.get_inputs()]
        self.output_names = [output.name for output in self.ort_session.get_outputs()]
        self.model_type = 'onnxruntime'
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.nms_thres = nms_thres

    def preprocess(self, original_image):
        input_shape = self.ort_session.get_inputs()[0].shape[2:]
        original_shape = original_image.shape[:2]
        resized_image = cv2.resize(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), tuple(input_shape))
        scaling_factor_x = original_shape[1] / input_shape[1]
        scaling_factor_y = original_shape[0] / input_shape[0]
        resized_image = resized_image.transpose(2, 0, 1)
        resized_image = resized_image.astype(np.float32) / 255.0
        resized_image = resized_image.reshape(1, resized_image.shape[0], resized_image.shape[1], resized_image.shape[2])
        return resized_image, scaling_factor_x, scaling_factor_y
    
    def postprocess(self, outputs, scaling_factor_x, scaling_factor_y):
        outputs = np.transpose(np.squeeze(outputs[0]))
        rows = outputs.shape[0]
        boxes = []
        scores = []
        class_ids = []

        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            if max_score >= self.confidence_thres:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                left = int((x - w / 2) * scaling_factor_x)
                top = int((y - h / 2) * scaling_factor_y)
                width = int(w * scaling_factor_x)
                height = int(h * scaling_factor_y)
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        result_items = []

        for i in indices:
            index = i  # Accessing the index correctly
            box = boxes[index]
            xyxy = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]])
            boxes_obj = BoxResult(xyxy=xyxy[np.newaxis, :], cls=np.array([class_ids[index]]).astype(int), conf=np.array([scores[index]]))
            result_item = Result(boxes=boxes_obj)
            result_items.append(result_item)

        return result_items



    def predict(self, original_image):
        preprocessed_image, scaling_factor_x, scaling_factor_y = self.preprocess(original_image)
        outputs = self.ort_session.run(self.output_names, {self.input_names[0]: preprocessed_image})[0]
        result_items = self.postprocess(outputs, scaling_factor_x, scaling_factor_y)
        return result_items


def test_onnxruntime_match():
    # Example usage
    onnx_yolo = ONNX_YOLOv8()
    original_image = cv2.imread('C:/Users/tomas/Pictures/131930298_728327044754880_2272474766692576674_n.jpg')
    start_time = time.time()
    results_onnx = onnx_yolo.predict(original_image.copy())
    inference_time = round(time.time() - start_time, 3)
    print(f"Number of results: {len(results_onnx)} with coords {results_onnx[0].boxes.xyxy.cpu().numpy().astype(int)}")
    print(f"Inference Time taken: {inference_time} seconds")
    # Extracting bounding boxes, classes, and confidence scores from ONNX YOLO results
    boxes_onnx = [result.boxes.xyxy.cpu().numpy().astype(int) for result in results_onnx]

    # Draw ONNX YOLO bounding boxes
    for box in boxes_onnx:
        for b in box:
            cv2.rectangle(original_image, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)

    ### Ultralytics YOLOv8 ###
    from ultralytics.yolo.engine.model import YOLO as uYOLO

    uyolo_path = 'C:/Projects/NDI_FaceTrack/models/yolov8n_640.onnx'
    uYOLO = uYOLO(uyolo_path, task='detect')
    results_uYOLO = uYOLO.predict(original_image.copy(), device='cpu', conf=0.50)

    # Draw Ultralytics YOLO bounding boxes
    for result in results_uYOLO:
        boxes = result.boxes.xyxy.cpu().numpy().astype(int) 
        for b in boxes:
            cv2.rectangle(original_image, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 2)
            
    # Save the resulting image with both ONNX YOLO and Ultralytics YOLO bounding boxes
    cv2.imwrite('comparison.jpg', original_image)

if __name__ == '__main__':
    test_onnxruntime_match()
