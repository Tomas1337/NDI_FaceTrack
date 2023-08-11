import onnxruntime
import cv2
import numpy as np

class ONNX_YOLOv8:
    def __init__(self, model_path='C:/Projects/NDI_FaceTrack/models/yolov8n_640.onnx'):
        opt_session = onnxruntime.SessionOptions()
        opt_session.enable_mem_pattern = False
        opt_session.enable_cpu_mem_arena = False
        opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        EP_list = ['CPUExecutionProvider']
        self.ort_session = onnxruntime.InferenceSession(model_path, providers=EP_list)
        self.input_names = [input.name for input in self.ort_session.get_inputs()]
        self.output_names = [output.name for output in self.ort_session.get_outputs()]

    def detect(self, image_path, conf_threshold=0.8, iou_threshold=0.3):
        # Read and preprocess the image
        image = cv2.imread(image_path)
        input_shape = self.ort_session.get_inputs()[0].shape[2:]
        resized = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), tuple(input_shape))
        input_tensor = (resized / 255.0).transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)

        # Run inference
        outputs = self.ort_session.run(self.output_names, {self.input_names[0]: input_tensor})[0]
        predictions = np.squeeze(outputs).T
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > conf_threshold, :]
        scores = scores[scores > conf_threshold]
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes = predictions[:, :4]
        return boxes, scores, class_ids

# Example usage
yolo = ONNX_YOLOv8()
boxes, scores, class_ids = yolo.detect('C:/Users/tomas/Pictures/peeps.jpg')
print(boxes, scores, class_ids)