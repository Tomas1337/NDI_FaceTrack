from ultralytics import YOLO

class YoloHandler:
    def __init__(self, model_path, device, width=640, height=480):
        self.model_path = model_path
        self.device = device
        self.width = width
        self.height = height
        self.model = self.load_model()

    def load_model(self):
        self.yolo_model = YOLO(self.model_path, device=self.device)
        return self.yolo_model
        
    def convert_to_onnx(self):
        onnx_model = self.yolo_model.export(format='onnx', imgsz=[self.height, self.width], optimize_cpu=True)
        return onnx_model
    
    def predict(self, image):
        image = image.to(self.device)
        results = self.yolo_model(image)
        return results

    def get_model(self):
        return self.model