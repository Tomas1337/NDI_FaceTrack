import os
import cv2
import json
import torch
from mtcnn import MTCNN
from PIL import Image
from typing import Tuple
from ultralytics import YOLO

def create_annotations(directory: str, output_file: str="annotations.json", chunk_size=500):
    # Initialize the list to store annotations
    mtcnn = MTCNN()
    yolov8 = YOLO('models/yolov8n.pt')
    annotations = []
    images = []
    image_id = 0
    ann_id = 0
    file_id = 0  # to number the chunked files
    
    
    # Process all images in the directory
    for f_idx, filename in enumerate(os.listdir(directory)):
        
        # if f_idx >= 10:
        #     break
        
        if filename.endswith(".jpg"):
            # Load image
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path)
            img_cv = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

            # Run face detection
            faces = mtcnn.detect_faces(img_cv)

            # Run YOLOv8 model for person detection
            results = yolov8(img_cv)[0]
            
            # Add image information
            images.append({
                "id": image_id,
                "file_name": filename,
                "height": img_cv.shape[0],
                "width": img_cv.shape[1],
            })
            
            person_detections = []
            for idx, cls in enumerate(results.boxes.cls.cpu().numpy().astype(int)):
                if cls == 0:
                    person_detections.append(results.boxes[idx].xyxy.cpu().numpy()) 

            # Prepare COCO annotations for persons
            for dets in person_detections:  # If persons detected
                for det in dets:
                    x, y, w, h = det[0], det[1], det[2] - det[0], det[3] - det[1]
                    x,y,w,h = int(x), int(y), int(w), int(h)
                    annotation = {"id": ann_id, "image_id": image_id, "category_id": 1, "bbox": [x, y, w, h], "area": w*h, "iscrowd": 0}
                    annotations.append(annotation)
                    ann_id += 1

            # Prepare COCO annotations for faces
            if faces:  # If faces detected
                for face in faces:
                    x, y, w, h = face['box']
                    x,y,w,h = int(x), int(y), int(w), int(h)
                    annotation = {"id": ann_id, "image_id": image_id, "category_id": 0, "bbox": [x, y, w, h], "area": w*h, "iscrowd": 0}
                    annotations.append(annotation)
                    ann_id += 1
                    
            image_id += 1

        # Save the data to a JSON file every chunk_size images
        if f_idx % chunk_size == 0 and f_idx != 0:
            data = {
                "categories": [{"id": 0, "name": "face", "supercategory": "none"}, {"id": 1, "name": "person", "supercategory": "none"}],
                "images": images,
                "annotations": annotations
            }
            with open(f"{output_file.split('.')[0]}_{file_id}.json", "w") as f:
                json.dump(data, f, indent=4)
            file_id += 1

            # Reset the lists for the next chunk
            annotations = []
            images = []

def train_yolo(yaml_path="C:/Projects/NDI_FaceTrack/datasets/FaceBodyTracking/data.yaml"):
    pretrained_model = 'models/yolov8n.pt'
    model = YOLO(pretrained_model)  # load a pretrained model (recommended for training)
    model.train(data=yaml_path, epochs=200, imgsz=480, device=0, batch=4, workers=2)  # train the model
    
    
def convert_to_onnx(yolo_model_path):
    pass
            
if __name__ == "__main__":
    # Call the function
    # create_annotations('D:/Downloads/person 2.v2i.coco/train/')
    
    # Train YOLOv8
    train_yolo()