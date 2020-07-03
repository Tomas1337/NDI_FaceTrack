import imutils
import cv2
import time
import numpy as np
from facenet_pytorch import MTCNN
from imutils.object_detection import non_max_suppression
import ptvsd
#from tool.darknet2pytorch import Darknet

class FastMTCNN(object):
    """Fast MTCNN implementation."""
    
    def __init__(self):
        self.resize = 1
        self.mtcnn = MTCNN(margin = 14, factor = 0.6, keep_all= True,post_process=True, select_largest=False,device= 'cpu')

    def update(self, frame, body_coords, minConf = 0.75):
        ptvsd.debug_this_thread()
        if len(body_coords) > 0 or body_coords is None:
            x_b, y_b, w_b, h_b = body_coords
            x_b, y_b, w_b, h_b = [0 if i < 0 else i for i in [x_b, y_b, w_b, h_b]]
        else:
            return []

        W, H = frame.shape[:2]
        if self.resize != 1:
            frame = cv2.resize(frame, (int(frame.shape[1] * self.resize), int(frame.shape[0] * self.resize)))
        
        frame = frame[y_b:y_b+h_b, x_b:x_b+w_b, :]
        boxes, results = self.mtcnn.detect(frame, landmarks=False)

        #print('Face Detection Inference is: {}'.format(time.time()-start_time))

        if boxes is None:
            return []
        
        #Filter out the results if there are faces less than minConf
        try:
            boxes = [i for i in boxes if results.any() > minConf]
        except ValueError:
            boxes = [i for i in boxes if results > minConf]

        if len(boxes) > 0:
            #Resize facebounding box to original size
            boxes = np.multiply(boxes,(1/self.resize))
            (x,y,x2,y2) =  boxes[0]
        else:
            return []

        w = x2-x
        h = y2-y
        x,y,w,h = [int(x) if x > 0 else 0 for x in [x,y,w,h]]

        #Translate into framfe coordinates, right now they are in reference to the body frame only
        x = x_b + x
        y = y_b + y
        
        boxes = [x,y,w,h]
        return boxes


class Hog_Detector(object):
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def update(self, frame, frameCenter, minConf = 0.4, threshold = 0.5):
        frame = imutils.resize(frame, width=min(400, frame.shape[1]))
        orig = frame.copy()

        rects, weights = self.hog.detectMultiScale(frame, winStride =(8,8), padding=(8,8), scale = 1.05)
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        boxes = non_max_suppression(rects, probs=None, overlapThresh=threshold)

        if len(boxes) > 0 and len(weights) > 0:
            boxes = boxes.tolist()
            weights = weights.tolist()
        else: 
            boxes = []
            weights = []
        return boxes, weights


#Yolov3 tiny prn 
#Open CV DNN Module implementation
#NO GPU yet; OpenCV doesn't support GPU implementation easily. 
class Yolov3(object):

    def __init__(self):
        np.random.seed(42)
        weightsPath = "C:/Projects/NDI_FaceTrack/models/yolov3-tiny-prn.weights"
        configPath = "C:/Projects/NDI_FaceTrack/models/yolov3-tiny-prn.cfg"
        labelsPath = "C:/Projects/NDI_FaceTrack/models/coco.names"
        self.LABELS = open(labelsPath).read().strip().split("\n")
        self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        # self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        ln = self.net.getLayerNames()
        self.ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3),
        dtype="uint8")
        self.resize = 1
        #self.mot_tracker = Sort()
        self.detections = []


    def update(self, frames, frameCenter, minConf = 0.4, threshold = 0.5):
        idxs = []
        self.boxes = []
        self.confidences = []
        self.classIDs = []

        (H, W) = frames.shape[:2]
        if self.resize != 1:
            frames = cv2.resize(frames, (int(frames.shape[1] * self.resize), int(frames.shape[0] * self.resize)))
        
        blob = cv2.dnn.blobFromImage(frames, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        start = time.time()
        layerOutputs = self.net.forward(self.ln)
        #print("Detection yolov3 time takes  {}".format(time.time()-start))
        # loop over each of the layer outputs

        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                #Filter out only person detections
                if not classID == 0:
                    continue 
                confidence = scores[classID]
                if confidence > minConf:
                    #print('Confidence: {}'.format(confidence))
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    
        
                    self.classIDs.append(classID)
                    self.boxes.append([x, y, int(width), int(height)])
                    self.confidences.append(float(confidence))
                    self.detections.append([x,y,x+width,y+height,confidence])


        idxs = cv2.dnn.NMSBoxes(self.boxes, self.confidences, minConf, threshold)
        return idxs, self.boxes, self.COLORS, self.LABELS, self.classIDs, self.confidences

#NOT WORKING
class Yolov3_2(object):
    #PyTorch implementation for GPU use
    def __init__(self):

        self.use_cuda = True
        np.random.seed(42)
        weightsPath = "C:/Projects/NDI_FaceTrack/models/yolov3-tiny.weights"
        configPath = "C:/Projects/NDI_FaceTrack/models/yolov3-tiny.cfg"
        labelsPath = "C:/Projects/NDI_FaceTrack/models/coco.names"
        self.LABELS = open(labelsPath).read().strip().split("\n")
        self.m = Darknet(configPath)

        self.class_names = load_class_names(labelsPath)
        self.m = Darknet(configPath, img_size=(416, 416))
        self.m.eval()
        load_darknet_weights(self.m, weightsPath)

        if self.use_cuda:
            self.m.cuda()

    def update(self, frames, frameCenter, minConf = 0.4, threshold = 0.3):
        idxs = []
        self.boxes = []
        self.confidences = []
        self.classIDs = []

        (H, W) = frames.shape[:2]
        sized = cv2.resize(frames, (416, 416))
        start = time.time()
    
        #with torch.no_grad():
        boxes = do_detect(self.m, sized, 0.3, 80, 0.4, self.use_cuda)

        print("Detection time takes {}".format(time.time()-start))
        result_img = plot_boxes_cv2(frames, boxes, savename=None, class_names= self.class_names)
        cv2.imshow('Yolo demo', result_img)
        cv2.waitKey(1)

#NOT WORKING
class Yolov4(object):
    #PyTorch implentations
    def __init__(self):
        self.use_cuda = True
     
        np.random.seed(42)
        weightsPath = "C:/Projects/NDI_FaceTrack/models/yolov3-tiny-prn.weights"
        configPath = "C:/Projects/NDI_FaceTrack/models/yolov3-tiny-prn.cfg"
        # weightsPath = "C:/Projects/NDI_FaceTrack/models/yolov4.weights"
        # configPath = "C:/Projects/NDI_FaceTrack/models/yolov4.cfg"
        labelsPath = "C:/Projects/NDI_FaceTrack/models/coco.names"
    
        self.class_names = load_class_names(labelsPath)
        self.m = Darknet(configPath)
        self.m.eval()
        #self.m.print_network()
        self.m.load_weights(weightsPath)

        if self.use_cuda:
            self.m.cuda()
                
        # self.LABELS = open(labelsPath).read().strip().split("\n")
        # self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3),
        # dtype="uint8")
        self.resize = 1


    def update(self, frames, frameCenter, minConf = 0.4, threshold = 0.3):
        idxs = []
        self.boxes = []
        self.confidences = []
        self.classIDs = []

        (H, W) = frames.shape[:2]
        if self.resize != 1:
            frames = cv2.resize(frames, (int(frames.shape[1] * self.resize), int(frames.shape[0] * self.resize)))
        
        sized = cv2.resize(frames, (self.m.width, self.m.height))
        start = time.time()
    
        with torch.no_grad():
            boxes = do_detect(self.m, sized, 0.3, 80, 0.4, self.use_cuda)

        print("Detection time takes {}".format(time.time()-start))
        result_img = plot_boxes_cv2(frames, boxes, savename=None, class_names= self.class_names)
        cv2.imshow('Yolo demo', result_img)
        cv2.waitKey(1)

        #tracker, detection_class = self.deepsort.run_deep_sort(frames, self.confidences, self.boxes)

        # for track in tracker.tracks:
        #     if not track.is_confirmed() or track.time_since_update > 1:
        #         continue

        #     bbox = track.to_tlbr()
        #     id_num = str(track.track_id)
        #     features = track.features

        #     cv2.rectangle(frames, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 1)
        #     cv2.putText(frames, str(id_num),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),1)
        # [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]


        # tracked_objects = self.mot_tracker.update(detections.cpu())

