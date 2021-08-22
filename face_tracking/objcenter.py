import time, cv2, os
import numpy as np
from facenet_pytorch import MTCNN
<<<<<<< HEAD

#from imutils.object_detection import non_max_suppression
#import ptvsd
CURR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))

=======
from imutils.object_detection import non_max_suppression
import os
import time
CURR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
>>>>>>> development
class FastMTCNN(object):
    """MTCNN implementation."""
    
    def __init__(self):
        self.resize = 0.5
        self.mtcnn = MTCNN(margin = 14, factor = 0.6, keep_all= True,post_process=True, select_largest=False,device= 'cpu')

    def update(self, frame, minConf = 0.9):
        print('Running a MTCNN')
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

    
    def get_all_locations(self, frame, minConf = 0.6):
        if self.resize != 1:
            frame = cv2.resize(frame, (int(frame.shape[1] * self.resize), int(frame.shape[0] * self.resize)))
        boxes, results = self.mtcnn.detect(frame, landmarks=False)
        print('Running a MTCNN 2')
        #print('Face Detection Inference is: {}'.format(time.time()-start_time))
        if boxes is None:
            return []
        elif len(boxes) > 0:
            # Resize facebounding box to original size
            boxes = np.multiply(boxes,(1/self.resize))
            return boxes

#Yolov3 tiny prn 
#Open CV DNN Module implementation
#NO GPU yet; OpenCV doesn't support GPU implementation easily. 
class Yolov3(object):
    def __init__(self):
        np.random.seed(42)
        weightsPath = os.path.join(CURR_PATH, "models/yolov3-tiny-prn.weights")
        configPath = os.path.join(CURR_PATH, "models/yolov3-tiny-prn.cfg")
        labelsPath = os.path.join(CURR_PATH, "models/coco.names")
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

<<<<<<< HEAD
=======

>>>>>>> development
    def update(self, frames, minConf = 0.4, threshold = 0.5):
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

<<<<<<< HEAD
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

class Face_Locker(object):
    """Class that enables facial recognition to allow the Program to track only faces in memory.
    Uses face encodings to differentiate between faces.

    Not in active development as of 09/24/2020
    """
=======


class Yolov4(object):
>>>>>>> development
    def __init__(self):
        np.random.seed(42)
        weightsPath = "models/yolov4-tiny.weights"
        configPath = "models/yolov4-tiny.cfg"
        labelsPath = "models/coco.names"
        self.LABELS = open(labelsPath).read().strip().split("\n")
        # self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        # self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.net = cv2.dnn.readNet(configPath, weightsPath)
        ln = self.net.getLayerNames()
        self.ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3),
        dtype="uint8")
        self.resize = 1
        #self.mot_tracker = Sort()
        self.detections = []


    def update(self, frames, minConf = 0.4, threshold = 0.5):
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

<<<<<<< HEAD
        text_file = open('models/faces/faces_d.txt', 'r')
        self.known_face_names = text_file.read().splitlines()
        print('Loading Face Locker takes: {}s'.format(time.time()-toc))
    
    def face_encode(self, frame, face_coords = None):
        #ptvsd.debug_this_thread()
        scale_factor = 0.25
        """
        TODO: Change paths to be dynamic
        Expects a frame in the face with, with face_coords for ONE face in the format x1,y1,w,h
        Face encodings take the format [[y1, x2, y2, x1]] (List of Tuples)
=======
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
>>>>>>> development

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    
        
                    self.classIDs.append(classID)
                    self.boxes.append([x, y, int(width), int(height)])
                    self.confidences.append(float(confidence))
                    self.detections.append([x,y,x+width,y+height,confidence])


        idxs = cv2.dnn.NMSBoxes(self.boxes, self.confidences, minConf, threshold)
        return idxs, self.boxes, self.COLORS, self.LABELS, self.classIDs, self.confidences

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

# class Face_Locker(object):
#     def __init__(self):
#         #Load encodings
#         toc = time.time()
#         known_face_encodings = np.load('models/faces/faces_d.npy')
#         self.known_face_encodings = [x for x in known_face_encodings]
#         self.original_list_length = len(self.known_face_encodings)
        
#         self.face_locked_on = False

#         text_file = open('models/faces/faces_d.txt', 'r')
#         self.known_face_names = text_file.read().splitlines()
#         print('Loading Face Locker takes: {}s'.format(time.time()-toc))
    
#     def face_encode(self, frame, face_coords = None):
#         #ptvsd.debug_this_thread()
#         scale_factor = 0.25
#         """
#         Expects a frame in the face with, with face_coords for ONE face in the format x1,y1,w,h
#         Face encodings take the format [[y1, x2, y2, x1]] (List of Tuples)

#         If face_coords is not given, entire frame is searched. 
#             Expect face_encoding to be a list with length greater than 1
#         Else if the face_coords is empty return False since there are no faces to encode
#         Else if there are face coords given, return the face embeddings of that face.
#         """
#         toc = time.time()
#         small_frame = cv2.resize(frame, (0,0), fx=scale_factor, fy=scale_factor)
#         rgb_small_frame = small_frame[:,:,::-1]
#         width, height = rgb_small_frame.shape[:2]
        
#         if face_coords is None:
#             face_encoding = face_recognition.face_encodings(rgb_small_frame)
#             print('Scanning entire frame for face')

#         elif len(face_coords) <= 0:
#             print('No face to encode')
#             return False

#         elif not face_coords is None:
#             try: 
#                 x1,y1,w,h = face_coords
#             except ValueError:
#                 x1,y1,w,h = face_coords[0]

#             x1,y1,w,h = [int(b * scale_factor) for b in [x1,y1,w,h]]

#             x2 = x1+w
#             y2 = y1+h
#             face_coords = [y1,x2,y2,x1]
#             face_encoding = face_recognition.face_encodings(rgb_small_frame, [face_coords])
#             print("frame to encode has dimenions {}W and {}H ".format(width, height))
#         print('Face Locker Encoding took: {}s'.format(time.time()-toc))
#         return face_encoding
    
#     def register_new_face(self, frame, face_coords, name = None):
#         scale_factor = 0.25
#         """
#         Expects a frame in the face with, with face_coords for ONE face in the format x1,y1,w,h
#         Face encodings take the format [[y1, x2, y2, x1]] (List of Tuples)

#         If face_coords is not given, entire frame is searched. 
#             Expect face_encoding to be a list with length greater than 1
#         Else if the face_coords is empty return False since there are no faces to encode
#         Else if there are face coords given, return the face embeddings of that face.
#         """
#         toc = time.time()
#         small_frame = cv2.resize(frame, (0,0), fx=scale_factor, fy=scale_factor)
#         rgb_small_frame = small_frame[:,:,::-1]
#         face_coords= [x*scale_factor for x in face_coords]
#         face_encoding = self.face_encode(frame, face_coords)

#         if face_encoding is False:
#             return False
#         elif len(face_encoding) >= 0:
#             self.known_face_encodings.append(face_encoding[0])
#             if name is None:
#                 name = 'Target'
#             self.known_face_names.append(name)
#             return True
#         else:
#             return False

#     def does_it_match(self, face_encoding):
#         #ptvsd.debug_this_thread()
#         """
#         Input face_encoding is a SINGLE face_encoding
#         Return False if no matches are found.
#         Otherwise, return the index of the closest match to the the known_face_encodings
#         """
#         if face_encoding is not False:
#             matches = face_recognition.compare_faces(self.known_face_encodings,face_encoding[0])
#         else:
#             return False

#         if any(matches) == True:
#             face_distances = []
#             for i in self.known_face_encodings:
#                 face_distance = face_recognition.face_distance(i, face_encoding)
#                 face_distances.append(face_distance)

#             best_match_index = np.argmin(face_distances)
#             return best_match_index

#         else:
#             print('No faces match')
#             return False

#     def has_lock(self):
#         if len(self.known_face_encodings) > len(self.original_list_length):
#             self.face_locked_on = True
#         else:
#             self.face_locked_on = False

#     def get_face_locations(self,frame):
#         face_locations = face_recognition.face_locations(frame)
#         return face_locations