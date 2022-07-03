import sys
sys.path.insert(0, '.')
from face_tracking.objcenter import *
from TrackingServer_FastAPI import ReId_Object, Identity
import cv2
from pathlib import Path
import time


def get_video_path(input_folder="test/"):
    paths =  list(Path(input_folder).glob("*.mp4"))
    for path in paths:
        yield path 

IMAGE_HEIGHT = 360
IMAGE_WIDTH = 640

def test_reid_video():
    # To conduct this test properly we need to simulate real-world scnearios
    # 1. Load a video
    # 2. Extract frames from the video and pass to ObjectDetector
    # 3. Extract features from the Object Detector and pass to ReId_Object

    reid = ReId_Object()
    obj = Yolo_v4TINY()
    
    #Load the test_video
    video = cv2.VideoCapture(str(list(get_video_path())[0]))
    resize_frame_shape = (IMAGE_WIDTH,IMAGE_HEIGHT)
    start = 0
    toc = time.time()
    frame_count = 0 
    while video.isOpened():
        
        retval, frame = video.read()

        #Skip number of frames to get to person
        if frame_count <= 120:
            frame_count += 1
            continue

        if retval:
            frame =  cv2.resize(frame, resize_frame_shape)
            toc = time.time()

            # Detect Objects inside
            (idxs, boxes, _, _, classIDs, confidences) = obj.update(frame)
            print("Ran a YOLO")

            if len(boxes) <= 0:
                continue
            elif len(boxes) >= 1:
                x,y,w,h = boxes[np.argmax(confidences)] #Watch out as this will only return 1 object
                x,y,w,h = [0 if i < 0 else int(i) for i in  [x,y,w,h]] #Change all negative numbers to 0
        
            # REID Here
            # Extract features from the detected object
            features = reid.get_features(frame[y:y+h, x:x+w])
            reid.add_identity(features)

            identity_val = reid.is_new_identity(features)
            if identity_val == True:
                print('New Identity')
                reid.add_identity(features)
            tic = time.time()
            print(f"Num of features: {len(features)}")
            print(f"Legnth of features: {len(features[0])}")
            print(f"Time for one `get_features`: {tic-toc}")
        else:
            continue

        frame_count += 1

def test_reid_multi_video():
# Go through the test_reid_*.mp4 videos and
    video_list = list(get_video_path("test/reid_videos/"))
    for video_path in video_list:
        
        reid = ReId_Object()
        obj = Yolo_v4TINY()
        video = cv2.VideoCapture(str(video_path))
        resize_frame_shape = (IMAGE_WIDTH,IMAGE_HEIGHT)
        start = 0
        toc = time.time()
        frame_count = 0

        while video.isOpened():

            # Video Reading
            print(f"Frame Count: {frame_count}")
            retval, frame = video.read()

            if frame_count <= 120:
                frame_count += 1
                continue
            if retval:
                frame =  cv2.resize(frame, resize_frame_shape)
                # Detect Objects inside w/YOLO
                (idxs, boxes, _, _, classIDs, confidences) = obj.update(frame, only_one=False)
                print("Ran a YOLO")

                if len(boxes) <= 0:
                    continue
                elif len(boxes) >= 1:
                    pass
            else: #end of video
                break

            # REID Here
            for x,y,w,h in boxes:
                features = reid.get_features(frame[y:y+h, x:x+w]) # Extract features from the detected object            
                ret_val = reid.is_new_identity(features)  

                if ret_val is None: #Nothing to compare it to
                    reid.add_identity(features)
                    print('New FIRST Identity')

                elif ret_val is True: #Identity is new
                    # Add it to the identity cache
                    key = reid.add_identity(features)
                    print(f'New Identity with key: {key}')
                
                elif isinstance(ret_val, int): #Identity is known
                    print(f'Identity with key: {ret_val} Match was found')
            
                #Draw the bounding box
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                #Write the identity aboe the bounding box
                cv2.putText(frame, f'Person #{ret_val}', (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            cv2.imshow('Frame',frame)

             # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            frame_count += 1

        video_num = video_path.stem.split('_')[-1]
        #assert video_num == len(reid.identities)


def test_reid_pca():
    # Run a PCA on each identity object and plot on a x,y plot.
    # This is to see if the PCA is working as expected.

    reid = ReId_Object(pca=True)
    obj = Yolo_v4TINY() 
    video = cv2.VideoCapture(str(list(get_video_path())[0]))
    resize_frame_shape = (IMAGE_WIDTH,IMAGE_HEIGHT)
    start = 0
    toc = time.time()
    frame_count = 0

    while video.isOpened():
        retval, frame = video.read()

        if frame_count <= 120:
            frame_count += 1
            continue
        if retval:
            frame =  cv2.resize(frame, resize_frame_shape)
            toc = time.time()

            # Detect Objects inside
            (idxs, boxes, _, _, classIDs, confidences) = obj.update(frame)
            print("Ran a YOLO")

            if len(boxes) <= 0:
                continue
            elif len(boxes) >= 1:
                x,y,w,h = boxes[np.argmax(confidences)]

        # REID Here
        # Extract features from the detected object
        features = reid.get_features(frame[y:y+h, x:x+w])
        reid.add_identity(features)

        for x,y,w,h in boxes:
            features = reid.get_features(frame[y:y+h, x:x+w]) # Extract features from the detected object            
            ret_val = reid.is_new_identity(features)  

            if ret_val is None: #Nothing to compare it to
                reid.add_identity(features)
                print('New FIRST Identity')

            elif ret_val is True: #Identity is new
                # Add it to the identity cache
                key = reid.add_identity(features)
                print(f'New Identity with key: {key}')
            
            elif isinstance(ret_val, int): #Identity is known
                print(f'Identity with key: {ret_val} Match was found')

            #Apply reranking
            ret_val = reid.rerank()

            #Draw the bounding box
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            #Write the identity aboe the bounding box
            cv2.putText(frame, f'Person #{ret_val}', (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        cv2.imshow('Frame',frame)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        frame_count += 1

#test_reid_video()
#test_reid_multi_video()
test_reid_pca()