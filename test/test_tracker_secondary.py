from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
import sys, os
import psutil
import gc
from pathlib import Path
import multiprocessing

def get_video_path():
    paths =  list(Path("test/").glob("*.mp4"))
    for path in paths:
        yield path 

def trackers_test(tracker):
    #Get single frame from Video Object
    video = cv2.VideoCapture(str(list(get_video_path())[0]))
    f_tracker = tracker()
    lost_tracking_count = 0
    f_tracker = tracker()
    while video.isOpened():
        ret, frame = video.read()
        resize_frame_shape = (640,360)
        frame = cv2.resize(frame, resize_frame_shape)
        if ret:
            break
        else:
            pass


    (x,y,w,h) = (285, 73, 48, 62) 
    f_tracker.init(frame, (x,y,w,h))
    start = 0
    end = 300

    while start != end:
        ok, position = f_tracker.update(frame)
        if ok:
            pass
        else: 
            lost_tracking_count +=1
        
        start +=1 

    print ('ended')

OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        #"mil": cv2.TrackerMIL_create,
    	"kcf": cv2.TrackerKCF_create,
	}

def test_tracker_secondary():
    for (key, value) in OPENCV_OBJECT_TRACKERS.items():
        print(f'Tracker: {key}')
        trackers_test(value) #No Tracking

if __name__ == '__main__':
    for (key, value) in OPENCV_OBJECT_TRACKERS.items():
        print(f'Tracker: {key}')
        trackers_test(value) #No Tracking
    """
    Without Multithreading or MultiProcessing
    Cum Time: 10.3s 300Frames
    """