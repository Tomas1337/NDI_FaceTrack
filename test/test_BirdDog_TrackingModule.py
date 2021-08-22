import sys, os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from BirdDog_TrackingModule import DetectionWidget
from BirdDog_TrackingModule import tracker_main
import pytest
import cv2
from pathlib import Path


"""
Runs a test by completeing a non-exhaustive execution of the Tracking Module
"""


def get_video_path():
    paths =  list(Path("test/").glob("*.mp4"))
    for path in paths:
        yield path 

testdata = [
    (0, 0, 100),
    (1, 0, 100),
    (2, 0, 100),
    (3, 0, 100),
]

@pytest.mark.parametrize("track_type,show_video,num_of_frames", testdata)
def test_tracking_module(track_type, show_video, num_of_frames):
    video = cv2.VideoCapture(str(list(get_video_path())[0]))
    Tracker = DetectionWidget()
    Tracker.track_type = track_type

    start=0
    target_coordinates = (100,100)
    resize_frame_shape = (640,360)   
    custom_parameters = {}

    while video.isOpened():
        ret, frame = video.read()
        if ret:
            frame = cv2.resize(frame, resize_frame_shape)

            if show_video is True:
                cv2.imshow('frame',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): # wait for 1 millisecond
                    break
            else:
                pass
                
            Tracker.set_tracker_parameters(custom_parameters)
            output = tracker_main(Tracker, frame)
            xVel= output.get('x_Velocity')
            yVel= output.get('y_Velocity')
            
            #TODO: Conduct a more definitive test
            print(xVel, yVel)
            start +=1 
            if start >= num_of_frames:
                break
        else:
            break

        start += 1

if __name__ == '__main__':

    # test_tracking_module(track_type = None, show_video=False, num_of_frames=100) #No Tracking
    # test_tracking_module(track_type = 0, show_video=False, num_of_frames=100) #Face to body
    # test_tracking_module(track_type = 1, show_video=False, num_of_frames=100) #Face Only
    # test_tracking_module(track_type = 2, show_video=False, num_of_frames=100) #Body Only
    pass