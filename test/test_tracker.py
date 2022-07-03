# from imutils.video import VideoStream
# from imutils.video import FPS
# import argparse
# import imutils
# import time
# import cv2
# import sys, os
# import psutil
# import gc
# PACKAGE_PARENT = '..'
# SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
# sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
# import pytest 

# def get_video_path():
#     test_path = "test/"
#     for file in os.listdir(test_path):
#         if file.endswith(".mp4"):
#             video_path = os.path.join(test_path, file)
#             yield video_path

# def tracker_test(tracker):
#     video = cv2.VideoCapture(list(get_video_path())[0])
#     end = 300
#     start=0
#     target_coordinates = (100,100)   
#     f_tracker = None
#     (x,y,w,h) = (285, 73, 48, 62) #Test Track coordinates for short_video
#     last_position = None
#     lost_tracking_count = 0
#     stop_tracker = False
#     f_tracker = tracker()
#     track_init = False
    
#     while video.isOpened():
#         if start <= 100:
#             ret, frame = video.read()
#         else:
#             ret = ok
        
#         resize_frame_shape = (640,360)
#         frame = cv2.resize(frame, resize_frame_shape)

#         cv2.imshow('frame',frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'): # wait for 1 millisecond
#             break

#         if ret:
            
#             #Simulate tracking interactions
#             if start% 100 == 0:
#                 #Reset Trigger
#                 f_tracker =  None

#             elif (start == 150) and (start >= 1) and (stop_tracker == False):
#                 del f_tracker
#                 f_tracker = tracker()
#                 f_tracker.init(frame, last_position)
#                 print('Tracker re-inilization')
            
#             elif (start == 200) and (start >= 1) and (stop_tracker is False):
#                 #Stop the tracker
#                 stop_tracker = True
#                 f_tracker = None
#                 probe.add_marker('Tracker: stopped tracking}')
#                 print('stopped tracking')

#             elif (start == 250) and (start >= 1) and (stop_tracker is True):
#                 #Resume tracker
#                 stop_tracker = False
#                 track_init = False
#                 f_tracker = None
#                 print('Tracker: Resumed Tracker}')


#             #Actual tracker updates
#             if track_init is False:
#                 f_tracker = tracker()
#                 f_tracker.init(frame, (x,y,w,h))
#                 track_init = True

#             elif not f_tracker is None and stop_tracker == False:
#                 ok, position = f_tracker.update(frame)
#                 if position == (0,0,0,0):
#                     lost_tracking_count += 1
#                 elif ok:
#                     last_position = position
#                     print(f'Position: {position}')
#                 else:
#                     lost_tracking_count += 1
#                     print('Tracking failed')

#             else: 
#                 #continue video
#                 pass

#             start +=1 

#         if start == end:
#             print('Ended')
#             print(f"Tracking lost count:{lost_tracking_count}")
#             break
            
# OPENCV_OBJECT_TRACKERS = {
#         "csrt": cv2.TrackerCSRT_create,
#         #"mil": cv2.TrackerMIL_create,
#     	"kcf": cv2.TrackerKCF_create,
# 	}

# def test_tracker():
#     for (key, value) in OPENCV_OBJECT_TRACKERS.items():
#         print(f'Tracker: {key}')
#         tracker_test(value) #No Tracking


# if __name__ == '__main__':
#     for (key, value) in OPENCV_OBJECT_TRACKERS.items():
#         print(f'Tracker: {key}')
#         test_trackers(value) #No Tracking
