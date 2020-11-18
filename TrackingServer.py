from flask import Flask
from flask import request
from flask import jsonify
import time, math, io, win32file, win32pipe, cv2
import numpy as np
import pywintypes, cv2, struct
from PIL import Image
from threading import Thread
from BirdDog_TrackingModule import DetectionWidget
from BirdDog_TrackingModule import tracker_main



app = Flask(__name__)

BUFFERSIZE = 921654
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360
#pipeHandle = None

@app.route('/api/start_pipe_server', methods = ["GET", "POST"])
def start_pipe_server():
    """[summary]
    Initiates a Pipe Server
    We expect a Pipe Client from an extenral application to connect to the Pipe Server initiated by this method
    """
    #Starts pipe server
    pipeName, pipeHandle = init_pipe_server()
    init_response = checkPipe(pipeHandle)

    if init_response is True:
        tracking_thread = Thread(target = start_tracking, args=[pipeHandle])
        tracking_thread.start()
        response = jsonify(pipeName)

    else:
        print("Could not connect to pipe")
        response = jsonify(success=False)

    return response
    

def start_tracking(pipeHandle):
    Tracker = DetectionWidget()
    track_flag = True

    try:
        print("Waiting for client")
        win32pipe.ConnectNamedPipe(pipeHandle, None) #This operation is blocking
        print("Got client")

        res = win32pipe.SetNamedPipeHandleState(pipeHandle, win32pipe.PIPE_READMODE_MESSAGE, None, None)
        if res == 0:
            print(f"SetNamedPipeHandleState return code: {res}")
        
        while track_flag is True:   
            try:
                #Read Pipe in bytes form
                data = win32file.ReadFile(pipeHandle, 1000000)
                
                #Parse the bytes and unpack to variables
                data = data[1].split(b'frame')
                track_settings = struct.unpack('ffffff???i', data[0])
                
                frame = np.frombuffer(data[1], dtype = np.uint8).reshape(IMAGE_HEIGHT,IMAGE_WIDTH, 3)
                custom_parameters = {
                    'gamma' : track_settings[2],
                    'xminE' : track_settings[3],
                    'yminE' : track_settings[4],
                    'ZoomValue' : track_settings[5],
                    'y_trackState': track_settings[6],
                    'autozoom_state' : track_settings[7],
                    'reset_trigger' : track_settings[8],
                    'track_type': track_settings[9]
                    }

                target_coordinates = (track_settings[0], track_settings[1])

                #Pass to Track Module and get output
                output = tracker_main(Tracker, frame, target_coordinates, custom_parameters)
         
                    
                #Unpack and Pack
                xVel = output['xVelocity']
                yVel = output['yVelocity']
                box = output['boundingBox']
                if len(box[0]) == 0:
                    box = b''
                else:

                    try:
                        x,y,w,h = box[0]
                    except ValueError:
                        print('Cannot unbox box')
                    box = struct.pack('iiii', x,y,w,h)
                payload = struct.pack('ff', xVel ,yVel) + b'split'+ box
                
                #Write Pipe
                win32file.WriteFile(pipeHandle, payload)


            except pywintypes.error as e:
                print("Could not read from pipe")
                track_flag = False

            except ValueError as v:
                print(v)
                
    except pywintypes.error as e:
        if e.args[0] == 2:
            print("no pipe, trying again in a sec")
            time.sleep(1)
        elif e.args[0] == 109:
            print("broken pipe, bye bye")
        print("finished now")

    finally:
        win32pipe.DisconnectNamedPipe(pipeHandle)
        win32file.CloseHandle(pipeHandle)
        print("Pipe has been closed from server")


def decodeByteImage(data):
    """
    This function converts a byteArray(BMP format) into a Numpy Array.

    Args:
        data (tuple): data[1] should contain the byteArray

    Returns:
        frame: numpy array representing the image
    """
    try:
        frame = np.array(Image.frombytes("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), data[1]))
        frame = np.flipud(frame)
        # frame = np.rot90(frame, 2)
        cv2.imshow("Cam Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return 0
    except ValueError:
        print("Error Decoding Frame. Check Size of received frame")
        frame = None

    return frame

def init_pipe_server(pipeName = 'BD_Tracking'):
    try: 
        pipeNum += 1
    except NameError:
        pipeNum = 1

    while True:
        try:
            pipeName = 'BD_Tracking' + str(pipeNum)
            pipeHandle = win32pipe.CreateNamedPipe(
                r'\\.\pipe\{}'.format(pipeName),
                win32pipe.PIPE_ACCESS_DUPLEX,
                win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
                1, 921600, 921600,
                0,
                None)
            break

        except pywintypes.error as e:
            if e.winerror == 231:
                pipeNum +=1
            else:
                print(f'Could not Start pipe due to error: {e.strerror}')
                return None

    print(f"Starting a Python Server Pipe with name {pipeName} with pipeHandle of {pipeHandle}")

    pipeState = checkPipe(pipeHandle)
    print(f'Checked the pipeHandle which returned {pipeState}')
    return (pipeName, pipeHandle)

def checkPipe(pipeHandle):
    """Checks if pipe at fileHandle is open
    Args:
        fileHandle : Handler for the Pipe

    Returns:
        response
            0: success
            1: Error'd out
    """
    try:
        pipeFlag, _, _, _ = win32pipe.GetNamedPipeInfo(pipeHandle)
        if pipeFlag > 0:
            pipeState = True
        else: 
            pipeState = False

    except pywintypes.error as e:
        pipeState = False

    return pipeState


if __name__ == "__main___":
    app.run(debug = True)

        