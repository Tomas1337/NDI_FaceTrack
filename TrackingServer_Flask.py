from flask import Flask, request, jsonify, render_template, Response, session
from flask_script import Manager, Server
import time, math, io, win32file, win32pipe, cv2, requests, configparser
from tkinter import messagebox as mb
import numpy as np
import pywintypes, cv2, struct
#from licensing.models import *
#from licensing.methods import Key, Helpers, PaymentForm, AddCustomer
from PIL import Image
import NDIlib as ndi
from config import CONFIG
from threading import Thread
from BirdDog_TrackingModule import DetectionWidget
from BirdDog_TrackingModule import tracker_main
from ndi_camera import ndi_camera
import concurrent.futures
from multiprocessing import Process


BUFFERSIZE = 921654
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360
KEY = CONFIG.get('license','key')

app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = 'reds209ndsldssdsljdsldsdsljdsldksdksdsdfsfsfsfis'

@app.route("/")
def index():
	# return the rendered template
    
	return render_template("index.html")

#http://127.0.0.1:5000//video_feed?camera_name=DESKTOP-C16VMFB (VLC) 
@app.route('/video_feed')
def video_feed():
    camera_name = request.args.get('camera_name')
    return render_template("video_feed.html", camera_name=camera_name)


@app.route('/video_camera')
def video_camera():
    camera_name = request.args['camera_name']
    video_object = Video_Object()
    #tracker_object = Tracker_Object(video_object)
    # video_thread = Thread(target = start_tracking_html, args=[video_object])
    # video_thread.start()    
    #return_value = future.result()
    #print(return_value)
    print('hello')

    # tracking_process = Process(target = tracker_object.tracking_loop)
    # tracking_process.start()

    ndi_cam = ndi_camera()
    ndi_recv = ndi_cam.camera_connect(ndi_name=camera_name)

    return Response(video_object.frame_gen(ndi_recv),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

class Tracker_Object():
    def __init__(self):
        self.tracker = DetectionWidget()

    def track_frame(self, frame):  
        target_coordinates = (50,50)
        custom_parameters = {'gamma':0.0}
        if frame is not None:
            output = tracker_main(self.tracker, frame, target_coordinates, custom_parameters)
            #print(f'Output is {output}')
            bb_box = output.get('boundingBox')
            self.set_bounding_boxes(bb_box)

            return output
        else:
            print('frame is none')
            return None

    def get_bounding_boxes(self):
        return self.bounding_boxes

    def set_bounding_boxes(self, bounding_box):
        self.bounding_boxes = bounding_box

class Video_Object():
    def __init__(self):
        self.np_frame = None
        self.bounding_boxes = []
        self.tracker_object = Tracker_Object()
        
    def frame_gen(self, ndi_recv, draw = True, track = True):
        while True:
            t,v,_,_ = ndi.recv_capture_v2(ndi_recv, 0)
            if t == ndi.FRAME_TYPE_VIDEO:
                frame = v.data
                frame = frame[:,:,:3]
                frame = np.array(frame)
                self.np_frame = frame

                if track:
                    self.track()
                    if draw and len(self.bounding_boxes[0]) >= 1:
                        for (x,y,x2,y2) in self.bounding_boxes:
                            cv2.rectangle(frame, (x,y), (x2,y2), (255,0,0), 1)

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    def get_frame(self):
        return self.np_frame

    def track(self):
        output = self.tracker_object.track_frame(self.np_frame)
        if output is not None:
            print(output.get('xVelocity'), output.get('yVelocity'))
            self.bounding_boxes = self.tracker_object.get_bounding_boxes()
            

@app.route('/api/start_pipe_server', methods = ["GET", "POST"])
def start_pipe_server():
    """
    Initiates a Pipe Server
    We expect a Pipe Client from an external application to connect to the Pipe Server initiated by this method
    """

    # #Checks your current license
    # if check_license():
    #     pass
    # else:
    #     return jsonify(success=False), 403

    #Starts pipe server
    pipeName, pipeHandle = init_pipe_server()
    init_response = checkPipe(pipeHandle)

    if init_response is True:
        tracking_thread = Thread(target = start_tracking_pipe, args=[pipeHandle])
        tracking_thread.start()
        response = jsonify(pipeName)

    else:
        print("Could not connect to pipe")
        response = jsonify(success=False)

    return response
        
def start_tracking_pipe(pipeHandle):
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

        

# @app.route('/api/cf_createTrialLicense', methods = ["GET", "POST"])
# def cf_createTrialLicense():
#     """Google Cloud function to create a Trial License Key
#         This function contains the confidential ACCESS TOKENS that are
#         not to be shared to customers.
#     Returns: Tuple 
#         int: 0 or 1 signifying success(0) or failure(1)
#         string: Error message if error or trial_key if success
#     """
#     RSAPubKey = "<RSAKeyValue><Modulus>stP6+ZYvVTNNPnReyTr8oiTxTq9NgUHK09wIns+jZCmncfJn4QOxc0X7LfkaqZcRdhvjGtXFBvwn6Tn8z159dOETHGUzPMyc8r6RKWG0i2q0ChYUFbyiZZcvHpQR4bXLf25Mb0+DSMIcDqWNCbjytSG8dPpvhomGUFTeHp9xbePdhZyH0wiw5LF+CIxBghBBgpRGCxJ7JqffIjs7E/vqTc4I5jMUc4IeEtJ6KzPZYRjKz+sHbZFxNd82qe1WlYXAjz4PF6kZo/SXO9QVJ5XKoULvkxsxfSdwS8fQHOLCsrLkpI17Einp8ZySqiYJu/0kpZENWv9Rao0771+wVzAaBw==</Modulus><Exponent>AQAB</Exponent></RSAKeyValue>"
#     TRIAL_AUTH = "WyIyMzE0ODIiLCJrenpSZ291Umk5TG0rRnVhWk1RbXI4RU45T1grUDNrSzhoaTM3VlVRIl0="
#     AUTH = "WyIyMzE0NTQiLCIwNGVYNWZad2wybldBVmNkTFo1dkdJYTM3VTFOREdOelpiZjNucERNIl0="
#     machine_code = request.form.get('machine_code')

#     created_key = Key.create_trial_key(token=TRIAL_AUTH,\
#     product_id=8730, \
#     machine_code = machine_code)
#     if created_key[0] is not None:
#         print("Created a new trial license. Activating as Trial")
#         trial_key = created_key[0]
#         result = Key.activate(token=AUTH,\
#             rsa_pub_key=RSAPubKey,\
#             product_id=8730, \
#             key=trial_key,\
#             machine_code=machine_code,
#             metadata=True)

#         if result[0] is not None:
#             return jsonify(0, trial_key)
#         else:
#             return jsonify(1,result[1])
#     else:
#         return jsonify(1,created_key[1])


# @app.route('/api/cf_check_license', methods = ["GET", "POST"])
# def cf_check_license():
#     """A google cloud function that verifies the license
#     Input comes from request object

#     Returns:
#         JSON object: success: True/False
#     """
#     RSAPubKey = "<RSAKeyValue><Modulus>stP6+ZYvVTNNPnReyTr8oiTxTq9NgUHK09wIns+jZCmncfJn4QOxc0X7LfkaqZcRdhvjGtXFBvwn6Tn8z159dOETHGUzPMyc8r6RKWG0i2q0ChYUFbyiZZcvHpQR4bXLf25Mb0+DSMIcDqWNCbjytSG8dPpvhomGUFTeHp9xbePdhZyH0wiw5LF+CIxBghBBgpRGCxJ7JqffIjs7E/vqTc4I5jMUc4IeEtJ6KzPZYRjKz+sHbZFxNd82qe1WlYXAjz4PF6kZo/SXO9QVJ5XKoULvkxsxfSdwS8fQHOLCsrLkpI17Einp8ZySqiYJu/0kpZENWv9Rao0771+wVzAaBw==</Modulus><Exponent>AQAB</Exponent></RSAKeyValue>"
#     AUTH = "WyIyMzE0NTQiLCIwNGVYNWZad2wybldBVmNkTFo1dkdJYTM3VTFOREdOelpiZjNucERNIl0="
#     TRIAL_AUTH = "WyIyMzE0ODIiLCJrenpSZ291Umk5TG0rRnVhWk1RbXI4RU45T1grUDNrSzhoaTM3VlVRIl0="
#     key = request.form.get('key')
#     machine_code = request.form.get('machine_code')

#     result = Key.activate(token=AUTH,\
#                     rsa_pub_key=RSAPubKey,\
#                     product_id=8730, \
#                     key=key,\
#                     machine_code=machine_code,
#                     fields_to_return=all,
#                     metadata=True)

#     if result[0] == None or not Helpers.IsOnRightMachine(result[0]):
#         if result[1] == 'Could not find the key.':
#             print('Could not find the key.')

#         elif result[1] == 'The key is blocked and cannot be accessed.':
#             print('Your trial has expired or you may have an expired Key')

#         else:
#             print(f'Error with License:{result[1]}')
            
#         return jsonify(success=False)
    
#     else:
#         # everything went fine if we are here!
#         print("The license is valid!")
#         license_key = result[0]
#         print("Feature 1: " + str(license_key.f1))
#         print("License expires: " + str(license_key.expires))
#         return jsonify(success=True)


# @app.route('/api/check_license', methods = ["GET", "POST"])
# def check_license(): 
#     res = check_license_2()
#     print(f'result:{res}')

#     if res is True:
#         pass
#     else:
#         buy_license()
#     return jsonify(success=True)


# def buy_license(trial_finished=True):
#     buy_license_url = 'http://127.0.0.1:5000/api/cf_buy_license' #Cloud function URL  
#     create_customer_url =
#     #Start Dialog Box
#     if trial_finished:
#         message = "Your Trial period has ended. Would you like to purchase a perpetual license?"
#     else:
#         message = "Would you like to purchase a perpetual license?"
    
#     answer = mb.askquestion('License',message)
#     if answer == 'yes':
#         #Get User Information


#         user_infromation_payload = {
#             'token':
#             'Name':
#             'E'
#         }


#         response = requests.post(buy_license_url)
#         if response.json()['success'] is True:
#             #Open Dialog to enter License Key
        

#     else:
#         root.destroy()
#         return False


# @app.route('/api/cf_buy_license', methods = ["GET", "POST"])
# def cf_buy_license():
#     #Cloud function called to allow customer to purchase a license

#     create_key_token = 'WyIyMzczNDQiLCJsZThVdTBvUHRIUnFSamQwVEVONVZCM3ZTMHRxOElzTi9yaitqUlYrIl0='
#     create_key_url = f'https://app.cryptolens.io/api/key/CreateKey?token={create_key_token}&ProductId=8730&Period=1&F1=True&F2=False&F3=False&F4=False&F5=False&F6=False&F7=False&F8=False&Block=False&CustomerId=0&TrialActivation=True&MaxNoOfMachines=1&NoOfKeys=0&NewCustomer=False&AddOrUseExistingCustomer=False&ResellerId=0&EnableCustomerAssociation=False&AllowActivationManagement=False'
#     paymentform_token ='WyIyNDA0NzQiLCIzeVJWVUVlOHdqWjJwb2U1b2h6eWJXaDA3a2JnaUFHdVUvdjZFNjJUIl0='    
#     create_session_url = 'https://app.cryptolens.io/api/paymentform/CreateSession'
#     create_user_url = f'https://app.cryptolens.io/api/customer/AddCustomer?Name={customer_name}&token={paymentform_token}'
    
    
#     #Create Customer
#     customer_result = AddCustomer.create_customer(token = paymentform_token, 
#         name='Tumus', email='tomads@gmail.com', company_name='Fesas' )

#     if customer_result['result'] == 0:
#         #Sucessfully added a customer
#         customer_id = customer_result['cusomterId']
    
#     else:
#         print(f'Session creation failed due to:{session_result['message']}') 

    
#     session_result = PaymentForm.create_session(paymentform_token, 750, 
#         currency = 'USD', expires=300, price =150, 
#         heading='NDI Face Track', product_name='NDI Face Track')[0]
    
#     if session_result['result'] == 0:
#         #Successful session
#         session_id = session_result['sessionId']

#         #Start payment form
#         payment_url =  f'https://app.cryptolens.io/form/P/giZq28aW/750?sessionId={session_id}'
#         import webbrowser
#         webbroweser.open(payment_url)
#     else:
#         print(f'Session creation failed due to:{session_result['message']}') 


# def check_license_2():
#     RSAPubKey = "<RSAKeyValue><Modulus>stP6+ZYvVTNNPnReyTr8oiTxTq9NgUHK09wIns+jZCmncfJn4QOxc0X7LfkaqZcRdhvjGtXFBvwn6Tn8z159dOETHGUzPMyc8r6RKWG0i2q0ChYUFbyiZZcvHpQR4bXLf25Mb0+DSMIcDqWNCbjytSG8dPpvhomGUFTeHp9xbePdhZyH0wiw5LF+CIxBghBBgpRGCxJ7JqffIjs7E/vqTc4I5jMUc4IeEtJ6KzPZYRjKz+sHbZFxNd82qe1WlYXAjz4PF6kZo/SXO9QVJ5XKoULvkxsxfSdwS8fQHOLCsrLkpI17Einp8ZySqiYJu/0kpZENWv9Rao0771+wVzAaBw==</Modulus><Exponent>AQAB</Exponent></RSAKeyValue>"
#     verify_url = 'http://127.0.0.1:5000/api/cf_check_license' #Cloud function URL
#     trial_url = 'http://127.0.0.1:5000/api/cf_createTrialLicense' #Cloud function URL  

#     # Read license file from file
#     with open('licensefile.txt', 'r+') as f:
#         license_key = f.read()

#         #Simple Length Filter
#         if len(license_key) != 23:
#             print("NOTE: This license file cannot be verified.")
#             print("Requesting a Trial Key")
#             param = {
#             'machine_code': Helpers.GetMachineCode(),
#             }
#             response = requests.post(trial_url, data=param)
        
#             if response.json()[0] is 0:
#                 trial_key = response.json()[1]
#                 #Save trial Key
#                 f.seek(0)
#                 f.write(trial_key)
#                 f.truncate()
#                 f.flush()
#                 print(f'Message:{response.json()[1]}')
#                 return True
            
#             else:
#                 print(f'Could not generate a Trial key. Error Message: {response.json()[1]}')
#                 return False

#         else:
#         #Request a Verification of current key
#             param = {
#             'machine_code': Helpers.GetMachineCode(),
#             'key': license_key
#             }
#             response = requests.post(verify_url, data=param)
#             print(f'Message:{response.json()}')

#             if response.json()['success'] is True:
#                 return True
#             else:
#                 print('Verification was not a success')
#                 return False
