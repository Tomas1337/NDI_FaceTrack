import requests
import win32pipe, win32file
import os, io
import pywintypes
import ctypes
import numpy as np
from PIL import Image

class PipeClient:
    def __init__(self, parent = None):
        #self.pipeRequest("http://127.0.0.1:5000/api/start_pipe_server")
        pass
    
    def pipeRequest(self, url, content = None):
        "Requests to start a Pipe Server on the Tracking Server"
        r = requests.post(url = url, data = content)
        if r.ok:
            print("Post Request success with named Pipe {}".format(r.text))
            self.pipeName = r.json()
            
            print(f'creating handle to {self.pipeName}')
            self.pipeHandle = win32file.CreateFile(
            r'\\.\pipe\{}'.format(self.pipeName),
            win32file.GENERIC_READ | win32file.GENERIC_WRITE,
            0,
            None,
            win32file.OPEN_EXISTING,
            0,
            None
            )

            return r.text
        else:
            print("Post request was not successful")
            return False        

    def writeToPipe(self, payload = "none"):
        "Writes frame data to the Pipe"
        try:
            win32file.WriteFile(self.pipeHandle, payload)
            #print(f'Writing data of length {len(payload)} to pipe')
        except:
            print("Could not write to pipe from Client Side")

    def readFromPipe(self):
        try:
            _, r =win32file.ReadFile(self.pipeHandle, 128)
        except pywintype.error as e:
            print("Could not read from pipe from Client Side")
            r =b''
        finally:
            return r
    
    def genRandByteArray(self):
        "Convenience function to Generate a Random Byte array"
        data = np.random.randint(0,high=254, size =(480,640,3))

        img = Image.fromarray(data.astype(np.uint8))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='BMP')
        img_byte_arr = img_byte_arr.getvalue()

        return img_byte_arr

    def pipeClose(self):
        win32file.CloseHandle(self.pipeHandle)
        print("Pipe has been closed from client")

        


