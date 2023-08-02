import os, io, sys
import requests, pywintypes
import win32pipe, win32file
import ctypes, pickle
import numpy as np
from PIL import Image
import pickle
from pickle import UnpicklingError

sys.path.insert(0, 'tool')

class PipeClient:
    def __init__(self, parent = None):
        #self.pipeRequest("http://127.0.0.1:8000/api/start_pipe_server")
        self.pipe_handle = None
        pass

    def pipeRequest(self, url, content = None):
        """Requests to start a Pipe Server on the Tracking Server given by the url
        creates a pipe-handle """
        r = requests.post(url = url, data = content)
        if r.ok:
            print("Post Request success with named Pipe {}".format(r.text))
            pipeName = r.json()
            return pipeName
        else:
            print("Post request was not successful")
            return False  

    def createPipeHandle(self, pipeParams):
        pipeName = pipeParams.get('pipeName')
        print(f'creating handle to {pipeName}')
        try:
            self.pipe_handle = win32file.CreateFile(
            r'\\.\pipe\{}'.format(pipeName),
            win32file.GENERIC_READ | win32file.GENERIC_WRITE,
            0,
            None,
            win32file.OPEN_EXISTING,
            0,
            None
            )
            return True

        except NameError as e:
            print(f'Not able to create pipe_handle. Error: {e}')
            return False

    def getPipeHandle(self):
        try:
            return self.pipe_handle
        except NameError:
            return None
        

    def writeToPipe(self, payload = "none"):
        "Writes frame data to the Pipe"
        try:
            win32file.WriteFile(self.pipe_handle, payload)
            #print(f'Writing data of length {len(payload)} to pipe')
        except AttributeError as e:
            print(f"Error with message {e}")
        except:
            print("Could not write to pipe from Client Side")

            
    def readFromPipe(self):
        r=None
        try:
            bufSize = 4096
            win32file.SetFilePointer(self.pipe_handle, 0, win32file.FILE_BEGIN)
            result, data = win32file.ReadFile(self.pipe_handle, bufSize, None) 
            buf = data
            while len(data) == bufSize:            
                result, data = win32file.ReadFile(self.pipe_handle, bufSize, None)
                buf += data
            r = buf

        except pywintypes.error as e:
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
        if self.pipe_handle:
            win32file.CloseHandle(self.pipe_handle)
            print("Pipe has been closed from client")
        
    def checkPipe(self, pipe_handle = None):
        """Checks if pipe at fileHandle is open
        Args:
            fileHandle : Handler for the Pipe

        Returns:
            response
                0: success
                1: Error'd out
        """
        if pipe_handle is None:
            pipe_handle = self.pipe_handle
        
        try:
            pipeFlag, _, _, _ = win32pipe.GetNamedPipeInfo(pipe_handle)
            if pipeFlag > 0:
                pipeState = True
            else: 
                pipeState = False

        except pywintypes.error as e:
            pipeState = False
        
        return pipeState

    @staticmethod
    def unpickle_object(data):
        try:
            return pickle.loads(data)
        except (EOFError, UnpicklingError):
            return ('Cannot unpickle object')