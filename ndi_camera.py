import sys
import numpy as np
import cv2 as cv
import time
import NDIlib as ndi
# import ptvsd

class ndi_camera:
    def __init__(self):
        #ptvsd.debug_this_thread()
        self.sources = []
        self.ndi_find = None
        self.ndi_recv_create = None
        self.ndi_recv = None

    def find_sources(self):
        if not ndi.initialize():
            print('None')

        #start a find object
        self.ndi_find = ndi.find_create_v2()

        if self.ndi_find is None:
            print('None')
        
        #While the sources are empty keep looking
        while not len(self.sources) > 0:
            print('Looking for sources ...')
            ndi.find_wait_for_sources(self.ndi_find, 1000)
            self.sources = ndi.find_get_current_sources(self.ndi_find)
            print("Sources Length: {}".format(len(self.sources)))
            print(f'Camera has URL address of {self.sources[0].url_address}')

            
    

    def find_ptz(self):
        #initiate this temp object for checking
        #TODO
        ptz_list = []
        for i, src in enumerate(self.sources):
            #Connect to each source to receive meta_data
            #ndi.recv_connect(self.ndi_recv, src)
            #_,self.v,_,_ = ndi.recv_capture_v2(self.ndi_recv, 5000)
            # #if positive
            # if ndi.recv_ptz_is_supported(self.ndi_recv):
            #     print('PTZ camera found')
            #     ndi.recv_free_video_v2(self.ndi_recv, self.v)
            #     ptz_list.append(i)

            # else:
            #     #print("Checking next NDI source") 
            #     ndi.recv_free_video_v2(self.ndi_recv, self.v)
            #     ptz_list.append(i)

            #ndi.recv_free_video_v2(self.ndi_recv, self.v)
            ptz_list.append(i)
        return ptz_list, self.sources

    def camera_connect(self, src=None, ndi_name=None, ndi_address=None):
        #ptvsd.debug_this_thread()
        """
        Connect to the camera and returns a connected NDI Receive Object
        Args:
            src (int, optional): index indicating which camera to connect to in the self.sources list. Defaults to None.
            ndi_name (string, optional): Name of Camera to connect to. Defaults to None. Ex: 'DESKTOP-C16VMFB (VLC)'
            ndi_address (string, optional): IP Address of camera to connect to. Must match with ndi_name. Defaults to None. Ex:'192.168.2.48:5961

        Returns:
            NDI Receive Object: Fully defined Receive Object ready to be connected to
        """
        ndi_recv = self.create_ndi_recv_object()

        if src is not None:
            ndi.recv_connect(ndi_recv, self.sources[src])
            print(f'NDI Name: {self.sources[0].ndi_name}')
            print(f'NDI Address: {self.sources[0].url_address}')

        elif ndi_name is not None:
            ndi_recv = self.create_ndi_recv_object()
            ndi_source = self.create_ndi_source_preset(ndi_name)
            ndi.recv_connect(ndi_recv, ndi_source)
            print(f'NDI Name Preset: {ndi_name}')
            
        else:
            print("Could not create a proper NDI Receive Object")
            return None

        _,v,_,_ = ndi.recv_capture_v2(ndi_recv, 500)
        ndi.recv_free_video_v2(ndi_recv, v)
        #Destory the find object
        ndi.find_destroy(self.ndi_find)

        return ndi_recv

    def create_ndi_recv_object(self):      
        #Creates a generic (empty) properties object for an empty ndi_recv object
        ndi_recv_create = ndi.RecvCreateV3()
        ndi_recv_create.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
        ndi_recv_create.bandwidth = ndi.RECV_BANDWIDTH_LOWEST

        ndi_recv = ndi.recv_create_v3(ndi_recv_create)

        if ndi_recv is None:
            print('No NDI Receive Object')
            
        return ndi_recv
    
    def create_ndi_source_preset(self, ndi_name):      
        #Creates a source object manually
        ndi_source_obj = ndi.Source()
        ndi_source_obj.ndi_name = ndi_name
        return ndi_source_obj


