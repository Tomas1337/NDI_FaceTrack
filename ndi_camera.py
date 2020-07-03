import sys
import numpy as np
import cv2 as cv
import NDIlib as ndi
import time


class ndi_camera:
    def __init__(self):

        sources = []

        if not ndi.initialize():
            print('None')

        #start a find object
        ndi_find = ndi.find_create_v2()

        if ndi_find is None:
            print('None')
        
        #While the sources are empty keep looking
        while not len(sources) > 0:
            print('Looking for sources ...')
            ndi.find_wait_for_sources(ndi_find, 5000)
            sources = ndi.find_get_current_sources(ndi_find)

        #When found, create an instance using RecvCreateV3()        
        #This is a generic object

        
        ndi_recv_create = ndi.RecvCreateV3()
        ndi_recv_create.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
        ndi_recv_create.bandwidth = ndi.RECV_BANDWIDTH_LOWEST

        #NDI object that is the video stream?
        ndi_recv = ndi.recv_create_v3(ndi_recv_create)
        if ndi_recv is None:
            print('None')
        self.ndi_find = ndi_find
        self.ndi_recv = ndi_recv
        self.sources = sources
    

    def find_ptz(self):
        #initiate this temp object for checking
        ptz_list = []
        
        for i, src in enumerate(self.sources):
            #Connect to each source to receive meta_data
            ndi.recv_connect(self.ndi_recv, src)
            _,v,_,_ = ndi.recv_capture_v2(self.ndi_recv, 10000)

            #if positive
            if ndi.recv_ptz_is_supported(self.ndi_recv):
                print('PTZ camera found')
                ndi.recv_free_video_v2(self.ndi_recv, v)
                ptz_list.append(i)

            else:
                print("Checking next NDI source")

        return ptz_list, self.sources

    def camera_connect(self, src):
        #Connect to the object
        ndi.recv_connect(self.ndi_recv, self.sources[src])

        #Destory the find object
        ndi.find_destroy(self.ndi_find)
        print("Camera connected")
        return self.ndi_recv




