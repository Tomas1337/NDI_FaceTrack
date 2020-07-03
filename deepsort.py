from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.tracker import Tracker 
from deep_sort.application_util import preprocessing as prep
from deep_sort.application_util import visualization
from deep_sort.deep_sort.detection import Detection
from reid import Run_Reid
import numpy as np

import matplotlib.pyplot as plt

import torch
import torchvision
from scipy.stats import multivariate_normal
from pathlib import Path
import ptvsd

def get_gaussian_mask():
    #128 is image size
    x, y = np.mgrid[0:1.0:128j, 0:1.0:128j]
    xy = np.column_stack([x.flat, y.flat])
    mu = np.array([0.5,0.5])
    sigma = np.array([0.22,0.22])
    covariance = np.diag(sigma**2) 
    z = multivariate_normal.pdf(xy, mean=mu, cov=covariance) 
    z = z.reshape(x.shape) 

    z = z / z.max()
    z  = z.astype(np.float32)
    mask = torch.from_numpy(z)

    return mask


class deepsort_rbc():
    def __init__(self,wt_path=None):
        #loading this encoder is slow, should be done only once.
        #self.encoder = generate_detections.create_box_encoder("deep_sort/resources/networks/mars-small128.ckpt-68577")		
        self.encoder = Run_Reid()
        
        #CHANGE HERE?
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine",0.04 , 3) #Method of detection, matching thresholed, budget
        self.tracker= Tracker(self.metric)

        self.gaussian_mask = get_gaussian_mask().numpy()

        self.transforms = torchvision.transforms.Compose([ \
                torchvision.transforms.ToPILImage(),\
                torchvision.transforms.Resize((128,128)),\
                torchvision.transforms.ToTensor()])

        #self.csv_save = Csv_Save()
        self.feature_np = np.ndarray
        

    def reset_tracker(self):
        self.tracker= Tracker(self.metric)

    #Deep sort needs the format `top_left_x, top_left_y, width,height
    
    def format_yolo_output( self,out_boxes):
        for b in range(len(out_boxes)):
            out_boxes[b][0] = out_boxes[b][0] - out_boxes[b][2]/2
            out_boxes[b][1] = out_boxes[b][1] - out_boxes[b][3]/2
        return out_boxes				

    def pre_process(self,frame,detections):	

        transforms = torchvision.transforms.Compose([ \
            torchvision.transforms.ToPILImage(),\
            torchvision.transforms.Resize((128,128)),\
            torchvision.transforms.ToTensor()])

        crops = []
        for d in detections:

            for i in range(len(d)):
                if d[i] <0:
                    d[i] = 0	

            img_h,img_w,img_ch = frame.shape

            xmin,ymin,w,h = d

            if xmin > img_w:
                xmin = img_w

            if ymin > img_h:
                ymin = img_h

            xmax = xmin + w
            ymax = ymin + h

            ymin = abs(int(ymin))
            ymax = abs(int(ymax))
            xmin = abs(int(xmin))
            xmax = abs(int(xmax))

            try:
                crop = frame[ymin:ymax,xmin:xmax,:]
                crop = transforms(crop)
                crops.append(crop)
            except:
                continue

        crops = torch.stack(crops)

        return crops

    def extract_features_only(self,frame,coords):
        ptvsd.debug_this_thread()

        # for i in range(len(coords)):
        #     if coords[i] <0:
        #         coords[i] = 0	
        input_image = []
        img_h,img_w,img_ch = frame.shape
                
        xmin,ymin,w,h = coords

        if xmin > img_w:
            xmin = img_w

        if ymin > img_h:
            ymin = img_h

        xmax = xmin + w
        ymax = ymin + h

        ymin = abs(int(ymin))
        ymax = abs(int(ymax))
        xmin = abs(int(xmin))
        xmax = abs(int(xmax))
        
        crop = frame[ymin:ymax,xmin:xmax,:]
        #crop = frame[xmin:xmax,ymin:ymax,:]
        #crop = crop.astype(np.uint8)

        #print(crop.shape,[xmin,ymin,xmax,ymax],frame.shape)

        #crop = self.transforms(crop)

        #Our current model didn't train with a gaussian mask
        #gaussian_mask = self.gaussian_mask
        #input_ = crop * gaussian_mask
        #input_ = torch.unsqueeze(input_,0)

        # input_ = torch.unsqueeze(crop,0)
        # input_ = input_.numpy()

        input_image.append(crop)
        #print('input_image {}'.format(input_image))
        features = self.encoder.forward(input_image)
        features = features.detach().cpu().numpy()
        corrected_crop = [xmin,ymin,xmax,ymax]

        return features,corrected_crop

    def run_deep_sort(self, frame, out_scores, out_boxes):
        ptvsd.debug_this_thread()

        #If no current detections are present
        if out_boxes==[]:

            self.tracker.predict() #This changes the self.mean and self.covariance of the tracker.
            #print('No detections')
            trackers = self.tracker.tracks
            return trackers, []

        #If there are current detections
        detections = np.array(out_boxes) #Turn those detections into a numpy array
        #features = self.encoder(frame, detections.copy())
        #Preprocess those boxes #Resize, crop the boxes from the frame and apply a gaussian mask
        # processed_crops = self.pre_process(frame,detections).cuda()

        if len(detections) > 0:
            processed_crops = self.encoder.crop_out(frame, detections)
            #processed_crops = self.gaussian_mask * processed_crops

            #Extract features using a model/encoder and put them on the cpu
            #The features describe our object
            features = self.encoder.forward(processed_crops)
            features = features.detach().cpu().numpy()
        else:
            features = np.empty([0,2048])

        #self.csv_save.update(features)

        if len(features.shape)==1:
            features = np.expand_dims(features,0)

        dets = [Detection(bbox, score, feature) \
                    for bbox,score, feature in\
                zip(detections,out_scores, features)]

        #Convert bounding box to format `(min x, min y, max x, max y)`
        outboxes = np.array([d.tlwh for d in dets])
        outscores = np.array([d.confidence for d in dets])
        indices = prep.non_max_suppression(outboxes, 0.8, outscores)
        #indices = prep.non_max_suppression(outboxes, 1, outscores)
        dets = [dets[i] for i in indices]

        #I'm still unclear as to what this step is used for
        self.tracker.predict()

        #Runs a matching cascade on the detection objects(dets)
        #dets contains all the detection objects at the current time step
        self.tracker.update(dets)	
        
        return self.tracker,dets

class Csv_Save(object):

    def __init__(self,parent = None):
        import csv
        self.csvFile = open('sample.csv', 'wb')
        self.field_names = [] 
        self.writer = csv.writer(self.csvFile)

    def update(self, *args):

        if type(args) == np.ndarray:
            print('not yet supported')
            return

        else:
            m = args[0].round(decimals=5)
            np.savetxt(self.csvFile, m, fmt='%.5e', newline=",")
            self.csvFile.flush()

    def close_csv(self):
        self.csvFile.close()


