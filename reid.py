"""Example
python script/experiment/infer_images_example.py \
--model_weight_file YOUR_MODEL_WEIGHT_FILE
"""
from __future__ import print_function

import sys
sys.path.insert(0, '.')
import time

import torch
from torch.autograd import Variable

import numpy as np
import argparse
import cv2
from PIL import Image   
import os.path as osp

from tri_loss.model.Model import Model
from tri_loss.utils.utils import load_state_dict
from tri_loss.utils.utils import set_devices
from tri_loss.utils.dataset_utils import get_im_names
from tri_loss.utils.distance import normalize
from tri_loss.utils.nn_matching import _nn_cosine_distance as cosine_distance
from tri_loss.utils.nn_matching import _nn_euclidean_distance as euclidean_distance
class Config(object):
    def __init__(self):

        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--sys_device_ids', type=eval, default=(0,))
        parser.add_argument('--resize_h_w', type=eval, default=(256, 128))
        parser.add_argument('--last_conv_stride', type=int, default=1, choices=[1, 2])
        parser.add_argument('--ckpt_file', type=str, default='')
        parser.add_argument('--model_weight_file', type=str, default='models\\res50_reid.pth')

        args = parser.parse_args()

        # gpu ids
        self.sys_device_ids = args.sys_device_ids
        # Image Processing
        self.resize_h_w = args.resize_h_w
        self.im_mean = [0.486, 0.459, 0.408]
        self.im_std = [0.229, 0.224, 0.225]

        # The last block of ResNet has stride 2. We can set the stride to 1 so that
        # the spatial resolution before global pooling is doubled.
        self.last_conv_stride = args.last_conv_stride

        # This contains both model weight and optimizer state
        self.ckpt_file = args.ckpt_file
        # This only contains model weight
        self.model_weight_file = args.model_weight_file


def pre_process_im(im, cfg):
    """Pre-process image.
    `im` is a numpy array with shape [H, W, 3], e.g. the result of
    matplotlib.pyplot.imread(some_im_path), or
    numpy.asarray(PIL.Image.open(some_im_path))."""


#    im = im.T if im.shape == (3,128,128) else im
    # Resize.
    try: 
        im = cv2.resize(im, cfg.resize_h_w[::-1], interpolation=cv2.INTER_LINEAR)
    except cv2.error:
        return im
    # scaled by 1/255.
    im = im / 255.

    # Subtract mean and scaled by std
    im = im - np.array(cfg.im_mean)
    im = im / np.array(cfg.im_std).astype(float)

    # shape [H, W, 3] -> [1, 3, H, W]
    im = im.transpose(2, 0, 1)[np.newaxis]
    return im


class Run_Reid(object):
    def __init__(self):
            
        #########
        # Model #
        #########

        cfg = Config()
        TVT, TMO = set_devices(cfg.sys_device_ids)
        model = Model(last_conv_stride=cfg.last_conv_stride)
        # Set eval mode. Force all BN layers to use global mean and variance, also disable dropout.
        model.eval()
        # Transfer Model to Specified Device.
        TMO([model])
        
        #####################
        # Load Model Weight #
        #####################
        used_file = cfg.model_weight_file or cfg.ckpt_file
        loaded = torch.load(used_file, map_location=(lambda storage, loc: storage))
        
        if cfg.model_weight_file == '':
            loaded = loaded['state_dicts'][0]

        load_state_dict(model, loaded)
        print('Loaded ReID model weights from {}'.format(used_file))

        self.model = model
        self.cfg = cfg
        self.TVT = TVT

    def crop_out(self, frame, detections):
        cropped_object_ls = []

        
        for d in detections:
            d = [0 if i < 0 else i for i in d]
            x,y,w,h = d
            cropped_object = frame[y:y+h, x:x+w, :]
            cropped_object = pre_process_im(cropped_object, self.cfg)
            cropped_object_ls.append(cropped_object)
        
        return cropped_object_ls

    def forward(self, image_ls: list):
        feat_T = torch.empty([0,2048])
        feat_T = feat_T.cuda()
        for i in image_ls:
            im = pre_process_im(i, self.cfg)
            im = Variable(self.TVT(torch.from_numpy(im).float()), volatile = False)
            feat = self.model(im)
            feat_T = torch.cat((feat_T, feat), 0)
        return feat_T
