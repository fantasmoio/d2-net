import logging

import torch
from torch2trt import torch2trt

import numpy as np
import imageio
from lib.model_test import D2Net
from lib.utils import preprocess_image

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.exceptions import EmptyTensorError
from lib.utils import interpolate_dense_features, upscale_positions

import argparse

# Parse input
parser = argparse.ArgumentParser()
parser.add_argument('--imageWidth', type=int, required=True, help='The image width we optimize the model for')
parser.add_argument('--imageHeight', type=int, required=True, help='The image height we optimize the model for')
parser.add_argument('--outputModelName', required=True, help='The name of the output model (will be saved in models/)')
parser.add_argument('--fp16', required=True, choices=['on', 'off'], help='Floating point 16 optimization')
args = parser.parse_args()

# Constants
model_file = "models/d2_tf.pth"

# CUDA
use_cuda = torch.cuda.is_available()
logging.info(("Using cuda, ", use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")

# Creating CNN model
model = D2Net(
    model_file=model_file,
    use_relu=True,
    use_cuda=use_cuda
)
print(model)

x = torch.ones((1, 3, args.imageHeight, args.imageWidth)).cuda()

if(args.fp16):
    model_trt = torch2trt(model.dense_feature_extraction, [x], fp16_mode=True)
else:
    model_trt = torch2trt(model.dense_feature_extraction, [x], fp16_mode=False)

torch.save(model_trt.state_dict(), 'models/' + str(args.outputModelName) + '.pth')
