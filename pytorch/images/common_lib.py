import cv2
import time
import io
import sys
import os
import numpy as np

from torch import nn
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
import torch.onnx

import torchvision.transforms as transforms
import torchvision.models as models

model_dict={
"alexnet": models.alexnet,
"vgg16": models.vgg16,
"vgg19": models.vgg19,
"resnet18": models.resnet18, # NG
"resnet34": models.resnet34, # NG
"squeezenet1_0": models.squeezenet1_0, # NG: can't convert
"densenet161": models.densenet161,  # NG: cuda needed
"inception": models.inception_v3, # NG: can't convert
}

classfile="imagenet1000_clsid_to_human.txt"

if not os.path.exists(classfile):
    os.system("wget https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/c2c91c8e767d04621020c30ed31192724b863041/imagenet1000_clsid_to_human.txt")

f=open(classfile).read()
exec("class_dict="+f)
