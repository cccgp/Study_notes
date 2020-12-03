'''
同src/vggNet_10%.py

'''
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import tensorly as tl
import tensorly
from itertools import chain
from decompositions import tucker_decomposition_conv_layer

tl.set_backend('pytorch')
### 2. decompose part-important!!!
model = torch.load("../model/vgg_10%_model").cuda()
model.eval()
model.cpu()

# Send the model to GPU

N = len(model.features._modules.keys())
for i, key in enumerate(model.features._modules.keys()):
    if i >= N - 2:
        break
    if isinstance(model.features._modules[key], torch.nn.modules.conv.Conv2d):
        conv_layer = model.features._modules[key]
        decomposed = tucker_decomposition_conv_layer(conv_layer)
        model.features._modules[key] = decomposed

    torch.save(model, 'decomposed_vgg_model_10%')
### 3. fine_tune part 



