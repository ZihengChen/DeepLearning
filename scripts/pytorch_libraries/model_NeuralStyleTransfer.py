
from __future__ import division
from torch.backends import cudnn
from torch.autograd import Variable
from torchvision import models
from torchvision import transforms
from PIL import Image
import argparse
import torch
import torchvision
import torch.nn as nn
import numpy as np
from pylab import *


use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor



# Load image file and convert it into variable
# unsqueeze for make the 4D tensor to perform conv arithmetic
def image_loader(image_path, transform):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    
    return image.type(dtype)

def tensor_shower(tensor,imH,imW, title = None, denorm = None):
    unloader = transforms.ToPILImage()
    image    = tensor.clone().cpu()  
    image    = image.view(3, imH, imW)
    if not denorm is None:
        image = denorm(image)
    image    = unloader(image)

    plt.imshow(image)
    plt.axis('off')
    if not title is None:
        plt.title(title)
        

# Pretrained VGGNet 
class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.select = ['0','5', '10', '19', '28'] 
        #self.select = ['0','2', '5', '7', '10'] 
        self.vgg    = models.vgg19(pretrained=True).features
        
    def forward(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            if int(name)<=int(self.select[-1]):
                x = layer(x)
                if name in self.select:
                    features.append(x)
        return features






