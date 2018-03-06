
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


use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor



# Load image file and convert it into variable
# unsqueeze for make the 4D tensor to perform conv arithmetic
def load_image(image_path, transform=None, max_size=None, shape=None):
    image = Image.open(image_path)
    
    if max_size is not None:
        scale = max_size / max(image.size)
        size = np.array(image.size) * scale
        image = image.resize(size.astype(int), Image.ANTIALIAS)
    
    if shape is not None:
        image = image.resize(shape, Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image.type(dtype)

# Pretrained VGGNet 
class VGGNet(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.select = ['1','6', '11', '20', '29'] 
        self.vgg = models.vgg19(pretrained=True).features
        
    def forward(self, x):
        """Extract 5 conv activation maps from an input image.
        
        Args:
            x: 4D tensor of shape (1, 3, height, width).
        
        Returns:
            features: a list containing 5 conv activation maps.
        """
        features = []
        for name, layer in self.vgg._modules.items():
            if int(name)<=int(self.select[-1]):
                x = layer(x)
                if name in self.select:
                    features.append(x)
        return features




