import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from   torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from   torch.utils.data import Dataset, DataLoader



class AEdnn_MNIST(nn.Module):
    def __init__(self):
        super(AEdnn_MNIST, self).__init__()

        self.fc1  = nn.Linear(784, 400)
        self.fc2  = nn.Linear(400, 100)
        self.fc3  = nn.Linear(100, 20)
        self.fc4  = nn.Linear(20 , 100)
        self.fc5  = nn.Linear(100, 400)
        self.fc6  = nn.Linear(400, 784)

    def encode(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x

    def decode(self, x):
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.sigmoid(self.fc6(x))
        x = F.normalize(x)
        return x

    def forward(self, x):
        code = self.encode(x)
        recon = self.decode(code)
        return recon,code


        

class AEcnn_MNIST(nn.Module):
    def __init__(self):
        super(AEcnn_MNIST, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 40, kernel_size=5)
        self.fc1 = nn.Linear(40 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 10)

        self.fc4  = nn.Linear(10 , 32)
        self.fc5  = nn.Linear(32, 128)
        self.fc6  = nn.Linear(128, 40 * 5 * 5)
        self.deconv1 = nn.ConvTranspose2d(40, 10, kernel_size=5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(10,   1, kernel_size=4, stride=2)

    def encode(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2)
        x = x.view(-1, 40 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x


    def decode(self, x):
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        
        x = x.view(-1, 40, 5, 5 )
        x = F.relu(self.deconv1(x))
        x = F.sigmoid(self.deconv2(x))
        x = F.normalize(x)
        return x

    def forward(self, x):
        code = self.encode(x)
        recon = self.decode(code)
        
        return recon,code



def loss_BCE(recon_x, x):
    
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)

    return BCE