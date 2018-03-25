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



######################
# Valila GAN for MNIST
######################
class DIS_MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear( 784, 1024),
            nn.LeakyReLU(negative_slope=0.2,inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(1024,  512),
            nn.LeakyReLU(negative_slope=0.2,inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512,   128),
            nn.LeakyReLU(negative_slope=0.2,inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear( 128,  1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1,784)
        x = self.fc(x)
        return x


class GEN_MNIST(nn.Module):
    def __init__(self,):
        super(GEN_MNIST,self).__init__()
        self.fc = nn.Sequential(

            nn.Linear( 100, 256),
            nn.BatchNorm1d(256, affine=False),
            nn.LeakyReLU(negative_slope=0.2,inplace=True),

            nn.Linear( 256, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.LeakyReLU(negative_slope=0.2,inplace=True),
            
            nn.Linear( 512, 1024),
            nn.BatchNorm1d(1024, affine=False),
            nn.LeakyReLU(negative_slope=0.2,inplace=True),

            nn.Linear( 1024,784),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(-1,100)
        x = self.fc(x)
        return x
    

###################
# DC GAN for MNIST
##################

class dcDIS_MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=0),  # b, 16, 13, 13
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=0.3),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(16,32, 5, stride=2, padding=0),  # b, 32, 5, 5
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=0.3),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(32, 4, 3, stride=1, padding=0),  # b, 4, 3, 3
            nn.BatchNorm2d(4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(4*3*3, 16),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = x.view(-1, 1, 28, 28 )
        
        x = self.conv(x) 
        x = x.view(-1, 4*3*3)
        x = self.fc(x)
        return x



class dcGEN_MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(100, 64),
            nn.BatchNorm1d(64, affine=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(64, 4*3*3),
            nn.BatchNorm1d(4*3*3, affine=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(4, 32, 3, stride=1, padding=0),  # b, 32, 5, 5
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 5, stride=2, padding=0),  # b, 16, 13, 13
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1,  4, stride=2, padding=0),  # b, 1, 28, 28
            nn.BatchNorm2d(1),
            nn.Tanh()
        )


    def forward(self, x):
        x = x.view(-1,100)
        x = self.fc(x)

        x = x.view(-1,4,3,3)
        x = self.deconv(x)
        
        x = x.view(-1,784)
        return x

#####################
# DC GAN for CIRAR10
#####################

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class dcGEN_CIRAR10(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(GEN, self).__init__()

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.main(input)
        return output


class DIS(nn.Module):
    def __init__(self, ndf=64, nc=3):
        super(DIS, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        #.squeeze(1)
        return output.view(-1, 1)



#########################
#  other helper Functions
#########################


def discriminator_loss(score_real, score_fake, target_real, target_fake):
    loss_real = F.binary_cross_entropy( score_real, target_real ) 
    loss_fake = F.binary_cross_entropy( score_fake, target_fake )
    return loss_real + loss_fake 

def generator_loss(score_fake, target_real):
    loss = F.binary_cross_entropy( score_fake, target_real )
    return loss

def generator_GetNormalSeed(batch_size,N_Feature ):
    seed = np.random.normal(0,1,size=(batch_size, N_Feature))
    seed = torch.FloatTensor(seed)
    return seed

def generator_GetUniformlSeed(batch_size,N_Feature):
    seed = np.random.uniform(0,1,size=(batch_size, N_Feature))
    seed = torch.FloatTensor(seed)
    return seed


