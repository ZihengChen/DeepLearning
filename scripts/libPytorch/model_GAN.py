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



##############
# Valila GAN
##############
class DIS_MNIST(nn.Module):
    def __init__(self,input_dim, output_dim):
        super().__init__()

        self.fc1  = nn.Linear(input_dim, 1024)
        self.fc2  = nn.Linear(self.fc1.out_features, 512)
        self.fc3  = nn.Linear(self.fc2.out_features,  256)
        self.fc4  = nn.Linear(self.fc3.out_features, output_dim)

    def forward(self, x):

        x = F.leaky_relu(self.fc1(x),negative_slope=0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x),negative_slope=0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x),negative_slope=0.2)
        x = F.dropout(x, 0.3)
        x = F.sigmoid(self.fc4(x))
        return x


class GEN_MNIST(nn.Module):
    def __init__(self,input_dim, output_dim):
        super(GEN_MNIST,self).__init__()



        self.fc1  = nn.Linear(input_dim, 256)

        self.bn2  = nn.BatchNorm1d(self.fc1.out_features, affine=False)
        self.fc2  = nn.Linear(self.fc1.out_features, 512)

        self.bn3  = nn.BatchNorm1d(self.fc2.out_features, affine=False)
        self.fc3  = nn.Linear(self.fc2.out_features, 1024)

        self.bn4  = nn.BatchNorm1d(self.fc3.out_features, affine=False)
        self.fc4  = nn.Linear(self.fc3.out_features, output_dim)

    def forward(self, x):
        

        x = F.leaky_relu(self.fc1(x),negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x),negative_slope=0.2)
        x = F.leaky_relu(self.fc3(x),negative_slope=0.2)
        x = F.tanh(self.fc4(x))

        return x
    



##############
# DC GAN
##############

class DIScnn_MNIST(nn.Module):
    def __init__(self):
        super(DIScnn_MNIST, self).__init__()

        self.conv1 = nn.Conv2d( 1, 32,kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(32, 64,kernel_size=5, stride=1)

        self.fc1 = nn.Linear(64*4*4, 128)
        self.fc2 = nn.Linear(128,  1)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28 )
        
        x = F.max_pool2d( F.relu(self.conv1(x)), kernel_size=2) 
        x = F.max_pool2d( F.relu(self.conv2(x)), kernel_size=2) 

        x = x.view(-1, 64*4*4)


        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = F.sigmoid(self.fc2(x))
        return x



class GENcnn_MNIST(nn.Module):
    def __init__(self):
        super(GENcnn_MNIST,self).__init__()

        self.fc1  = nn.Linear(100, 1024)
        self.fc2  = nn.Linear(self.fc1.out_features, 128*7*7)

        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d( 64,  1, kernel_size=4, stride=2, padding=1)
        
        self.fc1_bn     = nn.BatchNorm1d(self.fc1.out_features)
        self.fc2_bn     = nn.BatchNorm1d(self.fc2.out_features)
        self.deconv1_bn = nn.BatchNorm2d(self.deconv1.out_channels)

    def forward(self, x):

        x = self.fc1_bn(F.relu(self.fc1(x)))
        x = self.fc2_bn(F.relu(self.fc2(x)))
        
        x = x.view(-1, 128, 7, 7 )
        x = self.deconv1_bn(F.relu(self.deconv1(x)))
        x = F.tanh(self.deconv2(x))

        x = x.view(-1,784)
        return x

##############
# DC GAN
##############

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class GEN(nn.Module):
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
        return output.view(-1, 1)#.squeeze(1)


#dis = DIS()
#gen = GEN()
#gen.apply(weights_init)
#dis.apply(weights_init)




##############
#  Functions
##############



def discriminator_loss(score_real, score_fake, target_real, target_fake):
    """
    Computes the discriminator loss described above.
    
    dis_loss = - torch.mean(torch.log(score_real) + torch.log(1.0 - score_fake))


    Inputs:
    - score_real: PyTorch Variable of shape (N,) giving scores for the real data.
    - score_fake: PyTorch Variable of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Variable containing (scalar) the loss for the discriminator.
    """

    loss_real = F.binary_cross_entropy( score_real, target_real ) 
    loss_fake = F.binary_cross_entropy( score_fake, target_fake )

    loss = loss_real + loss_fake 

    return loss


def generator_loss(score_fake, target_real):
    """
    Computes the generator loss described above.
    
    gen_loss = - torch.mean(torch.log(score_fake))

    Inputs:
    - score_fake: PyTorch Variable of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Variable containing the (scalar) loss for the generator.
    """

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




'''
gen_optimizer = optim.Adam(gen.parameters(), lr=1e-4)
dis_optimizer = optim.Adam(dis.parameters(), lr=1e-4)

gen.train()
dis.train()
# begin Training
for epoch in range(10):
    running_gen_loss = 0.0
    running_dis_loss = 0.0
    for i, batch in enumerate(batches):
        
        # 1. training dis and gen
        # 1.1 get input
        inputs_real = Variable(batch["data"]).cuda()
        gen_seed    = Variable(generator_GetNormalSeed(BATCHSIZE,SEEDDIM)).cuda()
        inputs_fake = gen(gen_seed)
        #inputs_fake = inputs_fake.detach()    
        
        # 1.2 get scores for real and fake
        score_real = dis(inputs_real)
        score_fake = dis(inputs_fake)
        
        # 1.3 calc loss
        dis_loss = discriminator_loss(score_real, score_fake, 
                                      target_real,target_fake)
        gen_loss = generator_loss(score_fake, target_real)
        #dis_loss = - torch.mean(torch.log(score_real) + torch.log(1.0 - score_fake))
        #gen_loss = - torch.mean(torch.log(score_fake))
        
        # 1.4 backprop and optimize
        dis_optimizer.zero_grad()
        dis_loss.backward(retain_graph=True)
        dis_optimizer.step()
        
        gen_optimizer.zero_grad()
        gen_loss.backward()
        gen_optimizer.step()      
        

        # TRAINING DESHBOARD #
        ##################################################################
        running_gen_loss += gen_loss.data[0]
        running_dis_loss += dis_loss.data[0]
        if i % 100 == 99:    # print every 100 mini-batches
            print('[{:5d}, {:5d}] gen-loss: {:8.6f}, dis-loss: {:8.6f}'.
                  format(epoch+1, i+1, running_gen_loss/100, running_dis_loss/100))
            clear_output(wait=True)
            running_gen_loss = 0.0
            running_dis_loss = 0.0
        ##################################################################


        # 1. training dis
        # 1.1 get input
        inputs_real = Variable(batch["data"]).cuda()
        gen_seed    = Variable(generator_GetNormalSeed(100,100)).cuda()
        inputs_fake = gen(gen_seed)
        inputs_fake = inputs_fake.detach()    
        
        # 1.2 get scores for real and fake
        score_real = dis(inputs_real)
        score_fake = dis(inputs_fake)
        
        # 1.3 calc loss
        dis_loss = - torch.mean(torch.log(score_real) + torch.log(1.0 - score_fake))
        
        # 1.4 backprop and optimize
        dis_optimizer.zero_grad()
        dis_loss.backward(retain_graph=True)
        dis_optimizer.step()
        
        # 2. training gen
        gen_seed    = Variable(generator_GetNormalSeed(100,100)).cuda()
        inputs_fake = gen(gen_seed)
        score_fake  = dis(inputs_fake)
        gen_loss    = - torch.mean(torch.log(score_fake))
        
        gen_optimizer.zero_grad()
        gen_loss.backward()
        gen_optimizer.step() 
'''

'''
# load MNIST as normal images
trnset = np.fromfile("../data/MNIST/MNIST_train_data.dat").reshape(-1,785)
trnset[:,:-1] = 255*trnset[:,:-1]/trnset[:,:-1].max(axis=1)[:,None]
trnset[:,-1]  = trnset[:,-1] - 1


trans  = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(mean=(0.5, 0.5, 0.5), 
                                                  std=(0.5, 0.5, 0.5))])
# data_loader
trnset = tcDataset(trnset,shape=(-1,28,28,1),transform = trans)
'''

'''
BATCHSIZE = 60
SEEDDIM   = 100

# load CSV into [0,255] image
trnset = np.fromfile("../data/MNIST/MNIST_train_data.dat").reshape(-1,785)
trnset[:,:-1] = 255*trnset[:,:-1]/trnset[:,:-1].max(axis=1)[:,None]
trnset[:,-1]  = trnset[:,-1] - 1

# normalize the MNIST data
trans  = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(mean=(0.5, 0.5, 0.5), 
                                                  std=(0.5, 0.5, 0.5))])

# data_loader
trnset = tcDataset(trnset,shape=(-1,28,28,1),transform = trans)
batches = DataLoader(trnset, batch_size=BATCHSIZE, shuffle=False)
'''

