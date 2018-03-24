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


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder_common = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(True),
            nn.Linear(400, 100),
            nn.ReLU(True),
        )

        self.encoder_mu = nn.Linear(100, 36)
        self.encoder_logvar = nn.Linear(100, 36)


        self.decoder = nn.Sequential(
            nn.Linear(36 , 100),
            nn.ReLU(True),
            nn.Linear(100, 400),
            nn.ReLU(True),
            nn.Linear(400, 784),
            nn.Tanh()
        )

    def encode(self, x):
        x = self.encoder_common(x)
        mu     = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()

        eps = torch.FloatTensor(std.size())
        eps = eps.normal_()
        eps = Variable(eps)
        
        if std.is_cuda:
            eps = eps.cuda()

        z = eps.mul(std).add_(mu)
        return z

    def decode(self, z):
        z = self.decoder(z)
        return z

    def forward(self, x):
        x = x.view(-1, 784)
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        z = self.decode(z)
        return z, mu, logvar






class dcVAE(nn.Module):
    def __init__(self):
        super(dcVAE, self).__init__()

        self.encoder_common = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=0),  # b, 16, 13, 13
            nn.ReLU(True),
            nn.Conv2d(16,32, 5, stride=2, padding=0),  # b, 32, 4, 4
            nn.ReLU(True),
            nn.Conv2d(32, 4, 3, stride=1, padding=0),  # b, 4, 3, 3
            nn.ReLU(True),
        )

        self.encoder_mu = nn.Linear(36, 36)
        self.encoder_logvar = nn.Linear(36, 36)


        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 32, 3, stride=1, padding=0),  # b, 32, 3, 3
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 5, stride=2, padding=0),  # b, 16, 13, 13
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1,  4, stride=2, padding=0),  # b, 1, 28, 28
            nn.Tanh()
        )

    def encode(self, x):
        x = self.encoder_common(x)
        x = x.view(-1, 4*3*3)
        mu     = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()

        eps = torch.FloatTensor(std.size())
        eps = eps.normal_()
        eps = Variable(eps)

        if std.is_cuda:
            eps = eps.cuda()

        z = eps.mul(std).add_(mu)
        return z

    def decode(self, z):
        z = z.view(-1,4,3,3)
        z = self.decoder(z)
        return z

    def forward(self, x):
        x = x.view(-1, 1,28,28)
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        
        z = self.decode(z)
        z = z.view(-1, 784)
        return z, mu, logvar


def loss_BCEandKLD(recon_x, x, mu, logvar, kld_factor=1.0 ):

    x,recon_x = (x+1)/2, (recon_x+1)/2
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)

    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return BCE + KLD * kld_factor


def generator_GetNormalSeed(batch_size,N_Feature ):
    seed = np.random.normal(0,1,size=(batch_size, N_Feature))
    seed = torch.FloatTensor(seed)
    return seed


'''
model.eval()
recons = []
tstitems = DataLoader(tstset, shuffle=False)

for item in tstitems:
    inputs  = Variable(item["data"]).cuda()
    labels  = Variable(item["label"]).cuda()
    recon, mu, logvar = model(inputs)
    recons.append(recon.data.cpu().numpy())
    
recons = np.array(recons)




fig = plt.figure(figsize=(10,10))
nrow = 9
ncol = 7
for row in range(nrow):
    plt.subplot(nrow,ncol, ncol*row+1)
    target = tstset[row]['data']
    target = target.reshape(28,28).T
    imshow(target)
    for i in range(1,ncol,1):
        plt.subplot(nrow,ncol ,ncol*row+1+i)
        inputs = tstset[row]['data']
        inputs = torch.from_numpy(inputs)
        inputs = Variable(inputs).cuda()
        recon, mu, logvar = model(inputs)
        recon  = recon.data
        recon  = recon.cpu().numpy()
        recon  = recon.reshape(28,28).T
        imshow(recon)



class VAE_MNIST(nn.Module):
    def __init__(self):
        super(VAE_MNIST, self).__init__()

        self.fc1  = nn.Linear(784, 400)
        self.fc2  = nn.Linear(400, 100)
        self.fc31 = nn.Linear(100, 20)
        self.fc32 = nn.Linear(100, 20)

        self.fc4  = nn.Linear(20 , 100)
        self.fc5  = nn.Linear(100, 400)
        self.fc6  = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        mu     = self.fc31(h2)
        logvar = self.fc32(h2)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()

        eps = torch.FloatTensor(std.size())
        eps = eps.normal_()
        eps = Variable(eps)

        if std.is_cuda:
            eps = eps.cuda()

        z = eps.mul(std).add_(mu)
        return z

    def decode(self, z):
        z = F.relu(self.fc4(z))
        z = F.relu(self.fc5(z))
        z = F.sigmoid(self.fc6(z)) 
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

    
class VAEcnn_MNIST(nn.Module):
    def __init__(self):
        super(VAEcnn_MNIST, self).__init__()

        self.conv1 = nn.Conv2d(1,   10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 100, kernel_size=5)
        self.fc1 = nn.Linear(100 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc31 = nn.Linear(128, 32)
        self.fc32 = nn.Linear(128, 32)

        self.fc4  = nn.Linear(32 , 128)
        self.fc5  = nn.Linear(128, 512)
        self.fc6  = nn.Linear(512, 100 * 5 * 5)
        self.conv3 = nn.ConvTranspose2d(100, 10, kernel_size=5, stride=2)
        self.conv4 = nn.ConvTranspose2d( 10,  1, kernel_size=4, stride=2)


    def encode(self, x):
        x = x.view(-1,1,28,28)

        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2)

        x = x.view(-1, 100 * 4 * 4)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mu     = self.fc31(x)
        logvar = self.fc32(x)

        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()

        eps = torch.FloatTensor(std.size())
        eps = eps.normal_()
        eps = Variable(eps)

        if std.is_cuda:
            eps = eps.cuda()

        z = eps.mul(std).add_(mu)
        return z

    def decode(self, z):
        z = F.relu(self.fc4(z))
        z = F.relu(self.fc5(z))
        z = F.relu(self.fc6(z))
        
        z = z.view(-1, 100, 5, 5 )
        z = F.relu(self.conv3(z))
        z = F.sigmoid(self.conv4(z))
        z = z.view(-1,784)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar




def generator_GetNormalSeed(batch_size,N_Feature ):
    seed = np.random.normal(0,1,size=(batch_size, N_Feature))
    seed = torch.FloatTensor(seed)
    return seed

'''