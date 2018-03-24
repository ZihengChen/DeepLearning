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




class DNN(nn.Module):
    def __init__(self ):
        super(DNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(784,400),
            nn.Dropout(p=0.1),
            nn.ReLU(True),

            nn.Linear(400,100),
            nn.Dropout(p=0.1),
            nn.ReLU(True),

            nn.Linear(100,10),
            nn.LogSoftmax()
        )

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc(x)
        return x

    
class CNN(nn.Module):
    def __init__(self ):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=0), # b, 16, 13, 13
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.Conv2d(16,32, 5, stride=2, padding=0), # b, 32,  5,  5
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 4, 3, stride=1, padding=0),  # b, 4, 3, 3
            nn.BatchNorm2d(4),
            nn.ReLU(True)
        )

        self.fc = nn.Sequential(
            nn.Linear(4*3*3,10),
            nn.LogSoftmax()
        )

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv(x)
        x = x.view(-1, 4*3*3)
        x = self.fc(x)
        return x





'''

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DNN_4l(nn.Module):
    def __init__(self, n,m1,m2,m3,c ):
        super(DNN_4l, self).__init__()
        
        self.nlayer  = 4

        self.fc1 = nn.Linear(n , m1)
        self.fc2 = nn.Linear(m1, m2)
        self.fc3 = nn.Linear(m2, m3)
        self.fc4 = nn.Linear(m3, c )


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x))
        return x
    
def ShowOneBatch(trainloader,net,classes):
    # get some random training images
    dataiter = iter(trainloader).next()
    images = dataiter["data"]
    labels = dataiter["label"]
    # show images
    imshow(torchvision.utils.make_grid(images))
    
    # print labels
    print('GroundTruth: ', ' '.join('%1s' % classes[labels[j]] for j in range(24)))
    outputs = net(Variable(images).cuda())
    _, predicted = torch.max(outputs.data, 1)
    print('Predicted:   ', ' '.join('%1s' % classes[predicted[j]] for j in range(24)))
    
    
def TestNetAcc(testloader,net,classes):
    correct = 0
    total = 0

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for data in testloader:
        images = data["data"].cuda()
        labels = data["label"].cuda()
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        for i in range(15):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
        correct += class_correct[i]
        total   += class_total[i]
    print('Accuracy of the network on the 10000 test images: %f %%' % (
        100 * correct / total))
    
    
def imshow(img):
         # unnormalize
    npimg = img.numpy()
    npimg = 1-npimg/np.max(npimg)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    print(npimg.shape)
'''