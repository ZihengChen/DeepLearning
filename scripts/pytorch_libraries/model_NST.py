import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from PIL import Image

import torchvision.transforms as transforms
import torchvision.models as models
import torchvision

from copy import *

from pylab import *

from IPython.display import clear_output

plt.ion()
imsize = 200

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor



# Load image file and convert it into variable
# unsqueeze for make the 4D tensor to perform conv arithmetic
def image_loader(image_name, transform=None):
    image = Image.open(image_name)
    image = Variable(transform(image))
    # fake batch dimension required to fit network's input dimensions
    image = image.unsqueeze(0)
    return image


def imgshow(tensor, title=None):
    unloader = transforms.ToPILImage()
    image = tensor.clone().cpu()  # we clone the tensor to not do changes on it
    image = image.view(3, imsize, imsize)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    #plt.pause(0.001) # pause a bit so that plots are updated
    
class GramMatrix(nn.Module):

    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

class ContentLoss(nn.Module):

    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        self.target = target.detach() * weight
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target)
        self.output = input
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

class StyleLoss(nn.Module):

    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.output = input.clone()
        self.G = self.gram(input)
        self.G.mul_(self.weight)
        self.loss = self.criterion(self.G, self.target)
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss



def get_style_model_and_losses(style_img, content_img,
                               style_weight=1000, content_weight=1,
                               content_layers=['conv_4'],
                               style_layers=['conv_1', 'conv_2', 'conv_3', 
                                             'conv_4', 'conv_5']):
    cnn = models.vgg19(pretrained=True).features
    cnn = deepcopy(cnn).cuda()

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses   = []

    model = nn.Sequential()  # the new Sequential module network
    gram = GramMatrix()  # we need a gram module in order to compute style targets

    # move these modules to the GPU if possible:
    if use_cuda:
        model = model.cuda()
        gram  = gram.cuda()

    i = 1
    for layer in list(cnn):
        if isinstance(layer, nn.Conv2d):
            name = "conv_" + str(i)
            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)

        if isinstance(layer, nn.ReLU):
            name = "relu_" + str(i)
            model.add_module(name, layer)

            i += 1

        if isinstance(layer, nn.MaxPool2d):
            name = "pool_" + str(i)
            model.add_module(name, layer)  # ***

    return model, style_losses, content_losses


def run_style_transfer(content_img, style_img, input_img, 
                       num_steps=10,
                       style_weight=1000, content_weight=1):
    """Run the style transfer."""
    
    model,style_losses,content_losses = get_style_model_and_losses(style_img, content_img, style_weight, content_weight)
    
    input_param = nn.Parameter(input_img.data)
    optimizer   = optim.LBFGS([input_param])

    for i in range(num_steps):

        def closure():
            # correct the values of updated input image
            input_param.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_param)
            
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.backward()
            for cl in content_losses:
                content_score += cl.backward()
            
            print('{:3d} Style Loss : {:4f} Content Loss: {:4f}'.format(i, style_score.data[0], content_score.data[0]))
            clear_output(wait=True)

            return style_score + content_score


        optimizer.step(closure)

        
        
        img = input_param.clone()
        img = img.data.clamp_(0, 1)
        torchvision.utils.save_image(img, '../../plot/nst/output-{}.png'.format(i+1))

    # a last correction...
    input_param.data.clamp_(0, 1)
    return input_param.data

'''
plt.subplot(1,2,1)
imgshow(style_img.data, imsize, title='Style Image')
plt.subplot(1,2,2)
imgshow(content_img.data, imsize, title='Content Image')

'''