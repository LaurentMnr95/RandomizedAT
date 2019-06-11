import torch
#from resnet import *
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from visdom import Visdom
from options_classifier import *
from argparse import ArgumentParser
from utils import *
from torch.optim import lr_scheduler
from networks import *

def get_scheduler(optimizer, policy="multistep",milestones=[60,120,160],gamma=0.2):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.start_epoch - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=milestones, gamma=gamma)
    elif policy == 'multistep':
        scheduler = lr_scheduler.MultiStepLR(optimizer,   milestones = milestones, gamma=gamma, last_epoch=-1)      
    # elif policy == 'plateau':
    #     scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', policy)
    return scheduler

def getNetwork(net_type="wide-resnet",depth=28,widen_factor=10,dropout=0.3,num_classes=10):
    if (net_type == 'lenet'):
        net = LeNet(num_classes)
        file_name = 'lenet'
    elif (net_type == 'vggnet'):
        net = VGG(depth, num_classes)
        file_name = 'vgg-'+str(depth)
    elif (net_type == 'resnet'):
        net = ResNet(depth, num_classes)
        file_name = 'resnet-'+str(depth)
    elif (net_type == 'wide-resnet'):
        net = Wide_ResNet(depth, widen_factor, dropout, num_classes)
        file_name = 'wide-resnet-'+str(depth)+'x'+str(widen_factor)
    else:
        print('Error : Network should be either [LeNet / VGGNet / ResNet / Wide_ResNet')
        sys.exit(0)

    return net, file_name

def load_data(dataset="CIFAR10",datadir="datasets",batch_size=128,train_mode=True):
    # TODO:add Imagenet 
    if dataset == "CIFAR100":
        if train_mode==True:
            transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])
        else:
            transform = transforms.Compose([
                    transforms.ToTensor(),
                ])
        loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR100(os.path.join(datadir,dataset), train=train_mode, download=True, transform=transform),
                batch_size=batch_size, shuffle=True)
        print("Loaded CIFAR 100 dataset")

    if dataset == "CIFAR10":
        if train_mode==True:
            transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])
        else:
            transform = transforms.Compose([
                    transforms.ToTensor(),
                ])
        loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(os.path.join(datadir,dataset), train=train_mode, download=True, transform=transform),
                batch_size=batch_size, shuffle=True)
        print("Loaded CIFAR 10 dataset")

    elif dataset == "MNIST":
        loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(os.path.join(datadir,dataset), train=train_mode, download=True, transform=transform),
                batch_size=batch_size, shuffle=True)
        print("Loaded MNIST dataset")
    return loader
