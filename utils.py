import torch
# from resnet import *
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
import torch.nn as nn
from torch.distributions import normal, laplace


def get_scheduler(optimizer, policy="multistep", milestones=[60, 120, 160], gamma=0.2):
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
            lr_l = 1.0 - max(0, epoch + opt.start_epoch -
                             opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=milestones, gamma=gamma)
    elif policy == 'multistep':
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,   milestones=milestones, gamma=gamma, last_epoch=-1)
    # elif policy == 'plateau':
    #     scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', policy)
    return scheduler


def getNetwork(net_type="wide-resnet", depth=28, widen_factor=10, dropout=0.3, num_classes=10):
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
    elif (net_type == 'inceptionresnetv2'):
        net = InceptionResNetV2(num_classes)
        file_name = 'inceptionresnetv2'
    else:
        print(
            'Error : Network should be either [LeNet / VGGNet / ResNet / Wide_ResNet')
        sys.exit(0)

    return net, file_name


def load_data(dataset="CIFAR10", datadir="datasets", batch_size=128, train_mode=True):
    # TODO:add Imagenet
    if dataset == "CIFAR100":
        if train_mode == True:
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
            torchvision.datasets.CIFAR100(os.path.join(
                datadir, dataset), train=train_mode, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
        print("Loaded CIFAR 100 dataset")

    if dataset == "CIFAR10":
        if train_mode == True:
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
            torchvision.datasets.CIFAR10(os.path.join(
                datadir, dataset), train=train_mode, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
        print("Loaded CIFAR 10 dataset")

    elif dataset == "MNIST":
        loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(os.path.join(
                datadir, dataset), train=train_mode, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
        print("Loaded MNIST dataset")
    elif dataset == "ImageNet":
        if train_mode:
            transform = transforms.Compose([transforms.Resize(299),
                                            transforms.CenterCrop(299),
                                            # transforms.RandomResizedCrop(299),
                                            # transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor()])
            loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(os.path.join(datadir, "train"),
                                                 transform),
                batch_size=batch_size,
                shuffle=True,
                num_workers=8,
                pin_memory=True)
        else:
            transform = transforms.Compose([transforms.Resize(299),
                                            transforms.CenterCrop(299),
                                            transforms.ToTensor()])
            loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(os.path.join(datadir, "val"),
                                                 transform),
                batch_size=batch_size,
                shuffle=True,
                num_workers=8,
                pin_memory=True)
        print("Loaded ImageNet dataset")
    return loader


class RandModel(nn.Module):
    def __init__(self, classifier, num_classes=10, noise=None, sigma=0.25):
        super(RandModel, self).__init__()
        self.classifier = classifier
        if noise == "Normal":
            self.noise = normal.Normal(0, sigma)
            self.sigma = sigma
        elif noise == "Laplace":
            self.noise = laplace.Laplace(0, sigma/np.sqrt(2))
            self.sigma = sigma
        else:
            self.noise = None

    def forward(self, x):
        if self.noise == None:
            return self.classifier(x)
        else:
            if x.is_cuda:
                return self.classifier(x+self.noise.sample(x.shape).cuda())
            else:
                return self.classifier(x+self.noise.sample(x.shape))


def delete_line(file, name):
    with open(file, "r") as f:
        lines = f.readlines()
    with open(file, "w") as f:
        for line in lines:
            if name not in line.strip("\n"):
                f.write(line)
