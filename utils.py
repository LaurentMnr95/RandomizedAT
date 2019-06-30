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
from pathlib import Path
import uuid
from dataset_folder import *
from collections import defaultdict, deque
import datetime
import pickle
import time

import torch
import torch.distributed as dist

import errno
import os


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


def load_data(job_env=None, dataset="CIFAR10", datadir="datasets",  batch_size_per_gpu=128, train_mode=True):
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

        dataset = torchvision.datasets.CIFAR100(os.path.join(
            datadir, dataset), train=train_mode, download=True, transform=transform)

        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=job_env.num_tasks, rank=job_env.global_rank)

        loader = torch.utils.data.DataLoader(dataset, num_workers=4,
                                             batch_size=batch_size_per_gpu,
                                             sampler=sampler)
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
        dataset = torchvision.datasets.CIFAR10(os.path.join(
            datadir, dataset), train=train_mode, download=True, transform=transform)

        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=job_env.num_tasks, rank=job_env.global_rank)

        loader = torch.utils.data.DataLoader(dataset, num_workers=4,
                                             batch_size=batch_size_per_gpu,
                                             sampler=sampler)
        print("Loaded CIFAR 10 dataset")

    elif dataset == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        dataset = torchvision.datasets.CIFAR100(os.path.join(
            datadir, dataset), train=train_mode, download=True, transform=transform)

        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=job_env.num_tasks, rank=job_env.global_rank)

        loader = torch.utils.data.DataLoader(dataset, num_workers=4,
                                             batch_size=batch_size_per_gpu,
                                             sampler=sampler)
        print("Loaded MNIST dataset")
    elif dataset == "ImageNet":
        if train_mode:
            transform = transforms.Compose([transforms.Resize(299),
                                            transforms.CenterCrop(299),
                                            # transforms.RandomResizedCrop(299),
                                            # transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor()])
            split = "train"
        else:
            transform = transforms.Compose([transforms.Resize(299),
                                            transforms.CenterCrop(299),
                                            transforms.ToTensor()])
            split = "val"

        dataset = ImageFolder_perso(os.path.join(datadir, split),
                                    transform)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=job_env.num_tasks, rank=job_env.global_rank)

        loader = torch.utils.data.DataLoader(dataset,
                                             sampler=sampler,
                                             batch_size=batch_size_per_gpu,
                                             num_workers=10*job_env.num_tasks,
                                             pin_memory=True)
        print("Loaded ImageNet dataset")
    return loader


def get_shared_folder() -> Path:
    if Path("/checkpoint/").is_dir():
        return Path("/checkpoint/laurentmeunier/trainers")
    if Path("/mnt/vol/gfsai-east").is_dir():
        return Path("/mnt/vol/gfsai-east/ai-group/users/laurentmeunier/trainers")
    raise RuntimeError("No shared folder available")


def get_init_file() -> Path:
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


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


def get_lp_norm(x, p=2):
    d = torch.abs(x)
    if p == np.inf:
        d, _ = d.view(d.shape[0], -1).max(dim=1)
        return d
    else:
        d = d**p
        d = (d.view(d.shape[0], -1).sum(dim=1))
        return d**(1./p)


def delete_line(file, names):
    with open(file, "r") as f:
        lines = f.readlines()
    with open(file, "w") as f:
        for line in lines:
            write_line = True
            for n in names:
                if n in line.strip("\n"):
                    write_line = False
            if write_line:
                f.write(line)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list
