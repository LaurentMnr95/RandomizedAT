import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import numpy as np
import os
from visdom import Visdom
from options_classifier import *
from argparse import ArgumentParser
from utils import *
from networks import *
import sys
import time
import copy
from advertorch import attacks
from torch.distributions import normal, laplace
import random
import submitit
# TODO: add EAD, FGM L1
# define options


def main(path_model="model_test/blabla",
         dataset='ImageNet', num_classes=1000,
         epochs=200, batch_size=64,
         resume_epoch=0, save_frequency=2,
         adversarial_training=None, attack_list=["PGDLinf", "PGDL2"],
         eot_samples=1,
         noise=None, sigma=0.25):

    job_env = submitit.JobEnvironment()
    torch.cuda.set_device(job_env.local_rank)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=TOOOOOOOOOOO,
        world_size=job_env.num_tasks,
        rank=job_env.global_rank,
    )
    print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")
    if not os.path.exists(path_model):
        os.makedirs(path_model)

    # Load inputs
    if dataset == "ImageNet":
        train_loader = load_data(dataset=dataset,
                                 datadir="datasets",  # "/datasets01_101/imagenet_full_size/061417/",  # to adapt
                                 batch_size=batch_size, train_mode=True)
    else:
        train_loader = load_data(dataset=dataset, datadir="datasets",
                                 batch_size=batch_size, train_mode=True)

    num_images = len(train_loader.dataset)

    # Classifier  definition
    if dataset == "ImageNet":
        Classifier = models.densenet161(pretrained=False)

        # Classifier, modelname = getNetwork(net_type='inceptionresnetv2', num_classes=num_classes)
    else:
        Classifier, modelname = getNetwork(net_type="wide-resnet", depth=28, widen_factor=10,
                                           dropout=0.3, num_classes=num_classes)
        Classifier.apply(conv_init)

    Classifier = RandModel(Classifier, noise=noise, sigma=sigma)
    Classifier.cuda()
    Classifier = torch.nn.DataParallel(
        Classifier, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
    print("Classifier intialized on:")
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))
    Classifier.train()

    # optimizer and criterion
    if adversarial_training == "MixMax":
        criterion = torch.nn.CrossEntropyLoss(reduction="none").cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        Classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    if dataset == "ImageNet":
        scheduler = get_scheduler(optimizer, policy="multistep", milestones=[
            60, 120, 160], gamma=0.2)
    else:
        scheduler = get_scheduler(optimizer, policy="multistep", milestones=[
            30, 60, 90], gamma=0.2)

    # resume learning
    if resume_epoch > 0:
        if os.path.isfile(path_model):
            print("=> loading checkpoint '{}'".format(path_model))
            checkpoint = torch.load(path_model)
            Classifier = checkpoint['net']
            print("=> loaded checkpoint (epoch {})".format(
                checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(path_model))

    adversaries = dict()

    adversaries["CW"] = attacks.CarliniWagnerL2Attack(Classifier, num_classes,
                                                      learning_rate=0.01, binary_search_steps=9,
                                                      max_iterations=15, abort_early=True,
                                                      initial_const=0.001, clip_min=0.0, clip_max=1.)

    adversaries["EAD"] = attacks.ElasticNetL1Attack(Classifier, num_classes,
                                                    confidence=0,
                                                    targeted=False, learning_rate=0.01,
                                                    binary_search_steps=9, max_iterations=60,
                                                    abort_early=True, initial_const=1e-3,
                                                    clip_min=0., clip_max=1., beta=1e-3, decision_rule='EN')

    adversaries["PGDL1"] = attacks.SparseL1PGDAttack(Classifier, eps=10., nb_iter=10, eps_iter=2*10./10,
                                                     rand_init=False, clip_min=0.0, clip_max=1.0,
                                                     sparsity=0.05, eot_samples=eot_samples)

    adversaries["PGDLinf"] = attacks.LinfPGDAttack(Classifier, eps=0.031, nb_iter=10, eps_iter=2*0.031/10,
                                                   rand_init=True, clip_min=0.0, clip_max=1.0, eot_samples=eot_samples)

    adversaries["PGDL2"] = attacks.L2PGDAttack(Classifier, eps=2., nb_iter=10, eps_iter=2*2./10,
                                               rand_init=True, clip_min=0.0, clip_max=1.0, eot_samples=eot_samples)

    adversaries["FGSM"] = attacks.GradientSignAttack(Classifier, loss_fn=None, eps=0.05, clip_min=0.,
                                                     clip_max=1., targeted=False, eot_samples=eot_samples)
    # TO add L1 attacks

    for epoch in range(epochs):
        current_num_input = 0

        running_loss = 0.0
        running_acc = 0

        start_time_epoch = time.time()
        for i, data in enumerate(train_loader, 0):

            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            if adversarial_training is None:
                outputs = Classifier(inputs)
                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            if adversarial_training == "Single":
                inputs_adv = adversaries[attack_list[0]].perturb(
                    inputs, labels)
                outputs = Classifier(inputs_adv)
                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            elif adversarial_training == "MixMean":
                loss = 0
                for att in attack_list:
                    inputs_adv = adversaries[att].perturb(inputs, labels)
                    outputs = Classifier(inputs_adv)
                    loss += criterion(outputs, labels)
                loss /= len(attack_list)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            elif adversarial_training == "MixRand":
                att = random.choice(attack_list)
                inputs_adv = adversaries[att].perturb(inputs, labels)
                outputs = Classifier(inputs_adv)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            elif adversarial_training == "MixMax":
                loss = torch.zeros_like(labels).float()
                for att in attack_list:
                    inputs_adv = adversaries[att].perturb(inputs, labels)
                    outputs = Classifier(inputs_adv)
                    l = criterion(outputs, labels).float()
                    loss = torch.max(loss, l)
                loss = loss.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                outputs = Classifier(inputs)
                _, predicted = torch.max(outputs.data, 1)

            running_loss += loss.item()
            running_acc += predicted.eq(labels.data).cpu().sum().numpy()
            curr_batch_size = inputs.size(0)

            if i % 5 == 4:
                print("Epoch :[", epoch+1, "/", epochs,
                      "] [", i*batch_size, "/", num_images,
                      "] Running loss:", running_loss/20,
                      ", Running accuracy:", running_acc/(20*curr_batch_size), " time:", time.time()-start_time_epoch)
                running_loss = 0.0
                running_acc = 0

        # save model
        if (epoch + 1) % save_frequency == 0:

            state = {
                'epoch': epoch + 1,
                'net': Classifier.module.classifier,
            }
            torch.save(state, os.path.join(
                path_model, "epoch_"+str(epoch+1)+'.t7'))

        scheduler.step()


if __name__ == "__main__":
    main()
