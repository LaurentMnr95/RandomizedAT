import torch
#from resnet import *
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

# TODO: add L1 attacks + eot CW
# define options


def main(path_model="/private/home/laurentmeunier/RAT/models/CIFAR10/baseline/epoch_68.t7",
         result_file='blabla.txt',
         attacks_norm="norm.txt",
         dataset="CIFAR10", num_classes=10,
         batch_size=256,
         attack="PGDL2", eot_samples=1,  # 80
         noise=None,
         batch_prediction=1, sigma=0.25, save_image=False):

    if noise is None:
        batch_prediction = None
        sigma = None
    # Load inputs
    test_loader = load_data(dataset=dataset, datadir="datasets",
                            batch_size=batch_size, train_mode=False)
    num_images = len(test_loader.dataset)

    # Classifier  definition
    model_load = torch.load(path_model)
    Classifier = model_load["net"]
    epoch = model_load["epoch"]
    Classifier = RandModel(Classifier, noise=noise, sigma=sigma)
    Classifier.cuda()
    Classifier = torch.nn.DataParallel(Classifier, device_ids=range(torch.cuda.device_count()))

    cudnn.benchmark = True
    print("Classifier intialized")
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))
    # print(Classifier)
    Classifier.eval()

    adversaries = dict()

    adversaries["CW"] = attacks.CarliniWagnerL2Attack(Classifier, num_classes,
                                                      learning_rate=0.01, binary_search_steps=9,
                                                      max_iterations=60, abort_early=True,
                                                      initial_const=0.001, clip_min=0.0, clip_max=1.)

    adversaries["EAD"] = attacks.ElasticNetL1Attack(Classifier, num_classes,
                                                    confidence=0,
                                                    targeted=False, learning_rate=0.01,
                                                    binary_search_steps=9, max_iterations=60,
                                                    abort_early=True, initial_const=1e-3,
                                                    clip_min=0., clip_max=1., beta=1e-3, decision_rule='EN')

    adversaries["PGDL1"] = attacks.SparseL1PGDAttack(Classifier, eps=10., nb_iter=40, eps_iter=2*10/40,
                                                     rand_init=False, clip_min=0.0, clip_max=1.0,
                                                     sparsity=0.05, eot_samples=eot_samples)

    adversaries["PGDLinf"] = attacks.LinfPGDAttack(Classifier, eps=0.031, nb_iter=40, eps_iter=2*0.031/40,
                                                   rand_init=True, clip_min=0.0, clip_max=1.0, eot_samples=eot_samples)

    adversaries["PGDL2"] = attacks.L2PGDAttack(Classifier, eps=2., nb_iter=40, eps_iter=2*2./40,
                                               rand_init=True, clip_min=0.0, clip_max=1.0, eot_samples=eot_samples)

    adversaries["FGSM"] = attacks.GradientSignAttack(Classifier, loss_fn=None, eps=0.05, clip_min=0.,
                                                     clip_max=1., targeted=False, eot_samples=eot_samples)

    current_num_input = 0
    running_acc = 0

    start_time_epoch = time.time()
    for i, data in enumerate(test_loader, 0):

        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        if (i == 0) and save_image:
            torchvision.utils.save_image(inputs, "images_nat.jpg", nrow=8, padding=2,
                                         normalize=False, range=None, scale_each=False, pad_value=0)
        if attack is not None:
            inputs = adversaries[attack].perturb(inputs, labels)

        if (i == 0) and save_image:
            torchvision.utils.save_image(inputs, "images_adv.jpg", nrow=8, padding=2,
                                         normalize=False, range=None, scale_each=False, pad_value=0)
        with torch.no_grad():
            if noise is None:
                outputs = Classifier(inputs)
                _, predicted = torch.max(outputs.data, 1)

            else:
                outputs = torch.FloatTensor(
                    labels.shape[0], num_classes).cuda()
                outputs.zero_()
                for _ in range(batch_prediction):
                    outputs += Classifier(inputs)
                _, predicted = torch.max(outputs.data, 1)

        # print statistics
        running_acc += predicted.eq(labels.data).cpu().sum().numpy()
        curr_batch_size = inputs.size(0)
        current_num_input += curr_batch_size
        print("[", (i+1)*batch_size, "/", num_images, "] running_acc=", running_acc/current_num_input)

    accuracy = (running_acc/num_images)
    print(accuracy)
    with open(result_file, 'a') as f:
        f.write('{} {} {} {} {} {} {}\n'.format(
            epoch, dataset, noise, batch_prediction, attack, eot_samples, accuracy))


if __name__ == "__main__":
    main(save_image=True)
