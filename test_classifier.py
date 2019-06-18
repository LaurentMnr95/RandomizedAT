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
from torch.distributions import normal,laplace

##TODO: add EAD, FGM L1
# define options
def main(path_model="/private/home/laurentmeunier/RAT/models/CIFAR10/models_RT/Normal/epoch_100.t7",
            result_file='blabla.txt',
            dataset="CIFAR10",num_classes = 10,
            batch_size=256,
            attack="No", batch_eot=1,
            noise="Normal",
            batch_prediction=3, sigma=0.25):


    # Load inputs
    test_loader = load_data(dataset=dataset,datadir="datasets", batch_size=batch_size,train_mode=False)
    num_images = len(test_loader.dataset)


    # Classifier  definition
    model_load = torch.load(path_model)
    Classifier = model_load["net"]
    epoch = model_load ["epoch"]
    Classifier.cuda()
    Classifier = torch.nn.DataParallel(Classifier,device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark =True
    print("Classifier intialized")
    print(Classifier)
    Classifier.eval()


    # if adversarial_training == "CW":
    #     attack_cw = attacks.CarliniWagnerL2Attack(Classifier, num_classes,
    #             learning_rate=0.01, binary_search_steps=9, max_iterations=15, abort_early=True,
    #              initial_const=0.001, clip_min=0.0, clip_max=1.)
    
    # elif adversarial_training == "PGDinf":
    #     print("-----------PGD inf training----------")
    #     attack_linf = attacks.LinfPGDAttack(Classifier,eps=0.031, nb_iter=10, eps_iter=0.031/10, 
    #          rand_init=True, clip_min=0.0, clip_max=1.0)

    # elif adversarial_training == "mixPGDmax" or adversarial_training == "mixPGDsum":
    #     print("-----------mixPGD  training----------")
    #     attack_linf = attacks.LinfPGDAttack(Classifier,eps=0.031, nb_iter=10, eps_iter=0.031/10, 
    #          rand_init=True, clip_min=0.0, clip_max=1.0) 
    #     attack_l2 = attacks.L2PGDAttack(Classifier, eps=5., nb_iter=10, eps_iter=2*5./10, 
    #         rand_init=True, clip_min=0.0, clip_max=1.0)
    


    current_num_input = 0
    running_acc= 0

    start_time_epoch = time.time()
    for i, data in enumerate(test_loader, 0):

        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()


        if attack == "CW":
            inputs = attack_cw.perturb(inputs,labels)

        elif attack == "PGDinf":
            inputs = attack_linf.perturb(inputs,labels)



        with torch.no_grad():
            if noise is None:
                outputs = Classifier(inputs)  
                _, predicted = torch.max(outputs.data, 1)

            else:
                if noise == "Normal":
                    noise = normal.Normal(0,sigma)
                if noise == "Laplace":
                    noise = laplace.Laplace(0,sigma/np.sqrt(2))

                outputs = 0
                for _ in range(batch_prediction):
                    inputs_rand = inputs+noise.sample(inputs.shape).cuda()
                    outputs += Classifier(inputs_rand)
                    _, predicted = torch.max(outputs.data, 1)


        # print statistics
        running_acc += predicted.eq(labels.data).cpu().sum().numpy()
        curr_batch_size = inputs.size(0)

        if i % 20 == 19:
            # print every 20 mini-batches
            print("[",i*batch_size,"/",num_images,"]")

    accuracy = (running_acc/num_images)
    print(accuracy)
    with open(result_file, 'a') as f:
        f.write('{} {} {} {} {}\n'.format(epoch, batch_prediction, attack, batch_eot, accuracy))
    
        


if __name__ == "__main__":
    main()