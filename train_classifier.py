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
from torch.distributions import normal

# define options
def main(dataset="CIFAR10",batch_size=128,resume_epoch=0,
            adversarial_training=None,
            sigma_gauss=0.05):

    epochs=200
    save_path = "models"
    save_frequency = 20
    num_classes = 10
    # Load inputs
    train_loader = load_data(dataset=dataset,datadir="datasets", batch_size=batch_size,train_mode=True)
    num_images = len(train_loader.dataset)


    # Classifier  definition
    Classifier, modelname = getNetwork(net_type="wide-resnet",depth=28,widen_factor=10,dropout=0.3,num_classes=num_classes)
    Classifier.apply(conv_init)
    Classifier.cuda()
    Classifier = torch.nn.DataParallel(Classifier,device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark =True
    print("Classifier intialized")
    print(Classifier)
    Classifier.train()



    # optimizer and criterion
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(Classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = get_scheduler(optimizer, policy="multistep",milestones=[60,120,160],gamma=0.2)

    # resume learning
    if resume_epoch>0:
        path_to_load = os.path.join(save_path,filename+'_'+dataset,"epoch_"+str(resume_epoch)+'.t7')

        if os.path.isfile(path_to_load):
            print("=> loading checkpoint '{}'".format(path_to_load))
            checkpoint = torch.load(path_to_load)
            opt.start_epoch = checkpoint['epoch']
            Classifier = checkpoint['net']

            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(path_to_load))


    if adversarial_training == "CW":
        attack_cw = attacks.CarliniWagnerL2Attack(Classifier, num_classes,
                learning_rate=0.01, binary_search_steps=9, max_iterations=15, abort_early=True,
                 initial_const=0.001, clip_min=0.0, clip_max=1.)
    elif adversarial_training == "PGDinf":
        attack_linf = attacks.LinfPGDAttack(Classifier,
             loss_fn=None, eps=0.031, nb_iter=10, eps_iter=0.031/10, 
             rand_init=True, clip_min=0.0, clip_max=1.0)
    for epoch in range(epochs):
        current_num_input = 0

        running_loss = 0.0
        running_acc= 0

        training_loss = 0
        training_acc = 0
        start_time_epoch = time.time()
        for i, data in enumerate(train_loader, 0):

            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            if sigma_gauss>0: 
                noise = normal.Normal(0,sigma_gauss)
                inputs += noise.sample(inputs.shape).cuda()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if adversarial_training is None:   
                outputs = Classifier(inputs)
            elif adversarial_training == "CW":
                inputs = attack_cw.perturb(inputs,labels)
                outputs = Classifier(inputs)
            elif adversarial_training == "PGDinf":
                inputs = attack_linf.perturb(inputs,labels)
                outputs = Classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)

            # print statistics
            running_loss += loss.item()
            running_acc += predicted.eq(labels.data).cpu().sum().numpy()
            curr_batch_size = inputs.size(0)

            if i % 20 == 19:
                # print every 20 mini-batches
                print("Epoch :[", epoch+1,"/",epochs,
                        "] [",i*batch_size,"/",num_images,
                        "] Running loss:",running_loss/20,
                        ", Running accuracy:",running_acc/(20*curr_batch_size)," time:",time.time()-start_time_epoch)
                running_loss = 0.0
                running_acc = 0


        
        # save model
        if (epoch +1) % save_frequency == 0:
            path_to_save = os.path.join(save_path,filename+'_'+dataset)
            if not os.path.exists(path_to_save):
                os.makedirs(path_to_save)
            state={
                    'epoch': epoch + 1,
                    'net': Classifier.module,
                    }
            torch.save(state,os.path.join(path_to_save,"epoch_"+str(epoch+1)+'.t7')) 

        scheduler.step()


if __name__ == "__main__":
    main()