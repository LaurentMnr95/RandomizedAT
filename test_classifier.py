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
import submitit
# TODO: add L1 attacks + eot CW
# define options


def main(init_file,
         path_model="test.t7",
         result_file='blabla.txt',
         dataset="CIFAR10", num_classes=10,
         batch_size=256,
         attack="PGDL2", eot_samples=1,  # 80
         noise=None,
         batch_prediction=1, sigma=0.25, save_image=False):

    torch.manual_seed(1234)

    job_env = submitit.JobEnvironment()
    print(job_env)
    torch.cuda.set_device(job_env.local_rank)

    torch.distributed.init_process_group(
        backend="nccl",
        init_method=init_file,
        world_size=job_env.num_tasks,
        rank=job_env.global_rank,
    )
    if noise is None:
        batch_prediction = None
        sigma = None
    # Load inputs
    test_loader = load_data(dataset=dataset, datadir="datasets",
                            batch_size_per_gpu=int(batch_size/job_env.num_tasks),
                            job_env=job_env, train_mode=False)
    num_images = len(test_loader.dataset)

    # Classifier  definition
    # torch.nn.Module.dump_patches = True

    # model_load = torch.load(path_model)
    # Classifier = model_load["net"]
    ckpt = torch.load(path_model)
    epoch = ckpt["epoch"]

    model, _ = getNetwork(net_type="wide-resnet", depth=28, widen_factor=10, dropout=0.3, num_classes=num_classes)

    model.load_state_dict(ckpt["model_state_dict"])
    Classifier = RandModel(model, noise=noise, sigma=sigma)
    Classifier.cuda(job_env.local_rank)

    cudnn.benchmark = True
    Classifier = torch.nn.parallel.DistributedDataParallel(
        Classifier, device_ids=[job_env.local_rank], output_device=job_env.local_rank)

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

    adversaries["PGDL1"] = attacks.SparseL1PGDAttack(Classifier, eps=10., nb_iter=40, eps_iter=2*10./40,
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

    if attack is not None:
        norms_l1 = []
        norms_l2 = []
        norms_linf = []

    for i, data in enumerate(test_loader, 0):
        if i > 0 and save_image:
            break
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.cuda(job_env.local_rank), labels.cuda(job_env.local_rank)
        if (i == 0) and save_image and (job_env.global_rank == 0):
            torchvision.utils.save_image(inputs, "images_nat.jpg", nrow=8, padding=2,
                                         normalize=False, range=None, scale_each=False, pad_value=0)
        if attack is not None:
            inputs_adv = adversaries[attack].perturb(inputs, labels)

            norms_l1_batch = get_lp_norm(inputs_adv-inputs, p=1)
            norms_l2_batch = get_lp_norm(inputs_adv-inputs, p=2)
            norms_linf_batch = get_lp_norm(inputs_adv-inputs, p=np.inf)

            norms_l1.append(norms_l1_batch)
            norms_l2.append(norms_l2_batch)
            norms_linf.append(norms_linf_batch)

            inputs = inputs_adv
            if (i == 0) and save_image and (job_env.global_rank == 0):
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

    running_acc = torch.Tensor([running_acc]).cuda(job_env.local_rank)
    torch.distributed.all_reduce(running_acc,
                                 op=torch.distributed.ReduceOp.SUM)

    accuracy = (running_acc/num_images).cpu().sum().numpy()
    print(accuracy)
    if attack is not None:
        norms_l1 = torch.cat(norms_l1).view(-1)
        norms_l2 = torch.cat(norms_l2).view(-1)
        norms_linf = torch.cat(norms_linf).view(-1)

        norms_l1_gathered = all_gather(norms_l1)
        norms_l2_gathered = all_gather(norms_l2)
        norms_linf_gathered = all_gather(norms_linf)

        norms_l1_gathered = torch.cat(norms_l1_gathered).view(-1).detach().cpu().numpy()
        norms_l2_gathered = torch.cat(norms_l2_gathered).view(-1).detach().cpu().numpy()
        norms_linf_gathered = torch.cat(norms_linf_gathered).view(-1).detach().cpu().numpy()
    if job_env.global_rank == 0:
        if attack is not None:
            np.save(result_file+"_"+attack+"_l1norm", norms_l1_gathered)
            np.save(result_file+"_"+attack+"_l2norm", norms_l2_gathered)
            np.save(result_file+"_"+attack+"_linfnorm", norms_linf_gathered)
        with open(result_file+".txt", 'a') as f:
            f.write('{} {} {} {} {} {} {}\n'.format(
                epoch, dataset, noise, batch_prediction, attack, eot_samples, accuracy))

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    print(job_env.local_rank, job_env.global_rank)
    return job_env.local_rank, job_env.global_rank


if __name__ == "__main__":
    main(save_image=True)
