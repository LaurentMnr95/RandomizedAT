import argparse
import os


class options_sheduler:
    def __init__(self):
        self.datadir ='/private/home/laurentmeunier/datasets'
        self.dataset = 'CIFAR10' #only cifar 10 and mnist for now
        self.input_nc = 3 #3 for cifar 1 for mnist, number of input channels
        self.num_classes = 10
        
        self.net_type = 'wide-resnet'

        self.sparsify = -1
        self.depth = 28
        self.widen_factor = 10
        self.dropout = 0.3


        self.epochs = 200
        self.start_epoch = 0
        self.batch_size = 128



        # selfimizer parameters for SGD
        self.lr = 0.1
        self.momentum = 0.9
        self.weight_decay = 5e-4

        #lr parameters TODO: config other steps
        self.lr_policy = 'multistep'

        if self.lr_policy == 'step':
            self.stepsize_lr = 100 
            self.step_gamma = 0.1

        if self.lr_policy == 'multistep':
            self.milestones = [60,120,160]
            self.step_gamma = 0.2

        # save frequency
        self.save_frequency = 10
        self.save_path ='/private/home/laurentmeunier/models/sparse/'
        self.resume = False
        self.resume_name = "BEST.t7"

        self.visdom = False
        self.env_visdom = "main"

        self.val_test = True

        # self.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
        #                     help='evaluate model on validation set')
        # self.add_argument('--pretrained', dest='pretrained', action='store_true',
        #                     help='use pre-trained model')
        # self.add_argument('--world-size', default=-1, type=int,
        #                     help='number of nodes for distributed training')
        # self.add_argument('--rank', default=-1, type=int,
        #                     help='node rank for distributed training')
        # self.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
        #                     help='url used to set up distributed training')
        # self.add_argument('--dist-backend', default='nccl', type=str,
        #                     help='distributed backend')
        # self.add_argument('--seed', default=None, type=int,
        #                     help='seed for initializing training. ')
        self.gpu = True
        # self.add_argument('--multiprocessing-distributed', action='store_true',
        #                     help='Use multi-processing distributed training to launch '
        #                          'N processes per node, which has N GPUs. This is the '
        #                          'fastest way to use PyTorch for either single node or '
        #                          'multi node data parallel training')

        # args = self.parse_args()
        # return args


class options_train:
    def __init__(self):
        self.datadir ='/private/home/laurentmeunier/datasets'
        self.dataset = 'CIFAR10' #only cifar 10 and mnist for now
        self.input_nc = 3 #3 for cifar 1 for mnist, number of input channels
        self.num_classes = 10
        
        self.net_type = 'wide-resnet'

        self.sparsify = -1
        self.depth = 28
        self.widen_factor = 10
        self.dropout = 0.3


        self.epochs = 200
        self.start_epoch = 0
        self.batch_size = 128



        # selfimizer parameters for SGD
        self.lr = 0.1
        self.momentum = 0.9
        self.weight_decay = 5e-4

        #lr parameters TODO: config other steps
        self.lr_policy = 'multistep'

        if self.lr_policy == 'step':
            self.stepsize_lr = 100 
            self.step_gamma = 0.1

        if self.lr_policy == 'multistep':
            self.milestones = [60,120,160]
            self.step_gamma = 0.2

        # save frequency
        self.save_frequency = 10
        self.save_path ='/private/home/laurentmeunier/models/sparse/'
        self.resume = False
        self.resume_name = "BEST.t7"

        self.visdom = False
        self.env_visdom = "main"

        self.val_test = True

        # self.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
        #                     help='evaluate model on validation set')
        # self.add_argument('--pretrained', dest='pretrained', action='store_true',
        #                     help='use pre-trained model')
        # self.add_argument('--world-size', default=-1, type=int,
        #                     help='number of nodes for distributed training')
        # self.add_argument('--rank', default=-1, type=int,
        #                     help='node rank for distributed training')
        # self.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
        #                     help='url used to set up distributed training')
        # self.add_argument('--dist-backend', default='nccl', type=str,
        #                     help='distributed backend')
        # self.add_argument('--seed', default=None, type=int,
        #                     help='seed for initializing training. ')
        self.gpu = True
        # self.add_argument('--multiprocessing-distributed', action='store_true',
        #                     help='Use multi-processing distributed training to launch '
        #                          'N processes per node, which has N GPUs. This is the '
        #                          'fastest way to use PyTorch for either single node or '
        #                          'multi node data parallel training')

        # args = self.parse_args()
        # return args

class options_test:
    def __init__(self):
        self.datadir ='/private/home/laurentmeunier/datasets'
        self.dataset = 'CIFAR10' #only cifar 10 and mnist for now
        self.input_nc = 3 #3 for cifar 1 for mnist, number of input channels
        self.num_classes = 10

        self.batch_size = 1 # if attacks, need to be =1



        self.sparsify = -1
        self.path_to_model = "/private/home/laurentmeunier/models/wide-resnet-28x10_sparse_-1_CIFAR10/BEST.t7"
        # self.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
        #                     help='evaluate model on validation set')
        # self.add_argument('--pretrained', dest='pretrained', action='store_true',
        #                     help='use pre-trained model')
        # self.add_argument('--world-size', default=-1, type=int,
        #                     help='number of nodes for distributed training')
        # self.add_argument('--rank', default=-1, type=int,
        #                     help='node rank for distributed training')
        # self.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
        #                     help='url used to set up distributed training')
        # self.add_argument('--dist-backend', default='nccl', type=str,
        #                     help='distributed backend')
        # self.add_argument('--seed', default=None, type=int,
        #                     help='seed for initializing training. ')
        self.gpu = True
        # self.add_argument('--multiprocessing-distributed', action='store_true',
        #                     help='Use multi-processing distributed training to launch '
        #                          'N processes per node, which has N GPUs. This is the '
        #                          'fastest way to use PyTorch for either single node or '
        #                          'multi node data parallel training')

        # args = self.parse_args()
        # return args
