import fastargs
from fastargs import get_current_config, Param, Section
from fastargs.decorators import param

import argparse
import math
import shutil
import time
import numpy as np


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torch.optim



from src.utils import (
    bool_flag,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
    init_distributed_mode,
)
from src.multicropdataset import MultiCropDataset



# ================================================
# =           Setting Hyperparameters            =
# ================================================

Section('data', 'Cifar100-Related Hyperparams').params(
    data_path=Param(str, 'Location of CIFAR100 dataset',
                    default='/home/mgj528/datasets'),
    number_crops=Param(Anything, 'How many crops to perform: e.g. [2,6]',
                       default=[2]),
    size_crops=Param(Anything, 'Crops resolutions', default=[32]),
    workers=Param(int, 'number of dataloader workers', default=10),
    )

Section('model', "Swav specific params").params(

    feat_dim=Param(int, 'feature dimension', default=128),
    nmb_prototypes=Param(int, 'number of prototypes', default=3000),
    arch=Param(str, 'Which architecture to use', default='resnet50'),
    hidden_mlp=Param(int, 'dimension in hidden layer in projection head', default=2048),

    )

Section('training', "optimizer paramaters").params(
    crops_for_assign=Param(Anything, 'List of cropsids used for computing assignments', default=[0, 1]),
    temperature=Param(float, 'temp for training loss', default=0.1),
    epsilon=Param(float, 'reg param for Sinkhorn-Knopp', default=0.05),
    sinkhorn_iterations=Param(int, 'number of SK iters', default=3),
    epochs=Param(int, 'number of epochs', default=100),
    batch_size=Param(int, 'batch size per gpu', default=64),
    base_lr=Param(float, 'base learning rate', default=4.8),
    final_lr=Param(float, 'final learning rate', default=0.0),
    freeze_prototypes_niters=Param(int, 'freeze prototypes after this many iters', default=313),
    wd=Param(float, 'weight decay', default=1e-6),
    warmup_epochs=Param(int, 'number of warmup epochs', default=10),
    start_warmup=Param(float, 'initial warmup learning rate', default=0),
    checkpoint_freq=Param(int, 'how often to checkpoint', default=25),
    use_fp16=Param(bool, 'whether to use mixed precision in training', default=True),
    dump_path=Param(str, 'experiment dump path for checkpoint + log', default='.'),
    queue_length=Param(int, 'length of queue (0 for no Q)', default=0),
    epoch_queue_starts=Param(int, 'from this epoch we start using queue', default=15),
    )


Section('training', 'Hyperparameters').params(
    lr=Param(float, 'The learning rate to use', default=0.1),
    lr_peak_epoch=Param(int, 'Epoch at which LR peaks', default=20),
    epochs=Param(int, 'Number of epochs to run for', default=50),
    batch_size=Param(int, 'Batch size', default=512),
    momentum=Param(float, 'Momentum for SGD', default=0.9),
    weight_decay=Param(float, 'l2 weight decay', default=5e-4),
    label_smoothing=Param(float, 'Value of label smoothing', default=0.1),
    num_workers=Param(int, 'The number of workers', default=4),
    lr_tta=Param(bool, 'Test time augmentation by averaging with horizontally flipped version', default=True),
    gpu=Param(int, 'Which GPU to use', default=0),
    round_size=Param(int, 'How many epochs to train between evaluations', default=2),
    num_rounds=Param(int, 'How many rounds to run', default=10)

)



# ================================================================
# =           Setup blocks                                       =
# ================================================================

@torch.no_grad()
def distributed_sinkhorn(out):
    # COMMENTING OUT DISTRIBUTED-NESS
    Q = torch.exp(out / args.epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] * args.world_size # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    #dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(args.sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        #dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()


@param('model.arch')
@param('model.hidden_mlp')
@param('model.feat_dim')
@param('model.nmb_prototypes')
def create_model(arch=None, hidden_mlp=None, feat_dim=None, nmb_prototypes=None):

    model = resnet_models.__dict__[arch](
        normalize=True,
        hidden_mlp=hidden_mlp,
        output_dim=feat_dim,
        nmb_prototypes=number_prototypes,
        cifar_mode=True)

    return model



@param('data.data_path')
@param('data.size_crops')
@param('data.nmb_crops')
@param('data.batch_size')
@param('data.workers')
def create_dataloaders(data_path=None, size_crops=None, nmb_crops=None,
                       batch_size=None, workers=None):
    train_dataset = MultiCropDataset(data_path, size_crops, nmb_crops)

    loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                         num_workers=workers, pin_memory=True, drop_last=True)
    return loader



def train_single_epoch(model, loader, optimizer, scheduler, epoch,
                       ):


    if queue_length > 0 and epoch_queue_starts