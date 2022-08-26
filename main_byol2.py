""" Reimplementation of BYOL for CIFAR-100"""

from tqdm import tqdm
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as datasets
import torchvision.transforms as T
import torchvision.models
import torchvision.transforms.functional as TF
import random
from torch.utils.data import DataLoader


import fastargs
from fastargs import get_current_config, Param, Section
from fastargs.decorators import param



# =========================================
# =           Data Loader stuff           =
# =========================================
class Augment:
    """
    A stochastic data augmentation module
    Transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """
    def __init__(self, img_size, s=1):
        color_jitter = T.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        blur = T.GaussianBlur((3, 3), (0.1, 2.0))

        self.train_transform = T.Compose([
            T.ToTensor(),
            T.RandomResizedCrop(size=img_size),
            T.RandomHorizontalFlip(p=0.5),  # with 0.5 probability
            T.RandomApply([color_jitter], p=0.8),
            T.RandomApply([blur], p=0.5),
            T.RandomGrayscale(p=0.2),
            # imagenet stats
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x),


def get_cifar100_dataloader(batch_size, data_dir=None, train=True, transform=Augment(32)):
    dataset = CIFAR100(root=data_dir, train=train, transform=transform, download=True)
    return DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4, drop_last=True)



# ==============================================
# =           Model definition stuff           =
# ==============================================

def make_resnet():
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)



class MLP(nn.Module):
    def __init__(self, dim, embedding_size=256, hidden_size=2048, batch_norm_mlp=False):
        super().__init__()
        norm = nn.BatchNorm1d(hidden_size) if batch_norm_mlp else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            norm,
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, embedding_size)
        )

    def forward(self, x):
        return self.net(x)



class AddProjHead(nn.Module):
    def __init__(self, model, in_features, layer_name, hidden_size=4096,
                 embedding_size=256, batch_norm_mlp=True):
        super(AddProjHead, self).__init__()
        self.backbone = model
        # remove last layer 'fc' or 'classifier'
        setattr(self.backbone, layer_name, nn.Identity())
        self.backbone.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = torch.nn.Identity()
        # add mlp projection head
        self.projection = MLP(in_features, embedding_size, hidden_size=hidden_size, batch_norm_mlp=batch_norm_mlp)

    def forward(self, x, return_embedding=False):
        embedding = self.backbone(x)
        if return_embedding:
            return embedding
        return self.projection(embedding)

def loss_fn(x, y):
    # L2 normalization
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)



class EMA():
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.alpha + (1 - self.alpha) * new



class BYOL(nn.Module):
    def __init__(
            self,
            net,
            batch_norm_mlp=True,
            layer_name='fc',
            in_features=512,
            projection_size=256,
            projection_hidden_size=2048,
            moving_average_decay=0.99,
            use_momentum=True):
        """
        Args:
            net: model to be trained
            batch_norm_mlp: whether to use batchnorm1d in the mlp predictor and projector
            in_features: the number features that are produced by the backbone net i.e. resnet
            projection_size: the size of the output vector of the two identical MLPs
            projection_hidden_size: the size of the hidden vector of the two identical MLPs
            augment_fn2: apply different augmentation the second view
            moving_average_decay: t hyperparameter to control the influence in the target network weight update
            use_momentum: whether to update the target network
        """
        super().__init__()
        self.net = net
        self.student_model = AddProjHead(model=net, in_features=in_features,
                                         layer_name=layer_name,
                                         embedding_size=projection_size,
                                         hidden_size=projection_hidden_size,
                                         batch_norm_mlp=batch_norm_mlp)
        self.use_momentum = use_momentum
        self.teacher_model = self._get_teacher()
        self.target_ema_updater = EMA(moving_average_decay)
        self.student_predictor = MLP(projection_size, projection_size, projection_hidden_size)

    @torch.no_grad()
    def _get_teacher(self):
        return copy.deepcopy(self.student_model)

    @torch.no_grad()
    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum ' \
                                  'for the target encoder '
        assert self.teacher_model is not None, 'target encoder has not been created yet'

        for student_params, teacher_params in zip(self.student_model.parameters(), self.teacher_model.parameters()):
          old_weight, up_weight = teacher_params.data, student_params.data
          teacher_params.data = self.target_ema_updater.update_average(old_weight, up_weight)

    def forward(
            self,
            image_one, image_two=None,
            return_embedding=False):
        if return_embedding or (image_two is None):
            return self.student_model(image_one, return_embedding=True)

        # student projections: backbone + MLP projection
        student_proj_one = self.student_model(image_one)
        student_proj_two = self.student_model(image_two)

        # additional student's MLP head called predictor
        student_pred_one = self.student_predictor(student_proj_one)
        student_pred_two = self.student_predictor(student_proj_two)

        with torch.no_grad():
            # teacher processes the images and makes projections: backbone + MLP
            teacher_proj_one = self.teacher_model(image_one).detach_()
            teacher_proj_two = self.teacher_model(image_two).detach_()

        loss_one = loss_fn(student_pred_one, teacher_proj_one)
        loss_two = loss_fn(student_pred_two, teacher_proj_two)

        return (loss_one + loss_two).mean()


# =============================================
# =           Optimization methods            =
# =============================================
from torch.optim.optimizer import Optimizer, required

# almost copy paste from https://github.com/noahgolmant/pytorch-lars/blob/master/lars.py
class LARS(Optimizer):
    r"""Implements LARS (Layer-wise Adaptive Rate Scaling).
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        eta (float, optional): LARS coefficient as used in the paper (default: 1e-3)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        epsilon (float, optional): epsilon to prevent zero division (default: 0)
    Example:
        >>> optimizer = torch.optim.LARS(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(self, params, lr=required, momentum=0, eta=1e-3, dampening=0,
                 weight_decay=0, nesterov=False, epsilon=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, eta=eta, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, epsilon=epsilon)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(LARS, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LARS, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eta = group['eta']
            dampening = group['dampening']
            nesterov = group['nesterov']
            epsilon = group['epsilon']

            for p in group['params']:
                if p.grad is None:
                    continue
                w_norm = torch.norm(p.data)
                g_norm = torch.norm(p.grad.data)
                if w_norm * g_norm > 0:
                    local_lr = eta * w_norm / (g_norm +
                        weight_decay * w_norm + epsilon)
                else:
                    local_lr = 1
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(d_p, alpha=-local_lr * group['lr'])

        return loss




# ======================================================
# =           Training Stuff                           =
# ======================================================

def train_one_epoch(model, dataloader, optimizer, device):
    model = model.train()
    total_loss = 0.0
    num_batches = len(dataLoader)

    for data in train_dataloader:
        optimizer.zero_grad()
        (view1, view2), _  = data
        loss = model(view1.to(device), view2.to(device))
        loss.backward()
        optimizer.step()
        model.update_moving_average()
        total_loss += loss.item()

    return total_loss / num_batches



# =========================================================
# =           Main Training Block                         =
# =========================================================

Section('main', 'main training loop stuff').params(
    device=Param(int, 'which cuda device to use', default=0),
    batch_size=Param(int, 'batch size', default=256),
    data_dir=Param(str, 'location of cifar100 dataset', default='~/datasets'),
    lr=Param(float, 'learning rate for adam', default=3e-4),
    wd=Param(float, 'weight decay', default=1e-6),
    epochs=Param(int, 'number of epochs to run', default=100),
    save_path=Param(str, 'where to save checkpoint when done',
                    default='checkpoints/byol/')
    )


@param('main.device')
@param('main.batch_size')
@param('main.data_dir')
@param('main.lr')
@param('main.wd')
@param('main.save_path')
def main(device=None, batch_size=None, data_dir=None, lr=None, wd=None,
         save_path=None)
    # Step 1: create model
    model = make_resnet()
    model = BYOL(model, in_features=512, batch_norm_mlp=True)
    model = model.to(device)
    model = model.train()

    # Step 2: create dataset
    dataloader = get_cifar100_dataloader(batch_size, data_dir)

    # Step 3: Build optimizers
    optimizer = optim.Adam(model.parameters(), lr=lr, wd=wd)


    # Step 4: train and save
    for epoch in tqdm(range(epochs)):
        train_one_epoch(model, dataloader, optimizer, device)

    torch.save(model.state_dict(), os.path.join(save_path, '%04d.pth' % epochs))