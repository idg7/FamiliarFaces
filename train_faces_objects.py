#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 20:40:16 2019

@author: n
"""

import argparse
import os
import random
import shutil
import time
import warnings
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import cv2
from PIL import Image, ImageOps
import fnmatch

from shutil import copyfile
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy import stats
from scipy.special import softmax
import csv

#from matplotlib import pyplot as plt


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank f.or distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# NA
parser.add_argument('--num_classes', default=1000, type=int,
                    help='Number of classes to train on.')

parser.add_argument('--num_classes_reshape', default=1000, type=int,
                    help='Number of classes to reshape to.')


parser.add_argument('--test_lfw', default=False, type=bool,
                    help='Test LFW.')
parser.add_argument('--lfw_test_list', default='/home/n/School/arcface-pytorch/lfw_test_pair.txt', type=str,
                    help='File with LFW pairs list.')
parser.add_argument('--lfw_root', default='/home/n/datasets/lfw/lfw-align-128', type=str,
                    help='Root dir for LFW pics.')
parser.add_argument('--test_batch_size', default=60, type=int,
                    help='Batch size for LFW pics.')

parser.add_argument('--test_high_low_ps', default=False, type=bool,
                    help='Test high-ps low-ps images.')
parser.add_argument('--test_flip', default=False, type=bool,
                    help='Test flipped image')

parser.add_argument('--analyze_corrs_between', default=False, type=bool,
                    help='Analyze correlations between pictures.')

parser.add_argument('--lv', default=False, type=bool,
                    help='Layers visualization.')

parser.add_argument('--l2_metric', default=False, type=bool,
                    help='Use cosine metric or L2 metric.')

parser.add_argument('--out_dir', default='/home/n/School', type=str,
                    help='directory for trained models.')

parser.add_argument('--is_inception', default=False, type=bool,
                    help='is incpetion')

parser.add_argument('--lfw_val_freq', default=100, type=int,
                    help='lfw val freq')

parser.add_argument('--representation_diffs', default=False, type=bool,
                    help='Analyze diffs between representations.')

parser.add_argument('--analyze_corrs_objects', default=False, type=bool,
                    help='Analyze corrs between objects.')

parser.add_argument('--human_feature_dists', default=False, type=bool,
                    help='compare human and alg dists.')

parser.add_argument('--compare_human_feature_alg_dists', default=False, type=bool,
                    help='compare human and alg dists.')

parser.add_argument('--feature_select_mat', default='', type=str)


parser.add_argument('--num_layers_to_train', default=0, type=int,
                    help='number of layers to train If 0 - train all.')

parser.add_argument('--divide_identities', default=False, type=bool)

parser.add_argument('--use_prob', default=False, type=bool)

parser.add_argument('--evaluate_train', default=False, type=bool)

parser.add_argument('--save_embeddings', default=False, type=bool)

parser.add_argument('--embds_dir', default='', type=str)

parser.add_argument('--identify_images', default=False, type=bool)

parser.add_argument('--identify_list', default=False, type=bool)

parser.add_argument('--recognize_list', default=False, type=bool)

parser.add_argument('--familiar_unfamiliar_model', default=False, type=bool)

parser.add_argument('--test_same_images', default=False, type=bool)

parser.add_argument('--find_good_examples', default=False, type=bool)

parser.add_argument('--analyze_wrong_res', default=False, type=bool)

parser.add_argument('--lfw_use_all_layers', default=False, type=bool)

parser.add_argument('--reshape_output_layer', default=False, type=bool)

best_acc1 = 0


# NA
# create my resnet version
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


#__all__ = ['MyResNet', 'my_resnet18', 'my_resnet34', 'my_resnet50', 'my_resnet101',
#           'my_resnet152']
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


#def conv3x3(in_planes, out_planes, stride=1):
#    """3x3 convolution with padding"""
#    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                     padding=1, bias=False)
#
#
#class BasicBlock(nn.Module):
#    expansion = 1
#
#    def __init__(self, inplanes, planes, stride=1, downsample=None):
#        super(BasicBlock, self).__init__()
#        self.conv1 = conv3x3(inplanes, planes, stride)
#        self.bn1 = nn.BatchNorm2d(planes)
#        self.relu = nn.ReLU(inplace=True)
#        self.conv2 = conv3x3(planes, planes)
#        self.bn2 = nn.BatchNorm2d(planes)
#        self.downsample = downsample
#        self.stride = stride
#
#    def forward(self, x):
#        residual = x
#
#        out = self.conv1(x)
#        out = self.bn1(out)
#        out = self.relu(out)
#
#        out = self.conv2(out)
#        out = self.bn2(out)
#
#        if self.downsample is not None:
#            residual = self.downsample(x)
#
#        out += residual
#        out = self.relu(out)
#
#        return out
#
#
#class Bottleneck(nn.Module):
#    expansion = 4
#
#    def __init__(self, inplanes, planes, stride=1, downsample=None):
#        super(Bottleneck, self).__init__()
#        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#        self.bn1 = nn.BatchNorm2d(planes)
#        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                               padding=1, bias=False)
#        self.bn2 = nn.BatchNorm2d(planes)
#        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#        self.bn3 = nn.BatchNorm2d(planes * 4)
#        self.relu = nn.ReLU(inplace=True)
#        self.downsample = downsample
#        self.stride = stride
#
#    def forward(self, x):
#        residual = x
#
#        out = self.conv1(x)
#        out = self.bn1(out)
#        out = self.relu(out)
#
#        out = self.conv2(out)
#        out = self.bn2(out)
#        out = self.relu(out)
#
#        out = self.conv3(out)
#        out = self.bn3(out)
#
#        if self.downsample is not None:
#            residual = self.downsample(x)
#
#        out += residual
#        out = self.relu(out)
#
#        return out
#
#
#class MyResNet(nn.Module):
#
#    def __init__(self, block, layers, num_classes=1000):
#        self.inplanes = 64
#        super(MyResNet, self).__init__()
#        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                               bias=False)
#        self.bn1 = nn.BatchNorm2d(64)
#        self.relu = nn.ReLU(inplace=True)
#        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#        self.layer1 = self._make_layer(block, 64, layers[0])
#        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#        self.avgpool = nn.AvgPool2d(7, stride=1)
#        self.fc = nn.Linear(512 * block.expansion, num_classes)
#
#        for m in self.modules():
#            if isinstance(m, nn.Conv2d):
#                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                m.weight.data.normal_(0, math.sqrt(2. / n))
#            elif isinstance(m, nn.BatchNorm2d):
#                m.weight.data.fill_(1)
#                m.bias.data.zero_()
#
#    def _make_layer(self, block, planes, blocks, stride=1):
#        downsample = None
#        if stride != 1 or self.inplanes != planes * block.expansion:
#            downsample = nn.Sequential(
#                nn.Conv2d(self.inplanes, planes * block.expansion,
#                          kernel_size=1, stride=stride, bias=False),
#                nn.BatchNorm2d(planes * block.expansion),
#            )
#
#        layers = []
#        layers.append(block(self.inplanes, planes, stride, downsample))
#        self.inplanes = planes * block.expansion
#        for i in range(1, blocks):
#            layers.append(block(self.inplanes, planes))
#
#        return nn.Sequential(*layers)
#
#    def forward(self, x):
#        
#        # collect all layer data
#        ld = []
#        ld_full = []
#        x = self.conv1(x)
#        x = self.bn1(x)
#        x = self.relu(x)
#        x = self.maxpool(x)
#
#        ld_full.append(x)
#        ld.append(x.view(x.size(0), -1))
#        x = self.layer1(x)
#        ld_full.append(x)
#        ld.append(x.view(x.size(0), -1))
#        x = self.layer2(x)
#        ld_full.append(x)
#        ld.append(x.view(x.size(0), -1))
#        x = self.layer3(x)
#        ld_full.append(x)
#        ld.append(x.view(x.size(0), -1))
#        x = self.layer4(x)
#        ld_full.append(x)
#        ld.append(x.view(x.size(0), -1))
#
#        x = self.avgpool(x)
#        # calculate feature vector
#        x = x.view(x.size(0), -1)
#        feature_vec = x
#        x = self.fc(x)
#
#        return x, feature_vec,ld,ld_full
#
#
#def my_resnet18(pretrained=False, **kwargs):
#    """Constructs a ResNet-18 model.
#
#    Args:
#        pretrained (bool): If True, returns a model pre-trained on ImageNet
#    """
#    model = MyResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
#    if pretrained:
#        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
#    return model
#
#
#def my_resnet34(pretrained=False, **kwargs):
#    """Constructs a ResNet-34 model.
#
#    Args:
#        pretrained (bool): If True, returns a model pre-trained on ImageNet
#    """
#    model = MyResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
#    if pretrained:
#        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
#    return model
#
#
#def my_resnet50(pretrained=False, **kwargs):
#    """Constructs a ResNet-50 model.
#
#    Args:
#        pretrained (bool): If True, returns a model pre-trained on ImageNet
#    """
#    model = MyResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
#    if pretrained:
#        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
#    return model
#
#
#def my_resnet101(pretrained=False, **kwargs):
#    """Constructs a ResNet-101 model.
#
#    Args:
#        pretrained (bool): If True, returns a model pre-trained on ImageNet
#    """
#    model = MyResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
#    if pretrained:
#        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
#    return model
#
#
#def my_resnet152(pretrained=False, **kwargs):
#    """Constructs a ResNet-152 model.
#
#    Args:
#        pretrained (bool): If True, returns a model pre-trained on ImageNet
#    """
#    model = MyResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
#    if pretrained:
#        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
#    return model

# My Inception

from collections import namedtuple
import torch
#import torch.nn as nn
import torch.nn.functional as F
#from .utils import load_state_dict_from_url


__all__ = ['MyInception3', 'my_inception_v3']


model_urls = {
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}

_InceptionOutputs = namedtuple('InceptionOutputs', ['logits', 'aux_logits'])

my_inception_v3_layers = ["Conv2d_1a_3x3",
                          "Conv2d_2a_3x3",
                          "Conv2d_2b_3x3",
                          "Conv2d_3b_1x1",
                          "Conv2d_4a_3x3",
                          "Mixed_5b",
                          "Mixed_5c",
                          "Mixed_5d",
                          "Mixed_6a",
                          "Mixed_6b",
                          "Mixed_6c",
                          "Mixed_6d",
                          "Mixed_6e",
                          "Mixed_7a",
                          "Mixed_7b",
                          "Mixed_7c",
                          "fc"]

def my_inception_v3(pretrained=False, progress=True, **kwargs):
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.
    .. note::
        **Important**: In contrast to the other models the inception_v3 expects tensors with a size of
        N x 3 x 299 x 299, so ensure your images are sized accordingly.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        aux_logits (bool): If True, add an auxiliary branch that can improve training.
            Default: *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: *False*
    """
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        if 'aux_logits' in kwargs:
            original_aux_logits = kwargs['aux_logits']
            kwargs['aux_logits'] = True
        else:
            original_aux_logits = True
        model = MyInception3(**kwargs)
        model.load_state_dict(model_zoo.load_url(model_urls['inception_v3_google']))
#        state_dict = load_state_dict_from_url(model_urls['inception_v3_google'],
#                                              progress=progress)
#        model.load_state_dict(state_dict)
        if not original_aux_logits:
            model.aux_logits = False
            del model.AuxLogits
        return model

    return MyInception3(**kwargs)


class MyInception3(nn.Module):

    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False):
        super(MyInception3, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        self.fc = nn.Linear(2048, num_classes)

        module_ind = 0
        for m in self.modules():
            print(module_ind)
            module_ind += 1
            print(m)
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                # X = stats.truncnorm(-2, 2, scale=stddev)
                # values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                # values = values.view(m.weight.size())
                # with torch.no_grad():
                #     m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        # N x 3 x 299 x 299
        # collect all layers data
        ld = {}
        x = self.Conv2d_1a_3x3(x)
        ld['Conv2d_1a'] = x.view(x.size(0), -1)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        ld['Conv2d_2a'] = x.view(x.size(0), -1)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        ld['Conv2d_2b'] = x.view(x.size(0), -1)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        #ld['F.max_pool2d_a'] = x.view(x.size(0), -1)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        ld['Conv2d_3b'] = x.view(x.size(0), -1)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        ld['Conv2d_4a'] = x.view(x.size(0), -1)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        #ld['F.max_pool2d_b'] = x.view(x.size(0), -1)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        ld['Mixed_5b'] = x.view(x.size(0), -1)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        ld['Mixed_5c'] = x.view(x.size(0), -1)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        ld['Mixed_5d'] = x.view(x.size(0), -1)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        ld['Mixed_6a'] = x.view(x.size(0), -1)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        ld['Mixed_6b'] = x.view(x.size(0), -1)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        ld['Mixed_6c'] = x.view(x.size(0), -1)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        ld['Mixed_6d'] = x.view(x.size(0), -1)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        ld['Mixed_6e'] = x.view(x.size(0), -1)
        # N x 768 x 17 x 17
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        ld['Mixed_7a'] = x.view(x.size(0), -1)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        ld['Mixed_7b'] = x.view(x.size(0), -1)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        ld['Mixed_7c'] = x.view(x.size(0), -1)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        #ld['F.adaptive_avg_pool2d'] = x.view(x.size(0), -1)
        # N x 2048 x 1 x 1
        x = F.dropout(x, training=self.training)
        #ld['F.dropout'] = x.view(x.size(0), -1)
        # N x 2048 x 1 x 1
        x = x.view(x.size(0), -1)
        ld['last'] = x.view(x.size(0), -1)
        last = x
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        if self.training and self.aux_logits:
            return _InceptionOutputs(x, aux)
        return x, last,ld


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # N x 768 x 17 x 17
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 768 x 1 x 1
        x = x.view(x.size(0), -1)
        # N x 768
        x = self.fc(x)
        # N x 1000
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


def is_name_in_list(name,list_of_layers):
    for layer in list_of_layers:
        if name.find(layer) != -1:
            return True
    return False


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        #model = models.__dict__[args.arch](pretrained=True)
        #model = my_resnet18(pretrained=True)
        model = my_inception_v3(pretrained=True,num_classes = args.num_classes,transform_input=False)
    else:
        print("=> creating model '{}'".format(args.arch))
        #model = models.__dict__[args.arch](num_classes = args.num_classes)
        # model = models.__dict__[args.arch]()
        #model = my_resnet18(num_classes = args.num_classes)
        model = my_inception_v3(pretrained=False,num_classes = args.num_classes,transform_input=False)
        
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    
    # use triplet loss
#    criterion = nn.TripletMarginLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint {} (epoch {}, best_acc1 {})"
                  .format(args.resume, checkpoint['epoch'],best_acc1))
        else:
            print("=> no checkpoint found at {}".format(args.resume))

    # perhaps we need to reshape the last layer
    # if(args.num_classes != args.num_classes_reshape):
    if(args.reshape_output_layer):
        print('reshaping output layer: num_classes - {}, num_classes_reshape - {}'\
              .format(args.num_classes,args.num_classes_reshape))
        # Handle the auxilary net
        #num_ftrs = model.AuxLogits.fc.in_features
        #model.AuxLogits.fc = nn.Linear(num_ftrs, args.num_classes_reshape)
        num_ftrs = model._modules['module'].AuxLogits.fc.in_features
        model._modules['module'].AuxLogits.fc = nn.Linear(num_ftrs, args.num_classes_reshape)
        # Handle the primary net
        #num_ftrs = model.fc.in_features
        #model.fc = nn.Linear(num_ftrs,args.num_classes_reshape)
        num_ftrs = model._modules['module'].fc.in_features
        model._modules['module'].fc = nn.Linear(num_ftrs,args.num_classes_reshape)
        model.cuda()

    # now determine how many layers to train
    # if num_layers_to_train = 0 - do nothing, train all layers
    # else, we freeze all but the last num_layers_to_train
    # the layers are in the list at the beginning of the file
    if(args.num_layers_to_train > 0):
        print("setting: {} layers_to_train".format(args.num_layers_to_train))
        layers_to_freeze = my_inception_v3_layers[0:-args.num_layers_to_train]
        # we go over params and names, and if the name is in layers_to_freeze
        # then set requires_grad to False
        for name,p in model.named_parameters():
            if is_name_in_list(name,layers_to_freeze):
                print("freezing '{}'".format(name))
                p.requires_grad = False
            else:
                print("NOT freezing '{}'".format(name))
    
    cudnn.benchmark = True

    if args.test_high_low_ps:
        test_high_low_ps(model,args)
        return
    
    if args.analyze_corrs_between:
        analyze_corrs_between(model,args)
        return
        
    if args.test_lfw:
        test_face_accuracy(model,args.lfw_test_list,args)
        return
    
    if args.lv:
        layer_visualization(model,args)
        return

    if args.representation_diffs:
        representation_diffs(model,args)
        return
        
    if args.analyze_corrs_objects:
        analyze_corrs_objects(model,args)
        return
    
    if args.human_feature_dists:
        human_feature_dists(model,args)
        return
    
    if args.compare_human_feature_alg_dists:
        compare_human_feature_alg_dists(model,args)
        return
    
    if args.divide_identities:
        divide_identities(model,args)
        return

    if args.save_embeddings:
        save_embeddings(model,args)
        return
    
    if args.identify_images:
        identify_images(model,args)
        return

    if args.find_good_examples:
        find_good_examples(model,args)
        return
    
    if args.test_same_images:
        test_same_images(model,args)
        return

    if args.analyze_wrong_res:
        analyze_wrong_res()
        return

    if args.recognize_list:
        recognize_list(model,args)
        return
    
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # im_size = 224
    # if inception
    im_size = 299
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(im_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.familiar_unfamiliar_model:
        familiar_unfamiliar_model(model,train_dataset,args)
        return


    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

#    val_loader = torch.utils.data.DataLoader(
#        datasets.ImageFolder(valdir, transforms.Compose([
#            transforms.Resize(256),
#            transforms.CenterCrop(224),
#            transforms.ToTensor(),
#            normalize,
#        ])),
#        batch_size=args.batch_size, shuffle=False,
#        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(im_size),
            transforms.CenterCrop(im_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    if args.evaluate_train:
        validate(train_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args.out_dir)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        if args.is_inception:
            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
            output, aux_outputs = model(input)
            loss1 = criterion(output, target)
            loss2 = criterion(aux_outputs, target)
            loss = loss1 + 0.4*loss2
        else:
            output = model(input)
            loss = criterion(output, target)

#        # compute output
#        output,_ = model(input)
#        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
        
        if (i > 0 and i % args.lfw_val_freq == 0):
            print('after {0:d} train iterations, run lfw val'.format(i))
            test_face_accuracy(model,args.lfw_test_list,args)
            # switch back .to train mode
            model.train()
        if (i > 2000):
            return
            


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output,_,_ = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
            if (i > 100):
                break

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, out_dir, filename='checkpoint.pth.tar'):
    torch.save(state,os.path.join(out_dir,filename))
    if is_best:
        torch.save(state,os.path.join(out_dir,'model_best.pth.tar'))
        # shutil.copyfile(filename, os.path.join(out_dir,'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# NA
# lfw test code
def get_lfw_list(pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    data_list = []
    for pair in pairs:
        splits = pair.split()

        if splits[0] not in data_list:
            data_list.append(splits[0])

        if splits[1] not in data_list:
            data_list.append(splits[1])
    return data_list


def load_image(img_path,flip=False):
    # NA
    # convert image name to .png
#    img_path = img_path.split('.')
#    img_path = img_path[0]+'.png'
#    image = Image.open(img_path).convert('L')
#    red,green,blue = image.split()
#    image = Image.merge("RGB",(blue,green,red))
#    image = cv2.imread(img_path, 0)

    im_size = 299
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tt = transforms.Compose([
            transforms.Resize(im_size), #256
            transforms.CenterCrop(im_size), #224
            transforms.ToTensor(),
            normalize,
        ])

    im1 = Image.open(img_path)
    # fix bug from grayscale images
    # duplicate to make 3 channels
    if im1.mode != 'RGB':
        im1 = im1.convert('RGB')
    
    if(flip):
        im1 = ImageOps.flip(im1)
    im1t = tt(im1)
    im1t = im1t.unsqueeze(0)

#
#    image = cv2.imread(img_path,cv2.IMREAD_COLOR)
##    image = cv2.resize(image,(128,128))
#    if image is None:
#        return None
##    image = image.resize((128,128))
#    image = np.dstack((image, np.fliplr(image)))
#    image = image.transpose((2, 0, 1))
#    image = image[:, np.newaxis, :, :]
#    image = image.astype(np.float32, copy=False)
#    image -= 127.5
#    image /= 127.5
    return im1t


def get_featurs(model, test_list, batch_size=10,flip=False,layer_num=None):
    images = None
    features = None
    output = None
    cnt = 0
    for i, img_path in enumerate(test_list):
        image = load_image(img_path,flip)
#        print(i)
#        print(img_path)
#        print(image.shape)
        if image is None:
            print('read {} error'.format(img_path))

        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)

        if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:
            cnt += 1
            # fix bug when images is only one
            # no need to transfer to tensor
            if images.shape[0] > 1:
                data = torch.from_numpy(images)
            else:
                data = images
            
            data = data.to(torch.device("cuda"))
            # get output and feature-vec
            # output = model(data)
            output_vec, feature_vec,ld = model(data)
            output_vec = output_vec.data.cpu().numpy()
            if layer_num is None:
                feature_vec = feature_vec.data.cpu().numpy()
            else:
                ld_list = list(ld.items())
                feature_vec = ld_list[layer_num][1].data.cpu().numpy()

#            fe_1 = output[::2]
#            fe_2 = output[1::2]
#            feature = np.hstack((fe_1, fe_2))
            # print(feature.shape)

            if features is None:
                features = feature_vec
            else:
                features = np.vstack((features, feature_vec))

            if output is None:
                output = output_vec
            else:
                output = np.vstack((output, output_vec))

            images = None

    return features, output, cnt


def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        # key = each.split('/')[1]
        fe_dict[each] = features[i]
    return fe_dict


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def l2_metric(x1,x2):
    return np.linalg.norm(x1-x2)

def l1_metric(x1,x2):
    return sum(abs(x1-x2))

def cal_accuracy(y_score, y_true,args):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        if args.l2_metric:
            y_test = (y_score <= th)
        else: # cosine similarity
            y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)


def test_performance(fe_dict, pair_list,args):

    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()

    sims = []
    labels = []
    for pair in pairs:
        splits = pair.split()
        fe_1 = fe_dict[splits[0]]
        fe_2 = fe_dict[splits[1]]
        label = int(splits[2])
        if args.l2_metric:
            sim = l2_metric(fe_1, fe_2)
        else:        
            sim = cosin_metric(fe_1, fe_2)

        sims.append(sim)
        labels.append(label)

    acc, th = cal_accuracy(sims, labels,args)

    return acc, th


def lfw_test(model, img_paths, identity_list, compair_list, args):

    s = time.time()
    
    if args.lfw_use_all_layers:
        # we want to get the accuracy according to each layer
        # so every time the features is a different layer
        num_layers = 17
        
        with open(os.path.join(args.out_dir,'lfw_per_layer.txt'),'w') as lfw_layers:
            for i in range(num_layers):
                features, output, cnt = get_featurs(model, img_paths, batch_size=args.test_batch_size,flip=args.test_flip,layer_num=i)
                fe_dict = get_feature_dict(identity_list, features)
                acc, th = test_performance(fe_dict, compair_list,args)
                if args.l2_metric:
                    metric = 'l2'
                else:
                    metric = 'cosin'
                if args.use_prob:
                    rep = 'rep-prob'
                else:
                    rep = 'rep-last'
                lfw_layers.write('%d, %.4f\n' %(i, acc))           
                print('%s verification accuracy: %.4f threshold: %.3f, metric: %s, rep: %s, layer: %d'
                      %(compair_list,acc,th,metric,rep,i))
    else:
                
        features, output, cnt = get_featurs(model, img_paths, batch_size=args.test_batch_size,flip=args.test_flip)
        if args.use_prob:
            features = output
            
        if os.path.exists(args.feature_select_mat):
            sel_mat = np.load(args.feature_select_mat)
            print(sum(sel_mat[0]))
            all_acc = np.zeros((sel_mat.shape[0]))
            for i in range(sel_mat.shape[0]):
                tmp_features = features.copy()
                sel_indices = sel_mat[i,:]
                for j in range(len(sel_indices)):
                    if sel_indices[j] == 0:
                        tmp_features[:,j] = 0
                # now we have a new feature matrix
                fe_dict = get_feature_dict(identity_list, tmp_features)
                acc, th = test_performance(fe_dict, compair_list,args)
                all_acc[i] = acc
            mean_acc = np.mean(all_acc)
            print('%s verification with ablations %s avg acc: %.4f' %(compair_list,feature_select_mat,mean_acc))
            return mean_acc
            
        else:        
            print(features.shape)
            t = time.time() - s
            print('total time is {}, average time is {}'.format(t, t / cnt))
            fe_dict = get_feature_dict(identity_list, features)
            acc, th = test_performance(fe_dict, compair_list,args)
            if args.l2_metric:
                metric = 'l2'
            else:
                metric = 'cosin'
            if args.use_prob:
                rep = 'rep-prob'
            else:
                rep = 'rep-last'
                
            print('%s verification accuracy: %.4f threshold: %.3f, metric: %s, rep: %s'
                  %(compair_list,acc,th,metric,rep))
            return acc, fe_dict

def test_face_accuracy(model,test_list,args):

    identity_list = get_lfw_list(test_list)
    img_paths = [os.path.join(args.lfw_root, each) for each in identity_list]

    model.eval()
    lfw_test(model, img_paths, identity_list, test_list, args)


def test_high_low_ps(model,args):
    # get scores for face pairs
    rootdir1 = '/home/n/PHD/experiments/celebs-post/pictures/final/aligned_160/all'
    rootdir2 = '/home/n/PHD/experiments/5features/pictures/final_pics/all_aligned_160'

    #res_file = '/home/n/School/VSS2019/train_faces_objects/resnet_faces_only_l2.txt'
    res_file = os.path.join(args.out_dir,'high_low_dists.txt')
    model.eval()
    pairs_file_list1 = ['/home/n/PHD/experiments/celebs-post/bio/celebs_same_pairs.txt',
                        '/home/n/PHD/experiments/celebs-post/bio/celebs_low_ps_pairs.txt',
                        '/home/n/PHD/experiments/celebs-post/bio/celebs_high_ps_pairs.txt',
                        '/home/n/PHD/experiments/celebs-post/bio/celebs_diff_pairs.txt']
    
#    pairs_dists_file_list1 = ['/home/n/School/VSS2019/train_faces_objects/celebs_same_resnet_faces_250_objects.txt',
#                              '/home/n/School/VSS2019/train_faces_objects/celebs_low_ps_resnet_faces_250_objects.txt',
#                              '/home/n/School/VSS2019/train_faces_objects/celebs_high_ps_resnet_faces_250_objects.txt',
#                              '/home/n/School/VSS2019/train_faces_objects/celebs_diff_resnet_faces_250_objects.txt']
    
    pairs_file_list2 = ['/home/n/PHD/experiments/5features/5features_lineup/bio/unfamiliar_same_pairs.txt',
                        '/home/n/PHD/experiments/5features/5features_lineup/bio/unfamiliar_low_ps_pairs.txt',
                        '/home/n/PHD/experiments/5features/5features_lineup/bio/unfamiliar_high_ps_pairs.txt',
                        '/home/n/PHD/experiments/5features/5features_lineup/bio/unfamiliar_diff_pairs.txt']
    
#    pairs_dists_file_list2 = ['/home/n/School/VSS2019/train_faces_objects/unfamiliar_same_resnet_faces_250_objects.txt',
#                              '/home/n/School/VSS2019/train_faces_objects/unfamiliar_low_ps_resnet_faces_250_objects.txt',
#                              '/home/n/School/VSS2019/train_faces_objects/unfamiliar_high_ps_resnet_faces_250_objects.txt',
#                              '/home/n/School/VSS2019/train_faces_objects/unfamiliar_diff_resnet_faces_250_objects.txt']

    with open(res_file,'w') as pairs_dists:
        for i in range(4):
            pairs_file = pairs_file_list1[i]
            pairs_dists.write("%s\n" %(pairs_file))
            write_dists_to_file(model,pairs_file,pairs_dists,rootdir1,args)
            pairs_file = pairs_file_list2[i]
            pairs_dists.write("%s\n" %(pairs_file))
            write_dists_to_file(model,pairs_file,pairs_dists,rootdir2,args)

def write_dists_to_file(model,pairs_file,pairs_dists,rootdir,args):
    with open(pairs_file,'r') as pairs:
        for line in pairs:
#            imgs = []
            line = line.split(" ")
            image1 = line[0]
            image2 = line[1][:-1]

            im1 = load_image(os.path.join(rootdir,image1),args.test_flip)
            im2 = load_image(os.path.join(rootdir,image2),args.test_flip)

            # calculate data for each image separately
            
#            imgs = im1
#            imgs = np.concatenate((imgs, im2), axis=0)
            
            #data1 = torch.from_numpy(im1)
            #data1 = data1.to(torch.device("cuda"))
            
            data1 = im1.to(torch.device("cuda"))

            output1, feature_vec1,ld1 = model(data1)
            # compare with "get_layers_data" method
            # mk,ld,hk = get_layers_data(model,data1,is_inception=True)
            
            output1 = output1.data.cpu().numpy()
            feature_vec1 = feature_vec1.data.cpu().numpy()

            #data2 = torch.from_numpy(im2)
            #data2 = data2.to(torch.device("cuda"))

            data2 = im2.to(torch.device("cuda"))
            output2, feature_vec2,ld2 = model(data2)
            output2 = output2.data.cpu().numpy()
            feature_vec2 = feature_vec2.data.cpu().numpy()

            pairs_dists.write("%s,%s" %(image1,image2))
            # write similarity for original images:
            if args.l2_metric:
                sim = l2_metric(im1.view(-1,1).numpy(),im2.view(-1,1).numpy())
            else:
                sim = cosin_metric(im1.view(-1,1).numpy().T,im2.view(-1,1).numpy())

            pairs_dists.write(",%.3f" %(sim))
            # and now for the layers
            ld1_list = list(ld1.items())
            ld2_list = list(ld2.items())
            for i in range(len(ld1_list)):
                ld1_np = ld1_list[i][1].data.cpu().numpy()
                ld2_np = ld2_list[i][1].data.cpu().numpy()
                if args.l2_metric:
                    sim = l2_metric(ld1_np,ld2_np)
                else:
                    sim = cosin_metric(ld1_np,ld2_np.T)

                pairs_dists.write(",%.3f" %(sim))
            # finally write similarity for last layer (feature_vec)                
#            if args.l2_metric:
#                sim = l2_metric(feature_vec1,feature_vec2)
#            else:
#                sim = cosin_metric(feature_vec1,feature_vec2.T)
#
#            pairs_dists.write(",%.3f" %(sim))

            pairs_dists.write("\n")

#from my_cnn_layer_visualization import *

def save_layer_as_image(layer,layer_name):
    # we expect a tensor of 4 dimensions
    dims = layer.shape
    num_images = dims[1]
    im_w = dims[2]
    im_h = dims[3]
    # create a blank array
    dims = { 64:(8,8), 128:(11,12), 256:(16,16), 512:(22,24) }
    (rows,cols) = dims[num_images]
    out_im = np.uint8(np.zeros((rows*im_h,cols*im_w)))
    # read each filter and copy to array
    for i in range(rows):
        for j in range(cols):
            ind = i*j + j
            tmp_im = layer[0,ind,:,:]
            min_val = np.min(tmp_im)
            if min_val < 0:
                tmp_im = tmp_im - min_val
            
            tmp_im = tmp_im / np.max(tmp_im)
            tmp_im = np.uint8(tmp_im*255)
            # put in right location in big image
            row_ind = i*im_h
            col_ind = j*im_w
            out_im[row_ind:row_ind+im_h,col_ind:col_ind+im_w] = tmp_im
    
    # convert to image and save
    layer_im = Image.fromarray(out_im)
    layer_im.save(os.path.join("/home/n/School/VSS2019/visualization/objects_only/layer_activations/",layer_name),'PNG')

from torch.optim import Adam

def layer_visualization(model,args):
    # we want to show two things:
    # 1. layer activations across the network
    # 2. visualize the weights
    # 3. generate a pattern that optimizes pixel values
    
    # load the trained model
    # read an image and get the layer data
    # plot layers
    # plot the weights
    # call "visualize_layers" for optimal image
    imname = '/home/n/PHD/experiments/celebs-post/pictures/final/aligned_160/all/h_00_affleck.png'
    im = load_image(imname)
    data = im.to(torch.device("cuda"))
    output, feature_vec,ld,ld_full = model(data)
    for i in range(len(ld_full)):
        ld_np = ld_full[i].data.cpu().numpy()
        save_layer_as_image(ld_np,str(i))
        
    # visualize the weights
    # first conv1
    wconv1 = model.module.conv1.weight
    wconv1 = wconv1.cpu().detach().numpy()
    dims = wconv1.shape
    num_images = dims[0]
    im_w = dims[2]
    im_h = dims[3]
    # create a blank array
    dims = { 64:(8,8), 128:(11,12), 256:(16,16), 512:(22,24) }
    (rows,cols) = dims[num_images]
    out_im = np.uint8(np.zeros((rows*im_h,cols*im_w,3)))
    # read each filter and copy to array
    for i in range(rows):
        for j in range(cols):
            ind = i*j + j
            for k in range(3):
                tmp_im = wconv1[ind,k,:,:]
                min_val = np.min(tmp_im)
                if min_val < 0:
                    tmp_im = tmp_im - min_val
                
                tmp_im = tmp_im / np.max(tmp_im)
                tmp_im = np.uint8(tmp_im*255)
                # put in right location in big image
                row_ind = i*im_h
                col_ind = j*im_w
                out_im[row_ind:row_ind+im_h,col_ind:col_ind+im_w,k] = tmp_im
    
    # convert to image and save
    wconv1_im = Image.fromarray(out_im.astype('uint8'), 'RGB')
    wconv1_im.save('/home/n/School/VSS2019/visualization/objects_only/layer_activations/wconv1.png','PNG')
    
    # layer1
    wlayer1 = model.module.layer1[1].conv2.weight
    wlayer1 = wlayer1.cpu().detach().numpy()
    save_layer_as_image(wlayer1,'wlayer1.png')

    # layer2
    wlayer2 = model.module.layer2[1].conv2.weight
    wlayer2 = wlayer2.cpu().detach().numpy()
    save_layer_as_image(wlayer2,'wlayer2.png')

    # layer3
    wlayer3 = model.module.layer3[1].conv2.weight
    wlayer3 = wlayer3.cpu().detach().numpy()
    save_layer_as_image(wlayer3,'wlayer3.png')

    # layer4
    wlayer4 = model.module.layer1[1].conv2.weight
    wlayer4 = wlayer4.cpu().detach().numpy()
    save_layer_as_image(wlayer4,'wlayer4.png')

    # optimize pixel values
    selected_layer = 4
    for j in range(64):
        selected_filter = j
        created_image = np.uint8(np.random.uniform(150, 180, (224, 224, 3)))
        processed_image = preprocess_image(created_image,resize_im=False)
        #data = processed_image.to(torch.device("cuda"))
        # Define optimizer for the image
        
        #model.cpu()
        
        optimizer = torch.optim.SGD([processed_image], lr=100,
                                momentum=args.momentum,
                                weight_decay=1e-6)
    
    #    optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
        for i in range(1, 31):
            optimizer.zero_grad()
    
            _,_,_,ld_full = model(processed_image)
    
            # Here, we get the specific filter from the output of the convolution operation
            # x is a tensor of shape 1x512x28x28.(For layer 17)
            # So there are 512 unique filter outputs
            # Following line selects a filter from 512 filters so self.conv_output will become
            # a tensor of shape 28x28
            selected_filter = j
            conv_output = ld_full[selected_layer][0][selected_filter]
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.cpu().data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            created_image = recreate_image(processed_image)
            # Save image
            if i % 5 == 0:
                im_path = '/home/n/School/VSS2019/visualization/objects_only/layer_activations/generated/layer_vis_l' + str(selected_layer) + \
                    '_f' + str(selected_filter) + '_iter' + str(i) + '.jpg'
                save_image(created_image, im_path)


def get_layers_data(model,input_data,is_inception=False):

    def hook_fun(m,i,o):
        tmp = torch.zeros(o.data.size())
        tmp.copy_(o.data)
        layers_data.append(tmp)
        del tmp
        
        
    if(is_inception):
        module_keys = list(model._modules['module']._modules.keys())
    else:
        module_keys = list(model._modules.keys())
    
    layers_data = []
    hooks = []
    
    for key in module_keys:
        if(is_inception):
            layer = model._modules['module']._modules.get(key)
        else:            
            layer = model._modules.get(key)
        
        h = layer.register_forward_hook(hook_fun)
        hooks.append(h)
        
    model(input_data)
    
    return module_keys,layers_data,hooks

def process_dir(model,faces_dir,num_dists,dists_fname):

    faces = sorted(fnmatch.filter(os.listdir(faces_dir),'*.jpg'))
    num_comps = int(len(faces) * np.floor(len(faces)/2))
    all_dists = np.zeros((num_comps,num_dists))
    count = 0
    for i in range(len(faces)):

        model.eval()
        model.cuda()
        
        face1 = faces[i]
        print(face1)
        for j in range(i+1,len(faces)):
            face2 = faces[j]
            print(face2)
            try:
                imgs = []
                im1 = load_image(os.path.join(faces_dir,face1))
                im2 = load_image(os.path.join(faces_dir,face2))
    
                imgs = im1
                imgs = np.concatenate((imgs, im2), axis=0)

                data = torch.from_numpy(imgs)
                data = data.to(torch.device("cuda"))

                mk,ld,hk = get_layers_data(model,data)

                layer_index = 0
                d0 = imgs[1]-imgs[0]
                all_dists[count,layer_index] = torch.dot(d0.view(-1),d0.view(-1))
                for layer_data in ld:
                    layer_index = layer_index + 1
                    diff = layer_data[1]-layer_data[0]
                    all_dists[count,layer_index] = torch.dot(diff.view(-1),diff.view(-1))

                del I_
                del mk
                del ld
                del hk
                del img_tensor1
                del img_tensor2
                del imgs
                del im1
                del im2
                
                count = count+1
    
            except IOError as e:
                print("I/O error: {0}".format(e))
                raise e
    # np.save(dists_fname,all_dists)

from natsort import natsorted

def process_dir_resnet(model,faces_dir,num_dists,dists_fname,args,exclude_list = []):

    faces = natsorted(fnmatch.filter(os.listdir(faces_dir),'*.jpg'))

#    faces = sorted(fnmatch.filter(os.listdir(faces_dir),'*.jpg'))

    num_comps = int((len(faces)*len(faces) - len(faces))/2)
    all_dists = np.zeros((num_comps,num_dists))
    count = 0
    model.eval()

    for i in range(len(faces)):
        if i not in exclude_list:
            face1 = faces[i]
            #print(face1)
            for j in range(i+1,len(faces)):
                if j not in exclude_list:
                    face2 = faces[j]
                    #print(face2)
        
                    im1 = load_image(os.path.join(faces_dir,face1))
                    im2 = load_image(os.path.join(faces_dir,face2))
                    
                    data1 = im1.to(torch.device("cuda"))
                    output1, feature_vec1,ld1 = model(data1)
                    output1 = output1.data.cpu().numpy()
                    feature_vec1 = feature_vec1.data.cpu().numpy()
        
                    data2 = im2.to(torch.device("cuda"))
                    output2, feature_vec2,ld2 = model(data2)
                    output2 = output2.data.cpu().numpy()
                    feature_vec2 = feature_vec2.data.cpu().numpy()
        
                    # write similarity for original images:
                    layer_index = 0
                    if args.l2_metric:
                        sim = l2_metric(im1.view(-1,1).numpy(),im2.view(-1,1).numpy())
                    else:
                        sim = cosin_metric(im1.view(-1,1).numpy().T,im2.view(-1,1).numpy())
                    all_dists[count,layer_index] = sim
        
                    # and now for the layers
                    ld1_list = list(ld1.items())
                    ld2_list = list(ld2.items())
                    for i in range(len(ld1_list)):
                        ld1_np = ld1_list[i][1].data.cpu().numpy()
                        ld2_np = ld2_list[i][1].data.cpu().numpy()
        #            for l in range(len(ld1)):
        #                ld1_np = ld1[l].data.cpu().numpy()
        #                ld2_np = ld2[l].data.cpu().numpy()
                        if args.l2_metric:
                            sim = l2_metric(ld1_np,ld2_np)
                        else:
                            sim = cosin_metric(ld1_np,ld2_np.T)
        
                        layer_index = layer_index + 1
                        all_dists[count,layer_index] = sim
                    # no need for feature_vec, it's included in the dictionary
        #            if args.l2_metric:
        #                sim = l2_metric(feature_vec1,feature_vec2)
        #            else:
        #                sim = cosin_metric(feature_vec1,feature_vec2.T)
#                    layer_index = layer_index + 1
#                    all_dists[count,layer_index] = sim

                    count = count+1
            
    return all_dists[0:count,:]

def process_corrs_dir(model,faces_dir,num_layers,dists_fname,args,exclude_list = []):

    # in this method we calculate the correlations between each identity and the rest
    # of the identities.
    # we measure the distances between each face and the other faces, so we get 
    # num_faces such distances per face (skipping same face and exclude_list)
    # we do this per layer
    # later, the correlations should give us num_faces numbers per layer

    faces = sorted(fnmatch.filter(os.listdir(faces_dir),'*.jpg'))
    num_faces = len(faces)

    all_dists = np.empty((num_layers,num_faces,num_faces))
    all_dists[:] = np.nan
    model.eval()

    for i in range(len(faces)):
        if i not in exclude_list:
            face1 = faces[i]
            #print(face1)
            for j in range(len(faces)):
                if (i != j) and (j not in exclude_list):
                    face2 = faces[j]
                    #print(face2)
        
                    im1 = load_image(os.path.join(faces_dir,face1))
                    im2 = load_image(os.path.join(faces_dir,face2))
                    
                    data1 = im1.to(torch.device("cuda"))
                    output1, feature_vec1,ld1 = model(data1)
                    output1 = output1.data.cpu().numpy()
                    feature_vec1 = feature_vec1.data.cpu().numpy()
        
                    data2 = im2.to(torch.device("cuda"))
                    output2, feature_vec2,ld2 = model(data2)
                    output2 = output2.data.cpu().numpy()
                    feature_vec2 = feature_vec2.data.cpu().numpy()
        
                    # write similarity for original images:
                    layer_index = 0
                    if args.l2_metric:
                        sim = l2_metric(im1.view(-1,1).numpy(),im2.view(-1,1).numpy())
                    else:
                        sim = cosin_metric(im1.view(-1,1).numpy().T,im2.view(-1,1).numpy())
                    # put value in right place:
                    # [layer_index,dist_index,face_index]
                    all_dists[layer_index,j,i] = sim
        
                    # and now for the layers
                    ld1_list = list(ld1.items())
                    ld2_list = list(ld2.items())
                    for layer_index in range(len(ld1_list)):
                        ld1_np = ld1_list[layer_index][1].data.cpu().numpy()
                        ld2_np = ld2_list[layer_index][1].data.cpu().numpy()
        #            for l in range(len(ld1)):
        #                ld1_np = ld1[l].data.cpu().numpy()
        #                ld2_np = ld2[l].data.cpu().numpy()
                        if args.l2_metric:
                            sim = l2_metric(ld1_np,ld2_np)
                        else:
                            sim = cosin_metric(ld1_np,ld2_np.T)
        
                        all_dists[layer_index+1,j,i] = sim
                    # no need for feature_vec, it's included in the dictionary
        #            if args.l2_metric:
        #                sim = l2_metric(feature_vec1,feature_vec2)
        #            else:
        #                sim = cosin_metric(feature_vec1,feature_vec2.T)
#                    layer_index = layer_index + 1
#                    all_dists[count,layer_index] = sim

            
    return all_dists


def analyze_corrs_between(model,args):

    layer_names = ['raw']
    layer_names = layer_names + my_inception_v3_layers

    print("run analyze_corrs_between")

    num_layers = 18
    # corrs_fname = '/home/n/School/VSS2019/pictures/resnet_faces_only_epohc71_corrs_l2.txt'
    corrs_fname = os.path.join(args.out_dir,'corrs_between_views.txt')

    frontal_faces = '/home/n/School/VSS2019/pictures/frontal/aligned'
    # dists_fname1 = '/home/n/School/VSS2019/pictures/resnet_faces_objects_frontal_dists.npy'
    dists_fname1 = os.path.join(args.out_dir,'frontal_dists.npy')
    # dists is a tensor of shape [num_layers,num_dists,num_faces]
    # to calculate the correlations, we do the following for each layer:
    # take column i from dists1, and correlate it with column i from dists2
    # this should give us num_faces values for each layer
    dists1 = process_corrs_dir(model,frontal_faces,num_layers,dists_fname1,args)
    
    with open(corrs_fname,'w') as corrs:
        ref_faces = '/home/n/School/VSS2019/pictures/ref/aligned/'
        dists2 = process_corrs_dir(model,ref_faces,num_layers,dists_fname1,args)
        corrs.write("frontal vs. ref\n")
        for l in range(num_layers):
            dists1_l = dists1[l,:,:]
            dists2_l = dists2[l,:,:]
            num_faces = dists1_l.shape[1]
            corrs.write("%s, " %layer_names[l])
            for i in range(num_faces):
                tmp = np.array([dists1_l[:,i],dists2_l[:,i]])
                df = pd.DataFrame(tmp.T)
                corr = df.corr()                
#                corr = np.corrcoef(dists1_l[:,i],dists2_l[:,i])
                corrs.write("%.3f, " %corr.iloc[0,1])
            corrs.write("\n")
        corrs.write("\n")

        ql_faces = '/home/n/School/VSS2019/pictures/quarter-left/cropped/bb/aligned/'
        dists2 = process_corrs_dir(model,ql_faces,num_layers,dists_fname1,args)
        corrs.write("frontal vs. quarter-left\n")
        for l in range(num_layers):
            dists1_l = dists1[l,:,:]
            dists2_l = dists2[l,:,:]
            num_faces = dists1_l.shape[1]
            corrs.write("%s, " %layer_names[l])
            for i in range(num_faces):
                tmp = np.array([dists1_l[:,i],dists2_l[:,i]])
                df = pd.DataFrame(tmp.T)
                corr = df.corr()                
#                corr = np.corrcoef(dists1_l[:,i],dists2_l[:,i])
                corrs.write("%.3f, " %corr.iloc[0,1])
            corrs.write("\n")
        corrs.write("\n")
        
#        pl_faces = '/home/n/School/VSS2019/pictures/profile-left/'
#        dists2 = process_dir_resnet(model,pl_faces,num_layers,dists_fname1,args)
#        for i in range(num_layers):
#            corr = np.corrcoef(dists1[:,i],dists2[:,i])
#            corrs.write("%.3f, " %corr[0,1])
#        corrs.write("\n")
        
#        low_faces = '/home/n/School/VSS2019/pictures/low-ps/aligned/'
#        dists2 = process_dir_resnet(model,low_faces,num_layers,dists_fname1,args)
#        for i in range(num_layers):
#            corr = np.corrcoef(dists1[:,i],dists2[:,i])
#            corrs.write("%.3f, " %corr[0,1])
#        corrs.write("\n")
#        
#        high_faces = '/home/n/School/VSS2019/pictures/high-ps/aligned/'
#        dists2 = process_dir_resnet(model,high_faces,num_layers,dists_fname1,args)
#        for i in range(num_layers):
#            corr = np.corrcoef(dists1[:,i],dists2[:,i])
#            corrs.write("%.3f, " %corr[0,1])
#        corrs.write("\n")

        frontal_faces = '/home/n/School/VSS2019/pictures/frontal/aligned/for-hl'
        # dists_fname1 = '/home/n/School/VSS2019/pictures/resnet_faces_objects_frontal_dists.npy'
        dists_fname1 = os.path.join(args.out_dir,'frontal_dists_for_hl.npy')
        dists1 = process_corrs_dir(model,frontal_faces,num_layers,dists_fname1,args)

        hl_faces = '/home/n/School/VSS2019/pictures/half-left/cropped/bb/aligned/'
        dists2 = process_corrs_dir(model,hl_faces,num_layers,dists_fname1,args)
        corrs.write("frontal vs. half-left\n")
        for l in range(num_layers):
            dists1_l = dists1[l,:,:]
            dists2_l = dists2[l,:,:]
            num_faces = dists1_l.shape[1]
            corrs.write("%s, " %layer_names[l])
            for i in range(num_faces):
                tmp = np.array([dists1_l[:,i],dists2_l[:,i]])
                df = pd.DataFrame(tmp.T)
                corr = df.corr()                
#                corr = np.corrcoef(dists1_l[:,i],dists2_l[:,i])
                corrs.write("%.3f, " %corr.iloc[0,1])
            corrs.write("\n")
        corrs.write("\n")


def analyze_corrs_objects(model,args):

    print("run analyze_corrs_objects")

    layer_names = ['raw']
    layer_names = layer_names + my_inception_v3_layers

    model.transform_input = True
    
    num_layers = 18
    corrs_fname = os.path.join(args.out_dir,'corrs_between_objects.txt')

    frontal_cars = "/home/n/School/invariance_paper/car_pictures/front"
    dists_fname1 = "null"
    dists1 = process_corrs_dir(model,frontal_cars,num_layers,dists_fname1,args)
    
    with open(corrs_fname,'w') as corrs:
        back_cars = "/home/n/School/invariance_paper/car_pictures/back"
        dists2 = process_corrs_dir(model,back_cars,num_layers,dists_fname1,args)

        corrs.write("frontal vs. back\n")
        for l in range(num_layers):
            dists1_l = dists1[l,:,:]
            dists2_l = dists2[l,:,:]
            num_objects = dists1_l.shape[1]
            corrs.write("%s, " %layer_names[l])
            for i in range(num_objects):
                tmp = np.array([dists1_l[:,i],dists2_l[:,i]])
                df = pd.DataFrame(tmp.T)
                corr = df.corr()                
#                corr = np.corrcoef(dists1_l[:,i],dists2_l[:,i])
                corrs.write("%.3f, " %corr.iloc[0,1])
            corrs.write("\n")
        corrs.write("\n")


    
        side_cars = "/home/n/School/invariance_paper/car_pictures/side"
        dists2 = process_corrs_dir(model,side_cars,num_layers,dists_fname1,args)

        corrs.write("frontal vs. side\n")
        for l in range(num_layers):
            dists1_l = dists1[l,:,:]
            dists2_l = dists2[l,:,:]
            num_objects = dists1_l.shape[1]
            corrs.write("%s, " %layer_names[l])
            for i in range(num_objects):
                tmp = np.array([dists1_l[:,i],dists2_l[:,i]])
                df = pd.DataFrame(tmp.T)
                corr = df.corr()                
#                corr = np.corrcoef(dists1_l[:,i],dists2_l[:,i])
                corrs.write("%.3f, " %corr.iloc[0,1])
            corrs.write("\n")
        corrs.write("\n")


        angle_cars = "/home/n/School/invariance_paper/car_pictures/angle"
        dists2 = process_corrs_dir(model,angle_cars,num_layers,dists_fname1,args)

        corrs.write("frontal vs. angle\n")
        for l in range(num_layers):
            dists1_l = dists1[l,:,:]
            dists2_l = dists2[l,:,:]
            num_objects = dists1_l.shape[1]
            corrs.write("%s, " %layer_names[l])
            for i in range(num_objects):
                tmp = np.array([dists1_l[:,i],dists2_l[:,i]])
                df = pd.DataFrame(tmp.T)
                corr = df.corr()                
#                corr = np.corrcoef(dists1_l[:,i],dists2_l[:,i])
                corrs.write("%.3f, " %corr.iloc[0,1])
            corrs.write("\n")
        corrs.write("\n")




# measure the differences between representations for high-low-ps
# and for views
# we measure the differences per face identity across layers
# we should get 15 differences per condition
# this method has two modes:
# diff_id = False: compare the face of the same id in different views
# diff_id = True: compare a face with a different face (different id)
def rep_diffs_dirs(model,dir1,dir2,num_dists,args,diff_id=False):

    # take two face dirs
    # compare the distances between them across all layers
    faces1 = sorted(fnmatch.filter(os.listdir(dir1),'*.jpg'))
    faces2 = sorted(fnmatch.filter(os.listdir(dir2),'*.jpg'))

    if (len(faces1) != len(faces2)):
        print('rep_diffs_dir, different number of faces')
        return
    
    all_dists = np.zeros((len(faces1),num_dists))
    model.eval()

    for i in range(len(faces1)):

        im1 = load_image(os.path.join(dir1,faces1[i]))
        print(faces1[i])
        second_face_index = i
        if diff_id:
            second_face_index = (i+1)%len(faces1)
        print(faces2[second_face_index])
        im2 = load_image(os.path.join(dir2,faces2[second_face_index]))

        data1 = im1.to(torch.device("cuda"))
        _,_,ld1 = model(data1)

        data2 = im2.to(torch.device("cuda"))
        _,_,ld2 = model(data2)

        # write similarity for original images:
        layer_index = 0
        if args.l2_metric:
            sim = l2_metric(im1.view(-1,1).numpy(),im2.view(-1,1).numpy())
        else:
            sim = cosin_metric(im1.view(-1,1).numpy().T,im2.view(-1,1).numpy())
        all_dists[i,layer_index] = sim

        # and now for the layers
        ld1_list = list(ld1.items())
        ld2_list = list(ld2.items())
        for l in range(len(ld1_list)):
            ld1_np = ld1_list[l][1].data.cpu().numpy()
            ld2_np = ld2_list[l][1].data.cpu().numpy()
            if args.l2_metric:
                sim = l2_metric(ld1_np,ld2_np)
            else:
                sim = cosin_metric(ld1_np,ld2_np.T)

            layer_index = layer_index + 1
            all_dists[i,layer_index] = sim

    
    return all_dists

def representation_diffs(model,args):

    print("run representation_diffs")
    
    num_layers = 18

    frontal_faces = '/home/n/School/VSS2019/pictures/frontal/aligned/'

    ref_faces = '/home/n/School/VSS2019/pictures/ref/aligned/'
    # this is "SAME" condition:
    ref_diffs = rep_diffs_dirs(model,frontal_faces,ref_faces,num_layers,args)
    # this is "DIFFERENT" condition:
    ref_diffs_diff_id = rep_diffs_dirs(model,frontal_faces,ref_faces,num_layers,args,diff_id=True)
    
    
    ql_faces = '/home/n/School/VSS2019/pictures/quarter-left/cropped/bb/aligned/'
    # this is "QL" condition
    ql_diffs = rep_diffs_dirs(model,frontal_faces,ql_faces,num_layers,args)
    
#    pl_faces = '/home/n/School/VSS2019/pictures/profile-left/'
#    diffs_fname = os.path.join(args.out_dir,'frontal_pl_diffs.txt')
#    pl_diffs = rep_diffs_dirs(model,frontal_faces,pl_faces,num_layers,diffs_fname,args)
    
#    low_faces = '/home/n/School/VSS2019/pictures/low-ps/'
#    diffs_fname = os.path.join(args.out_dir,'frontal_low_diffs.txt')
#    low_diffs = rep_diffs_dirs(model,frontal_faces,low_faces,num_layers,diffs_fname,args)
#    
#    high_faces = '/home/n/School/VSS2019/pictures/high-ps/'
#    diffs_fname = os.path.join(args.out_dir,'frontal_high_diffs.txt')
#    high_diffs = rep_diffs_dirs(model,frontal_faces,high_faces,num_layers,diffs_fname,args)

    frontal_faces = '/home/n/School/VSS2019/pictures/frontal/aligned/for-hl'
    hl_faces = '/home/n/School/VSS2019/pictures/half-left/cropped/bb/aligned/'
    # this is "HL" condition
    hl_diffs = rep_diffs_dirs(model,frontal_faces,hl_faces,num_layers,args)
    
    # normalize by division with max
    all_diffs = np.concatenate((ref_diffs,ref_diffs_diff_id,ql_diffs,hl_diffs),axis=0)
    all_diffs = all_diffs*1/np.max(all_diffs,axis=0)


    # calculate means and std
    diff_means = np.zeros((8,num_layers))
    diff_means[0,:] = np.mean(all_diffs[0:15,:],axis=0) # ref-same    
    diff_means[1,:] = np.mean(all_diffs[15:30,:],axis=0) # ref-diff    
    diff_means[2,:] = np.mean(all_diffs[30:45,:],axis=0) # ql  
    diff_means[3,:] = np.mean(all_diffs[45:,:],axis=0) # hl    

    diff_means[4,:] = np.std(all_diffs[0:15,:],axis=0) # ref-same    
    diff_means[5,:] = np.std(all_diffs[15:30,:],axis=0) # ref-diff    
    diff_means[6,:] = np.std(all_diffs[30:45,:],axis=0) # ql  
    diff_means[7,:] = np.std(all_diffs[45:,:],axis=0) # hl    

    with open(os.path.join(args.out_dir,'rep_diffs_mean.csv'),'w') as diffs_file:  
        writer = csv.writer(diffs_file)
        writer.writerows(diff_means)
        
    with open(os.path.join(args.out_dir,'rep_diffs_all.csv'),'w') as diffs_file_all:  
        writer = csv.writer(diffs_file_all)
        writer.writerows(all_diffs)


def human_feature_dists(model,args):

    print("run human_feature_dists")

    num_layers = 18
   
    exclude_list = [] # [56,68] # [57,69]
    # get dists between all faces using all layers
    face_dists = process_dir_resnet(model,'/home/n/School/human_vs_openface_features/dlib_aligned',num_layers,"null",args,exclude_list)
    
    # read human features into pandas dataframe
    hf = pd.read_csv('/home/n/School/human_vs_openface_features/picture_z_scores.csv',header=None)
    #hf = hf.drop(exclude_list)

    # calculate human dists based on all features:
    feats_to_use = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19] # all features
    corrs_fname = os.path.join(args.out_dir,'human_vs_alg_dists_all.txt')
    correlate_human_features_alg_dists(hf,feats_to_use,exclude_list,corrs_fname,num_layers,face_dists)

    # calculate human dists based on 5 high-ps features:
    feats_to_use = [2,9,10,12,19] # features [3,10,11,13,20]
    corrs_fname = os.path.join(args.out_dir,'human_vs_alg_dists_high_ps.txt')
    correlate_human_features_alg_dists(hf,feats_to_use,exclude_list,corrs_fname,num_layers,face_dists)

    # calculate human dists based on low-ps features:
    feats_to_use = [0,4,13,16,18] # features [1,5,14,17,19]
    corrs_fname = os.path.join(args.out_dir,'human_vs_alg_dists_low_ps.txt')
    correlate_human_features_alg_dists(hf,feats_to_use,exclude_list,corrs_fname,num_layers,face_dists)

def correlate_human_features_alg_dists(hf,feats_to_use,exclude_list,corrs_fname,num_layers,face_dists):
    
    hf = hf.iloc[:,feats_to_use]
    h_dists = []
    for i in range(len(hf.index)):
        if i not in exclude_list:
            face_i = hf.iloc[i,:]
            for j in range(i+1,len(hf.index)):
                if j not in exclude_list:
                    face_j = hf.iloc[j,:]
#                    sim = l2_metric(face_i,face_j)
                    sim = l1_metric(face_i,face_j)
                    h_dists.append(sim)

    with open(corrs_fname,'w') as corrs:
        for i in range(num_layers):
            corr = np.corrcoef(h_dists,face_dists[:,i])
            corrs.write("%.3f, " %corr[0,1])
        corrs.write("\n")
    
def compare_human_feature_alg_dists(model,args):

    print("run compare_human_feature_alg_dists")

    # compare all faces to all faces using human features
    # take the closest 10 and farthest 10
    # measure their distances using the algorithm and write the results

    exclude_list = [56,68] # [57,69]
    
    model.eval()

    # read human features into pandas dataframe
    hf = pd.read_csv('/home/n/School/human_vs_openface_features/picture_z_scores.csv',header=None)

    # all features
    feats_to_use = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    h_dists = get_human_dists(hf,feats_to_use,exclude_list)

    dists_fname = os.path.join(args.out_dir,'alg_dists_human_max_min_all.txt')
    write_alg_dists_for_human_dists(h_dists,dists_fname,args,model)

    # low-ps features
    feats_to_use = [0,4,13,16,18] # features [1,5,14,17,19]
    h_dists = get_human_dists(hf,feats_to_use,exclude_list)

    dists_fname = os.path.join(args.out_dir,'alg_dists_human_max_min_low-ps.txt')
    write_alg_dists_for_human_dists(h_dists,dists_fname,args,model)

    # high-ps features
    feats_to_use = [2,9,10,12,19] # features [3,10,11,13,20]
    h_dists = get_human_dists(hf,feats_to_use,exclude_list)

    dists_fname = os.path.join(args.out_dir,'alg_dists_human_max_min_high-ps.txt')
    write_alg_dists_for_human_dists(h_dists,dists_fname,args,model)
    
def get_human_dists(hf,feats_to_use,exclude_list):
    
    hf = hf.iloc[:,feats_to_use]
    h_dists = pd.DataFrame(columns=['f1','f2','dist'])
    for i in range(len(hf.index)):
        if i not in exclude_list:
            face_i = hf.iloc[i,:]
            for j in range(i+1,len(hf.index)):
                if j not in exclude_list:
                    face_j = hf.iloc[j,:]
                    h_dists = h_dists.append({'f1':i,'f2':j,'dist':l2_metric(face_i,face_j)},ignore_index=True)

    return h_dists

def write_alg_dists_for_human_dists(h_dists,dists_fname,args,model):

    with open(dists_fname,'w') as dists:
        # sort h_dists
        sh_dists = h_dists.sort_values(by=['dist'])
        faces_dir = '/home/n/School/human_vs_openface_features/dlib_aligned'
        faces = natsorted(fnmatch.filter(os.listdir(faces_dir),'*.jpg'))

        dists.write('10 most similar faces:\n')
        # take 10 smallest and 10 largest
        for i in range(10):
           
            face1 = int(sh_dists.iloc[i,0])
            face2 = int(sh_dists.iloc[i,1])

            # first write human distance:
            dists.write('%d,' %face1)
            dists.write('%d,' %face2)
            dists.write('%.3f, ' %sh_dists.iloc[i,2])

            im1 = load_image(os.path.join(faces_dir,faces[face1]))
            im2 = load_image(os.path.join(faces_dir,faces[face2]))
    
            data1 = im1.to(torch.device("cuda"))
            _,_,ld1 = model(data1)
    
            data2 = im2.to(torch.device("cuda"))
            _,_,ld2 = model(data2)
    
            # write similarity for original images:
            if args.l2_metric:
                sim = l2_metric(im1.view(-1,1).numpy(),im2.view(-1,1).numpy())
            else:
                sim = cosin_metric(im1.view(-1,1).numpy().T,im2.view(-1,1).numpy())
            dists.write('%.3f, ' %sim)
    
            # and now for the layers
            ld1_list = list(ld1.items())
            ld2_list = list(ld2.items())
            for l in range(len(ld1_list)):
                ld1_np = ld1_list[l][1].data.cpu().numpy()
                ld2_np = ld2_list[l][1].data.cpu().numpy()
                if args.l2_metric:
                    sim = l2_metric(ld1_np,ld2_np)
                else:
                    sim = cosin_metric(ld1_np,ld2_np.T)

                dists.write('%.3f, ' %sim)

            dists.write('\n')
    
        dists.write('10 least similar faces:\n')
        num_dists = len(h_dists.index)
        for i in range(num_dists-10,num_dists):

            face1 = int(sh_dists.iloc[i,0])
            face2 = int(sh_dists.iloc[i,1])

            # first write human distance:
            dists.write('%d,' %face1)
            dists.write('%d,' %face2)
            dists.write('%.3f, ' %sh_dists.iloc[i,2])

            im1 = load_image(os.path.join(faces_dir,faces[face1]))
            im2 = load_image(os.path.join(faces_dir,faces[face2]))
    
            data1 = im1.to(torch.device("cuda"))
            _,_,ld1 = model(data1)
    
            data2 = im2.to(torch.device("cuda"))
            _,_,ld2 = model(data2)
    
            # write similarity for original images:
            if args.l2_metric:
                sim = l2_metric(im1.view(-1,1).numpy(),im2.view(-1,1).numpy())
            else:
                sim = cosin_metric(im1.view(-1,1).numpy().T,im2.view(-1,1).numpy())
            dists.write('%.3f, ' %sim)
    
            # and now for the layers
            ld1_list = list(ld1.items())
            ld2_list = list(ld2.items())
            for l in range(len(ld1_list)):
                ld1_np = ld1_list[l][1].data.cpu().numpy()
                ld2_np = ld2_list[l][1].data.cpu().numpy()
                if args.l2_metric:
                    sim = l2_metric(ld1_np,ld2_np)
                else:
                    sim = cosin_metric(ld1_np,ld2_np.T)

                dists.write('%.3f, ' %sim)

            dists.write('\n')
            

def divide_identities_using_average(model,args):

    model.eval()
    
#    train_root = '/home/n/datasets/vggface2_align/train'
#    clustered_train_root = '/home/n/datasets/vggface2_align/clustered'
    train_root = '/home/na/datasets/face_only_sample/train'
    clustered_train_root = '/home/na/datasets/face_only_sample/train'
    ids = natsorted(os.listdir(train_root))
    
    for i in range(len(ids)):
        cur_dir = os.path.join(train_root,ids[i])
        if os.path.isdir(cur_dir):
            print('processing dir: %s' %(ids[i]))
            identity_name = ids[i]
            imgs = natsorted(fnmatch.filter(os.listdir(cur_dir),'*.jpg'))
            img_paths = []
            for j in range(len(imgs)):
                img_paths.append(os.path.join(cur_dir,imgs[j]))
            features,_,_ = get_featurs(model, img_paths,args.test_batch_size)
    #
    #
    #        start = time.time()
    #
    #    num_ids = 1000
    #    #for i in range(len(all_embd_files)):
    #    for i in range(num_ids):
    #
    #        dir_embds = np.load(os.path.join(opt.embds_dir,all_embd_files[i]))
    #        
    #        # the file structure is:
    #        # 0 - index_of_identity (name of identity directory)
    #        # 1 - index_of_cluster (leave nan for now)
    #        # 2 - index_of_image (name of image)
    #        # 3 - serial_num_of_image_in_dir
    #
    #        # calculate average embedding
    #        norm_embd = dir_embds[:,4:]/np.linalg.norm(dir_embds[:,4:],ord=2,axis=1)[:,None]
    #        #mean_embd = np.mean(dir_embds[:,4:],axis=0)
    #        mean_embd = np.mean(norm_embd,axis=0)
    
            mean_embd = np.mean(features,axis=0)
            # now go over all embedding
            # if similarity from average is larger than th, put in inliers
            # else - put in outliers
            
            # inliers dir
    #        (fname,ext) = all_embd_files[i].split('.')
    #        identity_name = fname[5:]
    
            inliers_dir_name = os.path.join(clustered_train_root,identity_name)
            if not os.path.exists(inliers_dir_name):
                os.makedirs(inliers_dir_name)
            # outliers dir
            outliers_dir_name = os.path.join(clustered_train_root,identity_name + '_out')
            if not os.path.exists(outliers_dir_name):
                os.makedirs(outliers_dir_name)
            
            dists = []
            #for j in range(dir_embds.shape[0]):
            for j in range(features.shape[0]):
                #image_name = str(int(dir_embds[j,2])) + '.jpg'
                image_name = imgs[j]
                dist = l2_metric(features[j,:],mean_embd)
                dists.append(dist)
                if dist < 9.0:
                #if (cosin_metric(dir_embds[j,4:],mean_embd) > opt.cluster_sim_th):
                #if (cosin_metric(norm_embd[j,:],mean_embd) > opt.cluster_sim_th):
                # if np.linalg.norm(norm_embd[j,:]-mean_embd) < opt.cluster_dist_th:
                    copyfile(os.path.join(train_root,identity_name,image_name),
                             os.path.join(inliers_dir_name,image_name))
                elif dist > 9.0:
                    copyfile(os.path.join(train_root,identity_name,image_name),
                             os.path.join(outliers_dir_name,image_name))
            
            print(np.mean(dists))
            print(np.std(dists))
            
#        duration = time.time() - start
#        print('finished')
#        print(time.strftime("%H:%M:%S", time.gmtime(duration)))
        
def save_embeddings(model,args):

    # take the whole dataset, generate an embedding for each image, write the image id, and save
    # traindir = os.path.join(args.data,'train')
    traindir = args.data
    ids = natsorted(os.listdir(traindir))
    print('save_embeddings num identities: %d' %(len(ids)))
    embds_dir = args.embds_dir
    
    model.eval()
    
    first_id = 0
    # num_dirs_to_process = 2000
    # processed = 0
    # min_imgs = 350
    for i in range(first_id,len(ids)):
        print('processing dir: %s, i: %d' %(ids[i],i))
        id_index = float(ids[i])
        cur_dir = os.path.join(traindir,ids[i])
        imgs = natsorted(fnmatch.filter(os.listdir(cur_dir),'*.png'))
        print('num images: '+str(len(imgs)))
#        if(len(imgs)<min_imgs):
#            print('skipping')
#            continue
#        processed = processed+1
        img_paths = []
        # take maximum 300 images for train
        # num_imgs_to_process = 300
        num_imgs_to_process = len(imgs)
        for j in range(num_imgs_to_process):
            img_paths.append(os.path.join(cur_dir,imgs[j]))
        features,_,_ = get_featurs(model, img_paths,args.test_batch_size)
#        _,features,_ = get_featurs(model, img_paths,args.test_batch_size) # take output as features
        if features.shape[0] != num_imgs_to_process:
            print('save_embeddings got %d features for %d images dir %s' %(features.shape[0],num_imgs_to_process,cur_dir))
            return
        # for each image save:
        # 0 - index_of_identity (name of identity directory)
        # 1 - index_of_image (name of image)
        # 2 - serial_num_of_image_in_dir
        embds_array = np.empty((num_imgs_to_process,3 + features.shape[1]))
        for f in range(features.shape[0]):
            embds_array[f,0] = id_index
            imgs_fname = imgs[f]
            (img_name,ext) = imgs_fname.split('.')
            (season,episode,img_name_main,img_name_sub) = img_name.split('_')
            tmp = img_name_main+'.'+img_name_sub
            embds_array[f,1] = float(tmp)
            #embds_array[f,1] = int(img_name)
            embds_array[f,2] = f
            embds_array[f,3:] = features[f]

        embds_fname = 'embds'+ids[i]+'.npy'
        np.save(os.path.join(embds_dir,embds_fname),embds_array)
        
#        if processed >= num_dirs_to_process:
#            return

def identify_images(model,args):
    # go over all directories
    # collect all embeddings into one big vector
    # go over each one and find the the closest one
    # check if it's the right identity
    # take the whole dataset, generate an embedding for each image, write the image id, and save
    testdir = os.path.join(args.data, 'test')
    ids = natsorted(os.listdir(testdir))
    print('save_embeddings num identities: %d' %(len(ids)))
    
    model.eval()

    all_embds = None
    
    for i in range(len(ids)):
        print('processing dir: %s' %(ids[i]))
        id_index = float(ids[i])
        cur_dir = os.path.join(testdir,ids[i])
        imgs = natsorted(fnmatch.filter(os.listdir(cur_dir),'*.jpg'))
        img_paths = []
        for j in range(len(imgs)):
            img_paths.append(os.path.join(cur_dir,imgs[j]))
        features,_,_ = get_featurs(model, img_paths,args.test_batch_size)
        if features.shape[0] != len(imgs):
            print('save_embeddings got %d features for %d images dir %s' %(features.shape[0],len(imgs),cur_dir))
            return
        # for each image save:
        # 0 - index_of_identity (name of identity directory)
        # 1 - index_of_image (name of image)
        # 2 - serial_num_of_image_in_dir
        embds_array = np.empty((len(imgs),3 + features.shape[1]))
        for f in range(features.shape[0]):
            embds_array[f,0] = id_index
            imgs_fname = imgs[f]
            (img_name,ext) = imgs_fname.split('.')
            embds_array[f,1] = int(img_name)
            embds_array[f,2] = f
            embds_array[f,3:] = features[f]
            
        if all_embds is None:
            all_embds = embds_array
        else:
            all_embds = np.vstack((all_embds,embds_array))
            
    # now take only the embeddings and calculate all distances
    all_dists = squareform(pdist(all_embds[:,3:]))
    # to make sure we ignore the "self" distance, we put Inf in
    # the diagonal
    np.fill_diagonal(all_dists,np.Inf)
    # for each row, take the index of that minimal values
    indices_of_min_dist = np.argmin(all_dists,axis=1)
    # translate indices into identity and image name
    res_mat = np.zeros((all_dists.shape[0],5))
    res_mat[:,0:2] = all_embds[:,0:2]
    res_mat[:,2] = indices_of_min_dist
    for i in range(len(indices_of_min_dist)):
        res_mat[i,3:5] = all_embds[indices_of_min_dist[i],0:2]
    
    res_fname = 'ft_1000_4layers_test_res.txt'    
    np.savetxt(os.path.join(args.out_dir,res_fname),res_mat,delimiter=',')
    


def divide_identities_all_clusters(model,args):

    traindir = os.path.join(args.data, 'train')
    embds_dir = args.embds_dir
    all_embd_files = natsorted(fnmatch.filter(os.listdir(embds_dir),'*.npy'))
    print('divide_identities found %d embedding files in %s' %(len(all_embd_files),embds_dir))
    
    start = time.time()

    for i in range(len(all_embd_files)):
        dir_embds = np.load(os.path.join(embds_dir,all_embd_files[i]))
        # the file structure is:
        # 0 - index_of_identity (name of identity directory)
        # 1 - index_of_image (name of image)
        # 2 - serial_num_of_image_in_dir
        # so we take only the features for clustering
        
        # cluster the embeddings
        Z = linkage(dir_embds[:,3:])
        th = 1.4
        clusters = fcluster(Z,th,depth=10)

        identity_name = str(int(dir_embds[0,0]))

        for j in range(dir_embds.shape[0]):
            cluster_name = clusters[j]
            cluster_dir_name = identity_name + '.' + cluster_name
            if not os.path.exists(os.path.join(traindir,cluster_dir_name)):
                os.makedirs(os.path.join(traindir,cluster_dir_name))

            image_name = str(int(dir_embds[j,1])) + '.jpg'
            copyfile(os.path.join(traindir,identity_name,image_name),
                     os.path.join(cluster_dir_name,image_name))
        

    duration = time.time() - start
    print('finished')
    print(time.strftime("%H:%M:%S", time.gmtime(duration)))


def recognize_list(model,args):

    # Data loading code
    traindir = os.path.join(args.data)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # im_size = 224
    # if inception
    im_size = 299
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(im_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    
    print(args.resume)
    model.eval()
    
    start = time.time()

    flat_list = False
    if(flat_list):
        
        identification_list_fname = '/home/n/School/familiar_unfamiliar_2/wide_training/training_output/wide_familiar_train_id_list.txt'
        test_images = pd.read_csv(identification_list_fname,header=None)
    
        img_paths = []
        for i in range(len(test_images.index)):
            im_name = str(test_images.at[i,1])+'.jpg'
            img_paths.append(os.path.join(traindir,str(test_images.at[i,0]),im_name))
        
        _, outputs, _ = get_featurs(model, img_paths, batch_size=args.test_batch_size,flip=args.test_flip)
    
    
        output_fname = os.path.join(args.out_dir,'wide_train_recognition_res.txt')
    
        with open(output_fname,'w') as of:
            for i in range(len(outputs)):
                # get softmax label
                soft = softmax(outputs[i])
                label = np.argmax(soft)
                for key,val in train_dataset.class_to_idx.items():
                    if val == label:
                        label_id = key
                
                of.write('%d,%d,%s\n' %(test_images.at[i,0],test_images.at[i,1],label_id))
    else:

#        root_dir = '/home/na/lab_home/projects/familiar_unfamiliar/familiar_unfamiliar_6/set2_test'
#        identification_list_fname = '/home/na/lab_home/projects/familiar_unfamiliar/familiar_unfamiliar_6/set2_test_rec_list.txt'
        root_dir = '/home/n/School/familiar_unfamiliar_8/set1_test_ids'
        identification_list_fname = '/home/n/School/familiar_unfamiliar_8/set1_test_rec_list.txt'
        with open(identification_list_fname,'r') as id_fname:
            lines = id_fname.read().splitlines()
        
        img_paths = []
        im_ids = []
        for line in lines:
            [id_name, im_name] = line.split(',')
            im_ids.append((id_name,im_name))
            img_paths.append(os.path.join(root_dir,id_name,im_name))

        _, outputs, _ = get_featurs(model, img_paths, batch_size=args.test_batch_size,flip=args.test_flip)
    
#        output_fname = '/home/na/lab_home/projects/familiar_unfamiliar/familiar_unfamiliar_6/set2_test_rec_res.txt'
        output_fname = '/home/n/School/familiar_unfamiliar_8/set1_finetune/2_layers/set1_test_rec_res.txt'
        
        num_correct = 0
        with open(output_fname,'w') as of:
            for i in range(len(outputs)):
                # get softmax label
                soft = softmax(outputs[i])
                label = np.argmax(soft)
                for key,val in train_dataset.class_to_idx.items():
                    if val == label:
                        label_id = key
                        break
                # test if this is correct
                is_correct = 0
                if im_ids[i][0]==label_id:
                    is_correct = 1
                    num_correct += 1
                
                of.write('%s,%s,%s,%s\n' %(im_ids[i][0],im_ids[i][1],label_id,str(is_correct)))
        
            
    dur = time.time() - start
    print('finished recognition')
    print(time.strftime("%H:%M:%S", time.gmtime(dur)))
    print(num_correct)
    
        
def familiar_unfamiliar_model(model,train_dataset,args):

    # run the following tests and collect data:
    # 1. measure the distance between each pair
    # 2. classify each picture
    # 3. record whether it is in the dataset or not
    # 4. find the nearest neighbor in the trainset
    
#    analyze_fam_unfam_res_mat(args,'/home/n/School/familiar_unfamiliar_dnn/inception_ft_1000_4layers/res_mat_mean.npy')
#    analyze_fam_unfam_res_mat(args,'/home/n/School/familiar_unfamiliar_dnn/ft_clustered_4layers/res_mat_mean.npy')
#    return

    print(args.resume)
    model.eval()
    
    # read all embeddings into one array
    embds_dir = args.embds_dir
    all_embd_files = natsorted(fnmatch.filter(os.listdir(embds_dir),'*.npy'))
    print('found %d embedding files in %s' %(len(all_embd_files),embds_dir))
    
    
    familiar_ids = []
    for i in range(len(all_embd_files)):
        fname = all_embd_files[i]
        familiar_ids.append(float(fname[5:-4]))


    # 1. measure the distance between each pair
    start = time.time()

    identity_list = get_lfw_list(args.lfw_test_list)
    img_paths = [os.path.join(args.lfw_root, each) for each in identity_list]

    features, output, cnt = get_featurs(model, img_paths, batch_size=args.test_batch_size,flip=args.test_flip)

    fe_dict = get_feature_dict(identity_list, features)
    out_dict = get_feature_dict(identity_list, output)

    with open(args.lfw_test_list, 'r') as fd:
        pairs = fd.readlines()

    num_pairs = 200
    num_cols = 16 + (2*args.num_classes)
    res_mat = np.zeros((num_pairs,num_cols))
    
    output_fname = os.path.join(args.out_dir,'familiar_unfamiliar_res_1.txt')
    with open(output_fname,'w') as of:
        of.write('id1,im1,label1_id,id2,im2,label2_id,sim,same_diff\n')
        sims = []
        labels = []
        index = 0
        for pair in pairs:

            (img1,img2,label) = pair.split(' ')
            (id1,im1) = img1.split('/')
            (im1,_) = im1.split('.')
            (id2,im2) = img2.split('/')
            (im2,_) = im2.split('.')

            fe_1 = fe_dict[img1]
            fe_2 = fe_dict[img2]
            same_diff = int(label)
            if args.l2_metric:
                sim = l2_metric(fe_1, fe_2)
            else:        
                sim = cosin_metric(fe_1, fe_2)
    
            sims.append(sim)
            labels.append(same_diff)
            # 2. classify each picture
            # get softmax label
            soft1 = softmax(out_dict[img1])
            label1 = np.argmax(soft1)
            for key,val in train_dataset.class_to_idx.items():
                if val == label1:
                    label1_id = key
            
            soft2 = softmax(out_dict[img2])
            label2 = np.argmax(soft2)
            for key,val in train_dataset.class_to_idx.items():
                if val == label2:
                    label2_id = key
            
            res_mat[index,0] = float(id1)
            res_mat[index,1] = int(im1)
            res_mat[index,2] = float(label1_id)
            res_mat[index,3] = float(id2)
            res_mat[index,4] = int(im2)
            res_mat[index,5] = float(label2_id)
            res_mat[index,6] = sim
            res_mat[index,7] = same_diff
            res_mat[index,16:16+args.num_classes] = soft1
            res_mat[index,16+args.num_classes:16+2*args.num_classes] = soft2
            
            index = index + 1
            
            of.write('%f,%d,%f,%f,%d,%f,%.5f,%d\n' %(float(id1),int(im1),float(label1_id),
                                               float(id2),int(im2),float(label2_id),
                                               sim,same_diff))
            
        acc, th = cal_accuracy(sims, labels,args)
        
        print('same diff acc: %.4f, th: %.3f' %(acc,th))

        dur = time.time() - start
        print('finished verification')
        print(time.strftime("%H:%M:%S", time.gmtime(dur)))
    
    start = time.time()
    
    all_embds = np.load(os.path.join(embds_dir,all_embd_files[0]))
    mean_embds = np.zeros((len(all_embd_files),all_embds.shape[1]))
    mean_embds[0,0] = all_embds[0,0]
    mean_embds[0,3:] = np.mean(all_embds[:,3:],axis=0)
    for i in range(1,len(all_embd_files)):
        tmp = np.load(os.path.join(embds_dir,all_embd_files[i]))
        all_embds = np.concatenate((all_embds,tmp),axis=0)
        mean_embds[i,0] = tmp[0,0]
        mean_embds[i,3:] = np.mean(tmp[:,3:],axis=0)

    dur = time.time() - start
    print('finished loading all embeddings' + str(all_embds.shape[0]))
    print(time.strftime("%H:%M:%S", time.gmtime(dur)))

    output_fname = os.path.join(args.out_dir,'familiar_unfamiliar_res_2.txt')
    with open(output_fname,'w') as of:
        of.write('fam1,fam2,best_id1,best_im1,best_sim1,best_id2,best_im2,best_sim2\n')
        # 3. find the nearest neighbor in the trainset

        start = time.time()
        first_col = 8
        # then find nearest neighbor
        cnt = 0
        for pair in pairs:
            # first get id (identity indicator) and image name
            (img1,img2,label) = pair.split(' ')
            (id1,im1) = img1.split('/')
            (im1,_) = im1.split('.')
            (id2,im2) = img2.split('/')
            (im2,_) = im2.split('.')

            fam1 = int((int(id1) in familiar_ids))
            fam2 = int((int(id2) in familiar_ids))

            # now get feature vector
            fe_1 = fe_dict[img1]
            fe_2 = fe_dict[img2]
            # and find nearest
            (best_id1,best_im1,best_sim1) = get_best_match(all_embds,fe_1,id1,im1)
            (best_id2,best_im2,best_sim2) = get_best_match(all_embds,fe_2,id2,im2)
#            (best_id1,best_im1,best_sim1) = get_best_match(mean_embds,fe_1,id1,im1)
#            (best_id2,best_im2,best_sim2) = get_best_match(mean_embds,fe_2,id2,im2)

            res_mat[cnt,first_col] = fam1
            res_mat[cnt,first_col+1] = fam2
            res_mat[cnt,first_col+2] = float(best_id1)
            res_mat[cnt,first_col+3] = int(best_im1)
            res_mat[cnt,first_col+4] = best_sim1
            res_mat[cnt,first_col+5] = float(best_id2)
            res_mat[cnt,first_col+6] = int(best_im2)
            res_mat[cnt,first_col+7] = best_sim2

            of.write('%d,%d,%f,%d,%.4f,%f,%d,%.4f\n' 
                     %(fam1,fam2,float(best_id1),int(best_im1),best_sim1,float(best_id2),int(best_im2),best_sim2))

            cnt = cnt + 1
            if cnt%10 == 0:
                dur = time.time() - start
                print('tested: ' + str(cnt))
                print(time.strftime("%H:%M:%S", time.gmtime(dur)))
                
        dur = time.time() - start
        print('finished testing nearest neighbor')
        print(time.strftime("%H:%M:%S", time.gmtime(dur)))

    np.savetxt(os.path.join(args.out_dir,'res_mat.npy'),res_mat[:,0:16])

#    analyze_fam_unfam_res_mat(args)

def analyze_fam_unfam_res_mat(args,res_file):
    
    print('analyze_fam_unfam_res_mat file:' + res_file)
    res_mat = np.loadtxt(res_file)
    # columns: id1,im1,label1_id,id2,im2,label2_id,sim,same_diff,fam1,fam2,
    #          best_id1,best_im1,best_sim1,best_id2,best_im2,best_sim2,soft1,soft2
    
    # 1. verification using similarity: overall - with all pairs, familiar and unfamiliar
    # 1a. overall verification
    sims = res_mat[:,6]
    labels = res_mat[:,7]
    acc, verification_th = cal_accuracy(sims,labels,args)
    print('overall similarity verification acc: %.4f, th: %.3f' %(acc,verification_th))
    
    # 1b. familiar
    sims = []
    labels = []
    for i in range(res_mat.shape[0]):
        if res_mat[i,8]==1 and res_mat[i,9]==1:
            sims.append(res_mat[i,6])
            labels.append(res_mat[i,7])
    acc, familiar_ver_th = cal_accuracy(sims,labels,args)
    print('familiar similarity verification acc: %.4f, th: %.3f, num pairs: %d' %(acc,familiar_ver_th,len(sims)))
    
    # 1b. unfamiliar
    sims = []
    labels = []
    for i in range(res_mat.shape[0]):
        if res_mat[i,8]==0 and res_mat[i,9]==0:
            sims.append(res_mat[i,6])
            labels.append(res_mat[i,7])
    acc, unfam_ver_th = cal_accuracy(sims,labels,args)
    print('unfamiliar similarity verification acc: %.4f, th: %.3f, num pairs: %d' %(acc,unfam_ver_th,len(sims)))

    # 2. verification using class labels: overall, familiar, unfamiliar
    # take each row, check if labels are the same for same pairs
    # or labels are different for different pairs - count as true
    # 2a. overall
    overall_acc = 0
    for i in range(res_mat.shape[0]):
        correct_same = (res_mat[i,7]==1 and res_mat[i,2]==res_mat[i,5])
        correct_diff = (res_mat[i,7]==0 and res_mat[i,2]!=res_mat[i,5])
        if correct_same or correct_diff:
            overall_acc = overall_acc + 1
    print('overall class-based verification acc: %.3f' %(overall_acc/res_mat.shape[0]))
    
    # 2b. familiar
    familiar_acc = 0
    familiar_pairs = 0
    for i in range(res_mat.shape[0]):
        if res_mat[i,8]==1 and res_mat[i,9]==1:
            familiar_pairs = familiar_pairs+1
            correct_same = (res_mat[i,7]==1 and res_mat[i,2]==res_mat[i,5])
            correct_diff = (res_mat[i,7]==0 and res_mat[i,2]!=res_mat[i,5])
            if correct_same or correct_diff:
                familiar_acc = familiar_acc + 1
    print('familiar class-based verification acc: %.3f' %(familiar_acc/familiar_pairs))
    
    # 2c. unfamiliar
    unfamiliar_acc = 0
    unfamiliar_pairs = 0
    for i in range(res_mat.shape[0]):
        if res_mat[i,8]==0 and res_mat[i,9]==0:
            unfamiliar_pairs = unfamiliar_pairs+1
            correct_same = (res_mat[i,7]==1 and res_mat[i,2]==res_mat[i,5])
            correct_diff = (res_mat[i,7]==0 and res_mat[i,2]!=res_mat[i,5])
            if correct_same or correct_diff:
                unfamiliar_acc = unfamiliar_acc + 1
    print('unfamiliar class-based verification acc: %.3f' %(unfamiliar_acc/unfamiliar_pairs))

    # 3. conceptual matching using nearest neighbor - using different thresholds
    # first we find the best threshold for correctly finding familiar faces
#    sims = []
#    labels = []
#    for i in range(res_mat.shape[0]):
#        sims.append(res_mat[i,12]) # best_sim1
#        if res_mat[i,0]==res_mat[i,10]: # id1 == best_id1
#            labels.append(1)
#        else:
#            labels.append(0)
#        sims.append(res_mat[i,15]) # best_sim2
#        if res_mat[i,3]==res_mat[i,13]: # id1 == best_id1
#            labels.append(1)
#        else:
#            labels.append(0)
#
#    acc, best_fam_th = cal_accuracy(sims,labels,args)
#    print('accuracy for detecting familiar faces: %.4f, th: %.3f' %(acc,best_fam_th))
#
#    # calc ROC for finding familiar faces
#    thresholds = np.arange(-1.0,1.0,0.01)
#    tp = np.zeros((1,len(thresholds)))
#    fa = np.zeros((1,len(thresholds)))
#    for i in range(len(thresholds)):
#        th = thresholds[i]
#        num_true_fam = 0
#        num_false_fam = 0
#        for j in range(len(sims)):
#            if sims[j] >= th:
#                if labels[j] == 1:
#                    num_true_fam = num_true_fam + 1
#                else:
#                    num_false_fam = num_false_fam + 1
#        tp[0,i] = num_true_fam / sum(labels)
#        fa[0,i] = num_false_fam / (len(labels) - sum(labels))
#
#    tmp = np.where(tp[0] > 0.99)
#    percent_th = thresholds[tmp[0][-1]]
#
#    plt.plot(fa[0],tp[0])
#    plt.show()    
#
#
#    sims = []
#    labels = []
#    for i in range(res_mat.shape[0]):
#        sims.append(res_mat[i,12]) # best_sim1
#        if res_mat[i,8]==1: # fam1
#            labels.append(1)
#        else:
#            labels.append(0)
#        sims.append(res_mat[i,15]) # best_sim2
#        if res_mat[i,9]==1: # fam2
#            labels.append(1)
#        else:
#            labels.append(0)
#
#    acc, best_fam_th = cal_accuracy(sims,labels,args)
#    print('accuracy for detecting familiar faces: %.4f, th: %.3f' %(acc,best_fam_th))
#
#    # calc ROC for finding familiar faces
#    thresholds = np.arange(-1.0,1.0,0.01)
#    tp = np.zeros((1,len(thresholds)))
#    fa = np.zeros((1,len(thresholds)))
#    for i in range(len(thresholds)):
#        th = thresholds[i]
#        num_true_fam = 0
#        num_false_fam = 0
#        for j in range(len(sims)):
#            if sims[j] >= th:
#                if labels[j] == 1:
#                    num_true_fam = num_true_fam + 1
#                else:
#                    num_false_fam = num_false_fam + 1
#        tp[0,i] = num_true_fam / sum(labels)
#        fa[0,i] = num_false_fam / (len(labels) - sum(labels))
#
#    tmp = np.where(tp[0] > 0.9)
#    percent_th = thresholds[tmp[0][-1]]
#
#    plt.plot(fa[0],tp[0])
#    plt.show()    


    thresholds = np.arange(-1.0,1.0,0.01)
    best_acc = 0
    best_th = None
    best_same = None
    best_diff = None
    best_unf_same = None
    best_unf_diff = None
    
    for th in thresholds:
        #th = familiar_ver_th
        #th = verification_th
        # go over all pairs
        # find the best threshold for correct "familiar" tagging
        # then, test accuracy on unfamiliar faces using this threshold
#    thresholds = [verification_th,best_fam_th,percent_th]
#    for th in thresholds:
#        fam_acc = 0
#        num_fam = 0
#        unfam_acc = 0
#        num_unfam = 0
        acc = 0
        num_correct_same = 0
        num_correct_diff = 0
        num_unf_correct_same = 0
        num_unf_correct_diff = 0
        
        for i in range(int(res_mat.shape[0]/2)):
    
            if (res_mat[i,12] >= th) and (res_mat[i,15] >= th):
#                num_fam = num_fam + 1
                # both are detected as familiar, check same/diff according to nearest neighbor
                # if same then id1==best_id1, and id2==best_id2
                correct_same = (res_mat[i,7]==1 and res_mat[i,0]==res_mat[i,10] and res_mat[i,3]==res_mat[i,13])
                if correct_same:
                    num_correct_same = num_correct_same + 1
                    acc = acc + 1
                    
                # if diff then best_id1 != best_id2
                correct_diff = (res_mat[i,7]==0 and res_mat[i,10]!=res_mat[i,13])
                if correct_diff:
                    num_correct_diff = num_correct_diff + 1
                    acc = acc + 1
            else:
#                num_unfam = num_unfam + 1
                # check same/diff using sim
                correct_same = (res_mat[i,7]==1 and res_mat[i,6] >= unfam_ver_th)
                if correct_same:
                    num_unf_correct_same = num_unf_correct_same + 1
                    acc = acc+1
                correct_diff = (res_mat[i,7]==0 and res_mat[i,6] < unfam_ver_th)
                if correct_diff:
                    num_unf_correct_diff = num_unf_correct_diff + 1
                    acc = acc+1
        if acc > best_acc:
            best_acc = acc
            best_th = th
            best_same = num_correct_same
            best_diff = num_correct_diff
            best_unf_same = num_unf_correct_same
            best_unf_diff = num_unf_correct_diff
            
    print('conceptual nearest-neighbor 1, best_acc: %.3f best_th: %.3f, best_same: %d, best_diff: %d, best_unf_same: %d, best_unf_diff: %d' 
          %(best_acc,best_th,best_same,best_diff,best_unf_same,best_unf_diff))

    
def get_best_match(all_embds,fe,id_name,im_name):
    
    best_id = -1
    best_im = -1
    best_sim = -100
    for i in range(all_embds.shape[0]):
        sim = cosin_metric(fe,all_embds[i,3:])
        # we don't want to check against the same image
        if (all_embds[i,0] == int(id_name) and all_embds[i,1] == int(im_name)):
            print('checking against self %d, %d, sim: %.3f' %(int(id_name),int(im_name),sim))
        else:
            if sim > best_sim:
                best_sim = sim
                best_id = all_embds[i,0]
                best_im = all_embds[i,1]

    return best_id,best_im,best_sim

def find_good_examples(model,args):
    # take the test faces - 5 identities
    # use the stabdard dnn identities, find nearest neighbor for each image
    # record the images where nearest neighbor was wrong
    print(args.resume)
    model.eval()
    
    # read all embeddings into one array
    embds_dir = args.embds_dir
    all_embd_files = natsorted(fnmatch.filter(os.listdir(embds_dir),'*.npy'))
    print('found %d embedding files in %s' %(len(all_embd_files),embds_dir))
    
    start = time.time()
    all_embds = np.load(os.path.join(embds_dir,all_embd_files[0]))
    for i in range(1,len(all_embd_files)):
        tmp = np.load(os.path.join(embds_dir,all_embd_files[i]))
        all_embds = np.concatenate((all_embds,tmp),axis=0)
    dur = time.time() - start
    print('finished loading all embeddings' + str(all_embds.shape[0]))
    print(time.strftime("%H:%M:%S", time.gmtime(dur)))

    # read test_faces embeddings
    test_embds_dir = os.path.join(args.out_dir,'test_faces','all_train_embds')
    test_embds = natsorted(fnmatch.filter(os.listdir(test_embds_dir),'*.npy'))
    print('found %d embedding files in %s' %(len(test_embds),test_embds_dir))
    for i in range(len(test_embds)):
        tmp = np.load(os.path.join(test_embds_dir,test_embds[i]))
        all_embds = np.concatenate((all_embds,tmp),axis=0)

    # now we go over the test_embds, find the nearest neighbor for each one
    # if the nn is wrong - write in file
    start = time.time()
    good_examples_fname = os.path.join(args.out_dir,'test_faces','nn_examples.txt')
    with open(good_examples_fname,'w') as nn_ex:
        nn_ex.write('test_id,test_im,best_id,best_im,best_sim\n')
        cnt = 0
        for i in range(len(test_embds)):
            tmp = np.load(os.path.join(test_embds_dir,test_embds[i]))
            for j in range(tmp.shape[0]):
                cnt = cnt + 1
                (best_id,best_im,best_sim) = get_best_match(all_embds,tmp[j,3:],tmp[j,0],tmp[j,1])
#                if best_id != tmp[j,0]:
                nn_ex.write('%s,%s,%s,%s,%f\n' %(tmp[j,0],tmp[j,1],best_id,best_im,best_sim))
                if cnt > 0 and cnt%100 == 0:
                    dur = time.time() - start
                    print('num tested:' + str(cnt))
                    print(time.strftime("%H:%M:%S", time.gmtime(dur)))
                    

def test_same_images(model,args):

    test_embds_dir = args.embds_dir
    test_embds = natsorted(fnmatch.filter(os.listdir(test_embds_dir),'*.npy'))
    print('found %d embedding files in %s' %(len(test_embds),test_embds_dir))
    th = 0.498
    test_same_fname = os.path.join(args.out_dir,'test_same.txt')
    with open(test_same_fname,'w') as test_same:
        test_same.write('id1,im1,id2,im2,sim\n')
        for i in range(len(test_embds)):
            print(i)
            tmp = np.load(os.path.join(test_embds_dir,test_embds[i]))
            for j in range(tmp.shape[0]):
                em1 = tmp[j,3:]
                for k in range(j+1,tmp.shape[0]):
                    em2 = tmp[k,3:]
                    sim = cosin_metric(em1,em2)
                    if sim < th:
                        test_same.write('%s,%s,%s,%s,%f\n' %(tmp[j,0],tmp[j,1],tmp[k,0],tmp[k,1],sim))
                 
def analyze_wrong_res():
    # open the "wrong_same" file
    # for each wrong pair, check if the nearest neighbor in "standard" is correct
    # and if nn in "narrow" is correct
    f1 = '/home/n/School/familiar_unfamiliar_dnn/inception_ft_1000_4layers/test_faces/standard_wrong_same.txt'
    swn_df = pd.read_csv('/home/n/School/familiar_unfamiliar_dnn/inception_ft_1000_4layers/test_faces/standard_wrong_nn.txt')
    nan_df = pd.read_csv('/home/n/School/familiar_unfamiliar_dnn/ft_clustered_4layers/test_faces/narrow_all_nn.txt')
    f4 = '/home/n/School/familiar_unfamiliar_dnn/standard_narrow_nn.txt'
    with open(f1,'r') as standard_wrong_same, open(f4,'w') as standard_narrow_nn:
        standard_narrow_nn.write('id1,im1,id2,im2,sim,standard,nn_id1,nn_im1,nn_id2,nn_im2,narrow\n')
        sws = standard_wrong_same.readlines()
        cnt = 0
        num_fail_standard = 0
        num_succ_narrow = 0
        for line in sws:
            if cnt == 0:
                cnt = cnt + 1
                continue
            cnt = cnt + 1
            id1,im1,id2,im2,sim = line.split(',')
            id1 = float(id1)
            im1 = float(im1)
            id2 = float(id2)
            im2 = float(im2)
            sim = float(sim)
            standard_narrow_nn.write('%f,%f,%f,%f,%f,' %(id1,im1,id2,im2,sim))
            # check if these images appear in standard_wrong_nn
            # if either appear, it neans that they would have been wrong
            if ((swn_df['test_id']==id1) & (swn_df['test_im']==im1)).any() or ((swn_df['test_id']==id2) & (swn_df['test_im']==im2)).any():
                num_fail_standard = num_fail_standard + 1
                standard_narrow_nn.write('0,')
            else:
                standard_narrow_nn.write('1,')
                
            # check what is the nn using the narrow_dnn
            # if both nn's are correct, count it
            nn_id1 = nan_df[(nan_df['test_id']==id1) & (nan_df['test_im']==im1)]['best_id'].array[0]
            nn_im1 = nan_df[(nan_df['test_id']==id1) & (nan_df['test_im']==im1)]['best_im'].array[0]
            nn_id2 = nan_df[(nan_df['test_id']==id2) & (nan_df['test_im']==im2)]['best_id'].array[0]
            nn_im2 = nan_df[(nan_df['test_id']==id2) & (nan_df['test_im']==im2)]['best_im'].array[0]
            standard_narrow_nn.write('%f,%f,%f,%f,' %(nn_id1,nn_im1,nn_id2,nn_im2))
            if nn_id1==id1 and nn_id2==id2:
                num_succ_narrow = num_succ_narrow + 1
                standard_narrow_nn.write('1')
            else:
                standard_narrow_nn.write('0')
            
            standard_narrow_nn.write('\n')
        
    
    
if __name__ == '__main__':
    main()