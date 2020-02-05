from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models
from utils.generate_heatmap import gaussian, make_gaussian
import os, sys, json

class TrackNet(nn.Module):
    def __init__(self):
        super(TrackNet, self).__init__()
        self.vgg = self.make_vgg()
        self.deconv = self.make_deconv()
        self.final = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1, stride=1)
        
    def make_vgg(self):
        channels = [3, 64, 64, 128, 128, 128, 256, 256, 256, 512, 512, 512]
        layers = []
        for i in range(len(channels)-1):
            layers.append(nn.Conv2d(
                in_channels=channels[i], out_channels=channels[i+1], kernel_size=3,
                stride=1, padding=1
            ))
            layers.append(nn.BatchNorm2d(
                channels[i+1], eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
            ))
            layers.append(nn.ReLU(inplace=True))
            if i in [2, 6, 9]:
                layers.append(nn.MaxPool2d(
                    kernel_size=2, stride=2, padding=0, dilation=1,
                    ceil_mode=False, return_indices=True
                ))
        return nn.Sequential(*layers)
    
    def make_deconv(self):
        channels = [512, 512, 512, 512, 256, 256, 256, 128, 128, 128, 64, 64, 64]
        layers = []
        for i in range(len(channels)-1):
            if i in [0, 4, 7]:
                layers.append(nn.MaxUnpool2d(kernel_size=2))
            layers.append(nn.ConvTranspose2d(
                in_channels=channels[i], out_channels=channels[i+1], kernel_size=3,
                stride=1, padding=1
            ))
            layers.append(nn.BatchNorm2d(
                channels[i+1], eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
            ))
            layers.append(nn.ReLU(inplace=True))
        # forwardで展開してるから不要だけど、printしたときの見やすさからnn.Sequentialを使う
        return nn.Sequential(*layers)
    
    def forward(self, x):
        indices_list = []
        cnt = 0
        for layer in self.vgg:
            cnt += 1
            if isinstance(layer, nn.MaxPool2d):
                x, indices = layer(x)
                indices_list.append(indices)
            else:
                x = layer(x)
        for layer in self.deconv:
            cnt += 1
            if isinstance(layer, nn.MaxUnpool2d):
                indices = indices_list.pop()
                x = layer(x, indices)
            else:
                x = layer(x)
        #x = F.softmax(x, dim=1)
        #x, _ = torch.max(x, 1)
        return self.final(x)


class BallCrossEntropy(nn.Module):
    def __init__(self):
        super(BallCrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, output, target):
        size_h = output.size(2)
        size_w = output.size(3)
        loss = 0
        for h in range(size_h):
            for w in range(size_w):
                array_255 = output[:, :, h, w]
                g = target[:, h, w]
                loss += self.criterion(array_255, g)
        return loss / (size_h * size_w)
