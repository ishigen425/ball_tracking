import torch
import torch.nn as nn
import numpy as np
import os, sys, json
from.model_parts import *

class TrackNet(nn.Module):
    def __init__(self):
        super(TrackNet, self).__init__()
        self.conv = self.make_conv()
        self.deconv = self.make_deconv()
        self.final = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)
        
    def make_conv(self):
        channels = [9, 64, 64, 128, 128, 128, 256, 256, 256, 512, 512, 512]
        layers = []
        for i in range(len(channels)-1):
            layers.append(BasicLayers(channels[i], channels[i+1]))
            if i in [2, 6, 9]:
                layers.append(nn.MaxPool2d(
                    kernel_size=2, stride=2, padding=0, dilation=1,
                    ceil_mode=False
                ))
        return nn.Sequential(*layers)
    
    def make_deconv(self):
        channels = [512, 512, 512, 512, 256, 256, 256, 128, 128, 128, 64, 64, 64]
        layers = []
        for i in range(len(channels)-1):
            if i in [0, 4, 7]:
                layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            layers.append(BasicLayers(channels[i], channels[i+1]))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.deconv(x)
        return self.final(x)
