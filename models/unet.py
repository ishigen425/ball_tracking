import torch
import torch.nn as nn
import numpy as np
from .model_parts import *

class UNet(nn.Module):
    def __init__(self, in_channels):
        super(UNet, self).__init__()
        self.conv1 = BasicLayers(in_channels, 64)
        self.conv2 = BasicLayers(64, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = BasicLayers(64, 128)
        self.conv4 = BasicLayers(128, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv6 = BasicLayers(128, 256)
        self.conv7 = BasicLayers(256, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv9 = BasicLayers(256, 512)
        self.conv10 = BasicLayers(512, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv11 = BasicLayers(512, 1024)
        self.conv12 = BasicLayers(1024, 512)
        self.up1 = Upsample(scale_factor=2, mode='bilinear')
        self.conv13 = BasicLayers(1024, 256)
        self.conv14 = BasicLayers(256, 256)
        self.up2 = Upsample(scale_factor=2, mode='bilinear')
        self.conv15 = BasicLayers(512, 128)
        self.conv16 = BasicLayers(128, 128)
        self.up3 = Upsample(scale_factor=2, mode='bilinear')
        self.conv17 = BasicLayers(256, 64)
        self.conv18 = BasicLayers(64, 64)
        self.up4 = Upsample(scale_factor=2, mode='bilinear')
        self.conv19 = BasicLayers(128, 64)
        self.conv20 = BasicLayers(64, 64)
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def concat(self, x1, x2):
        # 360x640だからこのpadの方法を取る。他の画像サイズに対応できる方法ではない。
        if x1.size() != x2.size():
            x1 = F.pad(x1, [0,0,0,1])
        return torch.cat([x2, x1], 1)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.conv2(x)
        x = self.pool1(x1)
        x = self.conv3(x)
        x2 = self.conv4(x)
        x = self.pool2(x2)
        x = self.conv6(x)
        x3 = self.conv7(x)
        x = self.pool3(x3)
        x = self.conv9(x)
        x4 = self.conv10(x)
        x = self.pool4(x4)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.up1(x)
        x = self.concat(x, x4)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.up2(x)
        x = self.concat(x, x3)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.up3(x)
        x = self.concat(x, x2)
        x = self.conv17(x)
        x = self.conv18(x)
        x = self.up4(x)
        x = self.concat(x, x1)
        x = self.conv19(x)
        x = self.conv20(x)
        x = self.out(x)
        return x

