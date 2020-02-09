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
import os, sys, json
from model import TrackNet, BallCrossEntropy
from utils.get_dataloader import get_dataloader
from env import post_slack

cuda0 = torch.device('cuda:0')
net = TrackNet().to(cuda0)
criterion = nn.MSELoss().to(cuda0)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
train_data_laoder, test_data_loader = get_dataloader()

for epoch in range(300):  # loop over the dataset multiple times
    # train phase
    running_loss = 0.0
    net.train()
    for i, batch in enumerate(train_data_laoder):
        
        inputs = batch['image'].to(cuda0)
        target = batch['target'].to(cuda0)
        # 入力データと教師データのスタブ
        # inputs = torch.rand(1, 3, 360, 640).to(cuda0)
        # target = torch.rand(1, 360, 640).to(cuda0)

        optimizer.zero_grad()
        # 損失の計算
        outputs = net(inputs)
        batch_size = outputs.size(0)
        outputs = outputs.reshape((batch_size, -1))
        target = target.reshape((batch_size, -1))
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    train_info = '[%d, %5d] train_loss: %.3f' % (epoch + 1, i + 1, running_loss / (i + 1))
    print(train_info)
    torch.save(net.state_dict(), 'weight/epoch_{}_{}'.format(epoch, i))

    # test phase
    with torch.no_grad():
        net.eval()
        test_loss = 0.0
        for i, batch in enumerate(test_data_loader):
            inputs = batch['image'].to(cuda0)
            target = batch['target'].to(cuda0)
            outputs = net(inputs)
            batch_size = outputs.size(0)
            outputs = outputs.reshape((batch_size, -1))
            target = target.reshape((batch_size, -1))
            loss = criterion(outputs, target)
            test_loss += loss.item()
    
    test_info = '[%d, %5d] test_loss: %.3f' % (epoch + 1, i + 1, test_loss / (i + 1))
    print(test_info)

    # slackに投げる
    if epoch % 10 == 0:
        post_slack(train_info + "\n" + test_info)
