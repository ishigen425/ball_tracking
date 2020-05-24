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
import os, sys, json, datetime
from models.track_net import TrackNet
from utils.get_dataloader import get_dataloader
from env import post_slack
from utils.detector import judge
from models.unet import UNet

def write_log(path, context, mode="a"):
    with open(path, mode=mode) as f:
        f.writelines(context+"\n")

cuda0 = torch.device('cuda:0')
net = UNet(27).to(cuda0)
criterion = nn.MSELoss().to(cuda0)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
train_data_laoder, test_data_loader = get_dataloader(batch_size=4)
write_log("weight/train.log", str(datetime.datetime.now()), "w")
write_log("weight/train.log", "train start")
print(net)
for epoch in range(300):
    # train phase
    running_loss = 0.0
    net.train()
    for i, batch in enumerate(train_data_laoder):
        
        inputs = batch['image'].to(cuda0)
        target = batch['target'].to(cuda0)

        optimizer.zero_grad()
        outputs = net(inputs)
        batch_size = outputs.size(0)
        outputs = outputs.reshape((batch_size, -1))
        target = target.reshape((batch_size, -1))
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    train_info = 'epoch:%d train_loss: %.3f' % (epoch + 1, running_loss / (i + 1))
    print(train_info)
    write_log('weight/train.log', str(datetime.datetime.now()))
    write_log('weight/train.log', train_info)
    torch.save(net.state_dict(), 'weight/epoch_{}_{}'.format(epoch, i))

    # test phase
    with torch.no_grad():
        net.eval()
        test_loss = 0.0
        accuracy = 0
        count = 0
        for i, batch in enumerate(test_data_loader):
            inputs = batch['image'].to(cuda0)
            target = batch['target'].to(cuda0)
            outputs = net(inputs)
            # loss
            batch_size = outputs.size(0)
            loss = criterion(outputs.reshape((batch_size, -1)), target.reshape((batch_size, -1)))
            test_loss += loss.item()
            # accuracy
            target, outputs = target.cpu(), torch.squeeze(outputs.cpu(), dim=1)
            for tar, out in zip(target, outputs):
                accuracy += judge(tar, out)
                count += 1
    test_info = 'epoch:%d test_loss: %.3f accuracy: %.3f' % (epoch + 1, test_loss / (i + 1), accuracy / count)
    print(test_info)
    write_log('weight/train.log', test_info)

    # slackに投げる
    if epoch % 5 == 0:
        try:
            post_slack(train_info + "\n" + test_info)
        except:
            pass
