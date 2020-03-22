from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json
# Ignore warnings
import cv2
import warnings
from .generate_heatmap import make_gaussian
from PIL import Image

warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

class BallDataset(Dataset):
    def __init__(self, json_file, root_dir, transform=None):
        with open(os.path.join(root_dir, json_file)) as f:
            s = f.read()
            j = json.loads(s)

        self.data_set = []
        # filepathとcenterだけ抽出してリスト化する
        for idx, file_id in enumerate(j['assets']):
            asset = j['assets'][file_id]['asset']
            self.data_set.append([os.path.join(root_dir, asset['name'])])
            regions = j['assets'][file_id]['regions']
            flg = False
            if regions:
                for info in regions:
                    if 'ball' in info['tags']:
                        bbox = info['boundingBox']
                        self.data_set[idx].append(tuple(self.get_center(bbox['height'], bbox['width'], bbox['left'], bbox['top'])))
                        flg = True

            if not flg:
                self.data_set[idx].append(tuple((None, None)))
        self.data_set.sort()
        self.root_dir = root_dir
        self.transform = transform
        
    def get_center(self, height, width, top, left):
        top -= height / 2
        left += width / 2
        return left, top

    def __len__(self):
        return len(self.data_set)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        output_size = (int(360*1.5), int(640*1.5))
        img_name = self.data_set[idx][0]
        image = cv2.imread(img_name)
        center = self.data_set[idx][1]
        x = y = 0
        if center[0] != None:
            h, w = center
            h = int(h * output_size[0] / origin_size[0])
            w = int(w * output_size[1] / origin_size[1])
        heatmap = make_gaussian(size=output_size, center=(h, w), is_ball=center[0] != None)
        data = {'image': image, 'target': heatmap}
        if self.transform:
            data = self.transform(data)
        return data
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, target = sample['image'], sample['target']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        target = target.transpose((0, 1))
        return {'image': torch.from_numpy(image).float(),
                'target': torch.from_numpy(target).float()}

def get_dataloader():
    ball_dataset = BallDataset(json_file='table-tenni-ball-tracking-export.json',
                           root_dir='../data/tracking_data/vott-json-export/',
                           transform=transforms.Compose([ToTensor()])
                          )
    return DataLoader(ball_dataset, batch_size=2, shuffle=False, num_workers=0)

