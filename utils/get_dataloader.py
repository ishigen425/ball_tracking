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
import warnings
from .generate_heatmap import make_gaussian
from PIL import Image
import random

warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

class BallDataset(Dataset):
    def __init__(self, json_file_list, root_dir_list, transform=None):
        for i in range(len(json_file_list)):
            json_file, root_dir = json_file_list[i], root_dir_list[i]
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
        self.transform = transform
        
    def get_center(self, height, width, top, left):
        top -= height / 2
        left += width / 2
        return left, top

    def __len__(self):
        return len(self.data_set)

    def _get_image(self, path):
        return Image.open(path).convert("RGB")
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        output_size = (360, 640)
        img_name = self.data_set[idx][0]
        center = self.data_set[idx][1]
        image1 = self._get_image(img_name)
        origin_size = (image1.height, image1.width)
        image1 = np.asarray(image1.resize((output_size[1], output_size[0])))
        # idxが１以下の場合はオール0の画像を渡す
        if idx < 2:
            image2 = np.zeros(output_size)
            image3 = np.zeros(output_size)
        else:
            image2 = np.asarray(self._get_image(self.data_set[idx-1][0]).resize((output_size[1], output_size[0])))
            image3 = np.asarray(self._get_image(self.data_set[idx-2][0]).resize((output_size[1], output_size[0])))
        # チャネル方向にimageを結合
        image = np.dstack((image1, image2, image3))
        h = w = 0
        if center[0] != None:
            h, w = center
            h = (h/origin_size[0]) * output_size[0]
            w = (w/origin_size[1]) * output_size[1]
        heatmap = make_gaussian(size=output_size, center=(h, w), is_ball=center[0] != None)
        print('hetamap', heatmap.shape)
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
    
class RandomFlip(object):

    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        if random.random() < 0.5:
            image = np.flip(image, 1).copy()
            target = np.flip(target, 0).copy()
        if random.random() < 0.5:
            image = np.flip(image, 2).copy()
            target = np.flip(target, 1).copy()
        print('randomflip', image.shape, target.shape)
        return {'image': image, 'target': target}

def get_dataloader():
    ball_dataset = BallDataset(
                            json_file_list=[
                            'vott-json-export/test2-export.json',
                            'vott-json-export/test1-export.json',
                            ],
                           root_dir_list=
                            ['../data/DJI_0014/',
                            '../data/DJI_0015/',
                            ],
                           transform=transforms.Compose([
                            RandomFlip(),
                            ToTensor()
                           ]))
    # 分割する
    n = len(ball_dataset)
    train_size = int(n * 0.8)
    test_size = n - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(ball_dataset, [train_size, test_size])
    # 20000件のデータでは学習が終わらなさそうなので、4000件で実施する
    n = len(test_dataset)
    train_size = int(n * 0.8)
    test_size = n - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [train_size, test_size])
    return DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0), DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)

