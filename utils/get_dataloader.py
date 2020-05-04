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

class BallDataset(Dataset):
    def __init__(self, json_file_list, root_dir_list, interval=1,transform=None):
        self.data_set = []
        for i in range(len(json_file_list)):
            json_file, root_dir = json_file_list[i], root_dir_list[i]
            self.read_json(root_dir, json_file)
        self.data_set.sort()
        self.transform = transform
        self.interval = interval

    def read_json(self, root_dir, json_file):
        with open(os.path.join(root_dir, json_file)) as f:
            s = f.read()
            j = json.loads(s)
        # filepathとcenterだけ抽出してリスト化する
        for idx, file_id in enumerate(j['assets']):
            asset = j['assets'][file_id]['asset']
            regions = j['assets'][file_id]['regions']
            if regions:
                for info in regions:
                    # ballが写っていないフレームは無視する
                    if 'ball' in info['tags']:
                        bbox = info['boundingBox']
                        self.data_set.append([os.path.join(root_dir, asset['name']), tuple(self.get_center(bbox['height'], bbox['width'], bbox['left'], bbox['top']))])
                        break
        
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
        output_size = (int(360*1.5), int(640*1.5))
        img_name = self.data_set[idx][0]
        center = self.data_set[idx][1]
        image2 = self._get_image(img_name)
        origin_size = (image2.height, image2.width)
        image2 = np.asarray(image2.resize((output_size[1], output_size[0])))
        image1_idx = idx-self.interval if idx-self.interval >= 0 else 0
        image3_idx = idx+self.interval if idx+self.interval < len(self.data_set) else -1
        image1 = np.asarray(self._get_image(self.data_set[image1_idx][0]).resize((output_size[1], output_size[0])))
        image3 = np.asarray(self._get_image(self.data_set[image3_idx][0]).resize((output_size[1], output_size[0])))
        # チャネル方向にimageを結合
        image = np.dstack((image1, image2, image3))
        h = w = 0
        if center[0] != None:
            h, w = center
            h = int((h/origin_size[0]) * output_size[0])
            w = int((w/origin_size[1]) * output_size[1])
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
    
class RandomFlip(object):

    def __call__(self, sample):
        image, target = sample['image'], sample['target']
        # image, target H x W x C
        if random.random() < 0.5:
            image = np.flip(image, 0).copy()
            target = np.flip(target, 0).copy()
        if random.random() < 0.5:
            image = np.flip(image, 1).copy()
            target = np.flip(target, 1).copy()
        return {'image': image, 'target': target}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, target = sample['image'], sample['target']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        target = target[top: top + new_h,
                      left: left + new_w]
        return {'image': image, 'target': target}

def get_dataloader(batch_size):
    # local環境に合わせてるだけ。。。
    ball_dataset = BallDataset(
                            json_file_list=[
                            'vott-json-export/test2-export.json',
                            'vott-json-export/test1-export.json',
                            ],
                           root_dir_list=
                            ['../data/DJI_0014/',
                            '../data/DJI_0015/',
                            ],
                           interval=4,
                           transform=transforms.Compose([
                            RandomCrop((360, 640)),
                            RandomFlip(),
                            ToTensor()
                           ]))
    # 分割する
    n = len(ball_dataset)
    train_size = int(n * 0.8)
    test_size = n - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(ball_dataset, [train_size, test_size])
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0), DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
