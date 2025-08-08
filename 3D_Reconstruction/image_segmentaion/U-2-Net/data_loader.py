# data_loader.py
from __future__ import print_function, division
import glob
import torch
from skimage import io, transform, color
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

# ========================== Dataset Load ==========================
class RescaleT(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # 调整图像和标签的大小
        img = transform.resize(image, (self.output_size, self.output_size), mode='constant')
        lbl = transform.resize(label, (self.output_size, self.output_size), mode='constant', order=0, preserve_range=True)

        return {'imidx': imidx, 'image': img, 'label': lbl}

class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        if random.random() >= 0.5:
            image = image[::-1]
            label = label[::-1]

        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # 调整图像和标签的大小
        img = transform.resize(image, (new_h, new_w), mode='constant')
        lbl = transform.resize(label, (new_h, new_w), mode='constant', order=0, preserve_range=True)

        return {'imidx': imidx, 'image': img, 'label': lbl}

class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        if random.random() >= 0.5:
            image = image[::-1]
            label = label[::-1]

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        image = image[top: top + new_h, left: left + new_w]
        label = label[top: top + new_h, left: left + new_w]

        return {'imidx': imidx, 'image': image, 'label': label}

class ToTensor(object):
    """将 ndarrays 转换为张量。"""

    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
        tmpLbl = np.zeros(label.shape)

        # 处理图像归一化，避免除以零
        image_max = np.max(image)
        if image_max > 1e-6:
            image = image / image_max
        else:
            print(f"警告：图像索引 {imidx[0]} 的最大值为零。")
            image = image

        # 处理标签归一化，避免除以零
        label_max = np.max(label)
        if label_max > 1e-6:
            label = label / label_max
        else:
            print(f"警告：标签索引 {imidx[0]} 的最大值为零。")
            label = label

        # 图像预处理
        if image.shape[2] == 1:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
        else:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
            tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

        tmpLbl[:, :, 0] = label[:, :, 0]

        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = tmpLbl.transpose((2, 0, 1))

        return {
            'imidx': torch.from_numpy(imidx),
            'image': torch.from_numpy(tmpImg).float(),
            'label': torch.from_numpy(tmpLbl).float()
        }

class ToTensorLab(object):
    """将 ndarrays 转换为张量。"""

    def __init__(self, flag=0):
        self.flag = flag

    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        tmpLbl = np.zeros(label.shape)

        # 处理标签归一化，避免除以零
        label_max = np.max(label)
        if label_max > 1e-6:
            label = label / label_max
        else:
            print(f"警告：标签索引 {imidx[0]} 的最大值为零。")
            label = label

        # 根据 flag 处理图像
        if self.flag == 0:  # 使用 RGB 颜色
            image_max = np.max(image)
            if image_max > 1e-6:
                image = image / image_max
            else:
                print(f"警告：图像索引 {imidx[0]} 的最大值为零。")
                image = image

            tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
            if image.shape[2] == 1:
                tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
            else:
                tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
                tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

        # 其他 flag 的处理方式（如需要）...

        tmpLbl[:, :, 0] = label[:, :, 0]

        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = tmpLbl.transpose((2, 0, 1))

        return {
            'imidx': torch.from_numpy(imidx),
            'image': torch.from_numpy(tmpImg).float(),
            'label': torch.from_numpy(tmpLbl).float()
        }

class SalObjDataset(Dataset):
    def __init__(self, img_name_list, lbl_name_list, transform=None):
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.transform = transform

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):

        # 读取图像和标签
        image = io.imread(self.image_name_list[idx])
        imidx = np.array([idx])

        if len(self.label_name_list) == 0:
            label_3 = np.zeros(image.shape)
        else:
            label_3 = io.imread(self.label_name_list[idx])

        # 检查图像和标签是否有效
        if np.max(image) < 1e-6:
            print(f"警告：图像索引 {idx} 的最大值为零。")
        if np.max(label_3) < 1e-6:
            print(f"警告：标签索引 {idx} 的最大值为零。")

        # 确保标签的维度正确
        if len(label_3.shape) == 3:
            label = label_3[:, :, 0]
        else:
            label = label_3

        # 调整维度
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
        if len(label.shape) == 2:
            label = label[:, :, np.newaxis]

        sample = {'imidx': imidx, 'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
