import os
import numpy as np
import cv2
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import utils

class RandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.ch, self.cw = crop_size
        ih, iw = image_size

        self.h1 = random.randint(0, ih - self.ch)
        self.w1 = random.randint(0, iw - self.cw)

        self.h2 = self.h1 + self.ch
        self.w2 = self.w1 + self.cw

    def __call__(self, img):
        if len(img.shape) == 3:
            return img[self.h1: self.h2, self.w1: self.w2, :]
        else:
            return img[self.h1: self.h2, self.w1: self.w2]

class RAW2RGBDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.in_root = opt.in_root
        self.out_root = opt.out_root
        self.sal_root = opt.sal_root

    def img_aug(self, raw, rgb, sal):
        # random rotate
        if self.opt.angle_aug:
            # rotate
            rotate = random.randint(0, 3)
            if rotate != 0:
                raw = np.rot90(raw, rotate)
                rgb = np.rot90(rgb, rotate)
                sal = np.rot90(sal, rotate)
            # horizontal flip
            if np.random.random() >= 0.5:
                raw = cv2.flip(raw, flipCode = 1)
                rgb = cv2.flip(rgb, flipCode = 1)
                sal = cv2.flip(sal, flipCode = 1)
        return raw, rgb, sal

    def __getitem__(self, index):
        # Define path
        pngname = str(index) + '.png'                                   # png: input RGBA
        jpgname = str(index) + '.jpg'                                   # jpg: output RGB; saliency map Grayscale
        in_path = os.path.join(self.in_root, pngname)
        out_path = os.path.join(self.out_root, jpgname)
        sal_path = os.path.join(self.sal_root, jpgname)
        # Read images
        # input
        raw = Image.open(in_path)
        raw = np.array(raw).astype(np.float64)
        raw = (raw - 128) / 128
        # output
        rgb = Image.open(out_path)
        rgb = np.array(rgb).astype(np.float64)
        rgb = (rgb - 128) / 128
        # saliency map
        sal = Image.open(sal_path)
        sal = np.array(sal).astype(np.float64)
        sal = np.expand_dims(sal, axis = 2)
        sal = sal / 255
        raw, rgb, sal = self.img_aug(raw, rgb, sal)
        sal = sal.reshape([224, 224, 1])
        raw = torch.from_numpy(raw.transpose(2, 0, 1).astype(np.float32)).contiguous()
        rgb = torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32)).contiguous()
        sal = torch.from_numpy(sal.transpose(2, 0, 1).astype(np.float32)).contiguous()
        return raw, rgb, sal
    
    def __len__(self):
        return len(utils.get_jpgs(self.in_root))
