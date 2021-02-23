# -*-coding:utf-8-*-
import os
import sys
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from PIL import ImageFile
from os.path import getsize
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True
np.random.seed(0)

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
class GridImageDataset(Dataset):
    """
    Data producer that generate a square grid, e.g. 3x3, of patches and their
    corresponding labels from pre-sampled images.
    """

    def __init__(self, data_path, label_path, img_size=768, patch_size=256,
                 crop_size=224, normalize=True, way="train", key_word="", epoch=0):
        """
        Initialize the data producer.

        Arguments:
            data_path: string, path to pre-sampled images using patch_gen.py
            label_path: string, path to the label in npy format
            img_size: int, size of pre-sampled images, e.g. 768
            patch_size: int, size of the patch, e.g. 256
            crop_size: int, size of the final crop that is feed into a CNN,
                e.g. 224 for ResNet
            normalize: bool, if normalize the [0, 255] pixel values to [-1, 1],
                mostly False for debuging purpose
        """
        self._data_path = data_path
        self._label_path = label_path
        self._img_size = img_size
        self._patch_size = patch_size
        self._crop_size = crop_size
        self._normalize = normalize
        self._color_jitter = transforms.ColorJitter(64.0 / 255, 0.75, 0.25, 0.04)
        self._way = way
        self._key_word = key_word
        self.epoch = epoch
        self._preprocess()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
        ]
        )

    def _preprocess(self):

        self._patch_per_side = self._img_size // self._patch_size
        self._grid_size = self._patch_per_side * self._patch_per_side

        self._pth = []
        # 20倍
        if self._way == "train":
            sample_num = 10000
        else:
            sample_num = 2000
        for data_path in self._data_path:
            data_path = data_path + '/' + self._way
            self._read_list(data_path, sample=sample_num)

    def _read_list(self, path, sample=2000):
        for kind in os.listdir(path):
            # type pth
            kind_pth = os.path.join(path, kind)
            sub_sample = sample // len(os.listdir(kind_pth))
            for sub_kind in os.listdir(kind_pth):
                files = os.listdir(os.path.join(kind_pth, sub_kind))
                if len(files) > sub_sample:
                    rand = random.sample(range(len(files)), sub_sample)
                else:
                    rand = range(len(files))
                for idx in rand:
                    img_pth = os.path.join(os.path.join(kind_pth, sub_kind), files[idx])
                    if os.path.exists(img_pth):
                        size = getsize(img_pth)
                        size = size / 1024.0
                        if size < 650:
                            continue
                    self._pth.append(img_pth)


    def __len__(self):
        return len(self._pth)

    def __getitem__(self, idx):
        img = Image.open(self._pth[idx])
        label_name = self._pth[idx].split('/')[-3] + '/' + self._pth[idx].split('/')[-2] + '/' + self._pth[idx].split('/')[-1]
        label_name = label_name.split('.png')[0] + '.npy'
        label_flat = np.load(os.path.join(self._label_path + '/' + self._way, label_name))
        if (img.size)[0] != self._img_size:
            self._img_size = 672
            self._patch_size = 224
            self._crop_size = 224

        # color jitter
        if self._way == "train":
            img = self._color_jitter(img)

        # PIL image:   H x W x C
        # torch image: C X H X W

        img = self.transform(img)
        img = np.array(img, dtype=np.float32).transpose((2, 0, 1))
        if self._normalize:
            img = (img - 128.0) / 128.0
        assert not np.isnan(np.min(img))
        # flatten the square grid
        img_flat = np.zeros(
            (self._grid_size, 3, self._crop_size, self._crop_size),
            dtype=np.float32)
        idx_tmp = 0
        for x_idx in range(self._patch_per_side):
            for y_idx in range(self._patch_per_side):
                # center crop each patch
                x_start = int(
                    (x_idx + 0.5) * self._patch_size - self._crop_size / 2)
                x_end = x_start + self._crop_size
                y_start = int(
                    (y_idx + 0.5) * self._patch_size - self._crop_size / 2)
                y_end = y_start + self._crop_size
                img_flat[idx_tmp] = img[:, x_start:x_end, y_start:y_end]
                # label_flat[idx] = label_grid[x_idx, y_idx]
                #
                idx_tmp += 1
        return (img_flat, label_flat, self._pth[idx])
