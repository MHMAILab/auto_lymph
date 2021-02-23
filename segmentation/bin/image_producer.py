import torch.nn as nn
import torch
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import random
import cv2


class ImageDataset(nn.Module):
    def __init__(self, data_path, mask_path, way="train", crop_size=700, normalize=True, model='unet'):
        super(ImageDataset, self).__init__()
        self.imgs = {}
        self.mask = []
        for data_pth in data_path:
            # if "2001-2005" in data_pth:
            #     pth = data_pth
            # else:
            pth = os.path.join(data_pth, way)
            for img in os.listdir(pth):
                self.imgs[img] = os.path.join(pth, img)
        for mask_pth in mask_path:
            pth = os.path.join(mask_pth, way)
            for mask in os.listdir(pth):
                self.mask.append(os.path.join(pth, mask))
        self._way = way
        self._color_jitter = transforms.ColorJitter(
            64.0 / 255, 0.75, 0.25, 0.04)
        self._normalize = normalize
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage()
        ])
        self.use_scale = False
        self._crop_size = crop_size
        self.num_class = 1
        self.model = model

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = 1
        return torch.from_numpy(target)

    def _generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 16) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale,
                           interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale,
                           interpolation=cv2.INTER_NEAREST)
        return image, label

    def norm_or_Standard(self, img, normalize=True):
        if normalize == True:
            img = (img - 128.0) / 128
        else:
            img[:, :, 0] = (img[:, :, 0] - np.mean(img[:, :, 0])
                            ) / np.std(img[:, :, 0])
            img[:, :, 1] = (img[:, :, 1] - np.mean(img[:, :, 1])
                            ) / np.std(img[:, :, 1])
            img[:, :, 2] = (img[:, :, 2] - np.mean(img[:, :, 2])
                            ) / np.std(img[:, :, 2])
        return img

    def __len__(self):
        return len(self.mask)

    def __getitem__(self, idx):
        mask = Image.open(self.mask[idx])
        img_name = self.mask[idx].split(
            '/')[-1].split('_mask')[0] + '_origin_cut_x1.png'
        img = Image.open(self.imgs[img_name])
        if self._way == "train":
            if np.random.rand() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if self.use_scale:
                img, mask = self._generate_scale_label(
                    np.array(img), np.array(mask))
        img, mask = np.array(img), np.array(mask)
        img_h, img_w = mask.shape
        pad_h = max(self._crop_size - img_h, 0)
        pad_w = max(self._crop_size - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(img, 0, pad_h, 0,
                                         pad_w, cv2.BORDER_CONSTANT,
                                         value=(255.0, 255.0, 255.0))
            mask_pad = cv2.copyMakeBorder(mask, 0, pad_h, 0,
                                          pad_w, cv2.BORDER_CONSTANT,
                                          value=(0.0,))
        else:
            img_pad, mask_pad = img, mask
        img_h, img_w = mask_pad.shape
        h_off = random.randint(0, img_h - self._crop_size)
        w_off = random.randint(0, img_w - self._crop_size)
        while (np.all(mask_pad[h_off: h_off + self._crop_size, w_off: w_off + self._crop_size] == 0)):
            h_off = random.randint(0, img_h - self._crop_size)
            w_off = random.randint(0, img_w - self._crop_size)
        img = np.asarray(
            img_pad[h_off: h_off + self._crop_size, w_off: w_off + self._crop_size])
        mask = np.asarray(
            mask_pad[h_off: h_off + self._crop_size, w_off: w_off + self._crop_size], np.float32)

        img_cp = np.asarray(img, np.uint8)
        img, _ = cv2.decolor(img)

        img = np.array(img, dtype=np.float32)

        img = self.norm_or_Standard(img, self._normalize)

        if self._way == "train":
            # rotate
            num_rotate = np.random.randint(0, 4)
            img = np.rot90(img, num_rotate)
            mask = np.rot90(mask, num_rotate)
        mask = self._mask_transform(mask)
        img = torch.from_numpy(img.copy())

        img = img.unsqueeze(0)

        return img, mask, img_cp, img_name
