import cv2
import numpy as np
from torch.utils.data import Dataset
import openslide
import PIL
from PIL import Image
np.random.seed(0)


class GridWSIPatchDataset(Dataset):
    """
    Data producer that generate all of patches,
    from a WSI and its tissue mask, and their corresponding indices with
    respect to the tissue mask
    """

    def __init__(self, wsi_path, mask, image_size=768, patch_size=256,
                 crop_size=224, normalize=True, flip='NONE', rotate='NONE'):
        """
        Initialize the data producer.

        Arguments:
            wsi_path: string, path to WSI file
            mask_path: string, path to mask file in numpy format
            image_size: int, size of the image before splitting into grid, e.g.
                768
            patch_size: int, size of the patch, e.g. 256
            crop_size: int, size of the final crop that is feed into a CNN,
                e.g. 224 for ResNet
            normalize: bool, if normalize the [0, 255] pixel values to [-1, 1],
                mostly False for debuging purpose
            flip: string, 'NONE' or 'FLIP_LEFT_RIGHT' indicating the flip type
            rotate: string, 'NONE' or 'ROTATE_90' or 'ROTATE_180' or
                'ROTATE_270', indicating the rotate type
        """
        self._wsi_path = wsi_path
        self._mask = mask

        self._image_size = image_size
        self._patch_size = patch_size
        self._crop_size = crop_size
        self._normalize = normalize
        self._flip = flip
        self._rotate = rotate
        self._preprocess()

    def _preprocess(self):
        # self._mask = cv2.imread(self._mask_path, 0)
        self._slide = openslide.OpenSlide(self._wsi_path)
        X_slide, Y_slide = self._slide.level_dimensions[0]
        # print("level_1:", self._slide.level_dimensions[0])

        self._mask = self._mask.transpose(1, 0)

        X_mask, Y_mask = self._mask.shape
        X_mask = int(X_mask / 8)
        Y_mask = int(Y_mask / 8)
        # print("mask_1:", self._mask.shape)
        self._mask = cv2.resize(self._mask, (Y_mask, X_mask))
        # print("mask_2:", self._mask.shape)

        self._resolution1 = int(X_slide * 1.0 / X_mask)
        self._resolution2 = int(Y_slide * 1.0 / Y_mask)
        if np.log2(self._resolution1).is_integer():
            self._resolution = self._resolution1
        else:
            self._resolution = self._resolution2

        # all the idces for tissue region from the tissue mask
        self._X_idcs, self._Y_idcs = np.where(self._mask == 255)
        self._idcs_num = len(self._X_idcs)
        if self._image_size % self._patch_size != 0:
            raise Exception('Image size / patch size != 0 : {} / {}'.
                            format(self._image_size, self._patch_size))
        self._patch_per_side = self._image_size // self._patch_size
        self._grid_size = self._patch_per_side * self._patch_per_side

    def __len__(self):
        return self._idcs_num

    def __getitem__(self, idx):
        x_mask, y_mask = self._X_idcs[idx], self._Y_idcs[idx]
        x_center = int((x_mask + 0.5) * self._resolution)
        y_center = int((y_mask + 0.5) * self._resolution)

        x = int(x_center - self._image_size / 2)
        y = int(y_center - self._image_size / 2)

        img = self._slide.read_region(
            (x, y), 1, (self._image_size, self._image_size)).convert('RGB')
        img = img.resize((768, 768), PIL.Image.ANTIALIAS)
        # print(img.size)
        if self._flip == 'FLIP_LEFT_RIGHT':
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        if self._rotate == 'ROTATE_90':
            img = img.transpose(PIL.Image.ROTATE_90)

        if self._rotate == 'ROTATE_180':
            img = img.transpose(PIL.Image.ROTATE_180)

        if self._rotate == 'ROTATE_270':
            img = img.transpose(PIL.Image.ROTATE_270)
        # PIL image:   H x W x C
        # torch image: C X H X W

        img = np.array(img, dtype=np.float32).transpose((2, 0, 1))
        if self._normalize:
            img = (img - 128.0)/128.0

        # flatten the square grid
        img_flat = np.zeros(
            (self._grid_size, 3, self._crop_size, self._crop_size),
            dtype=np.float32)
        idx1 = 0
        # print(self._patch_per_side)
        for x_idx in range(self._patch_per_side):
            for y_idx in range(self._patch_per_side):
                # center crop each patch
                x_start = int(
                    (x_idx + 0.5) * 256 - self._crop_size / 2)
                x_end = x_start + self._crop_size
                y_start = int(
                    (y_idx + 0.5) * 256 - self._crop_size / 2)
                y_end = y_start + self._crop_size
                img_flat[idx1] = img[:, x_start:x_end, y_start:y_end]
                idx1 += 1
        return (img_flat, x_mask, y_mask)
