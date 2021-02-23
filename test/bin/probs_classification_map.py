import sys
import os
import logging
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import cv2
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from data.wsi_producer import GridWSIPatchDataset  # noqa

# def vis(mask, pth):
#     mask = mask.transpose(1, 0)
#     probs = mask * 255
#     probs = cv2.applyColorMap(np.uint8(probs), cv2.COLORMAP_JET)
#     cv2.imwrite(pth + '.png', probs)



def get_probs_map(model, dataloader):
    probs_map = np.zeros((dataloader.dataset._mask.shape))
    num_batch = len(dataloader)
    # only use the prediction of the center patch within the grid
    idx_center = dataloader.dataset._grid_size // 2

    count = 0
    time_now = time.time()
    with torch.no_grad():
        for (data, x_mask, y_mask) in dataloader:
            data = data.cuda()

            output = model(data)

            output = output.sigmoid()
            probs_map[x_mask, y_mask] = output[:, idx_center].cpu().data.numpy().flatten()
            count += 1


    return probs_map


def make_dataloader(tif_pth, mask, batch_size, patch_size, num_workers, flip='NONE', rotate='NONE'):


    dataloader = DataLoader(
        GridWSIPatchDataset(tif_pth, mask,
                            patch_size=patch_size,
                            crop_size=224, normalize=True,
                            flip=flip, rotate=rotate),
        batch_size=batch_size, num_workers=num_workers, drop_last=False)

    return dataloader



def run(model, tif_pth, mask, tif, batch_size, patch_size, num_workers, vis_pth):
    start_t = time.time()
    dataloader = make_dataloader(tif_pth, mask, batch_size, 
                                 patch_size, num_workers, flip='NONE', rotate='NONE')
    # print(len(dataloader))
    probs_map = get_probs_map(model, dataloader)

    # nsp = npy_pth + '/' + tif
    # np.save(nsp, probs_map)
    vsp = vis_pth + '/' + tif + '.png'
    # vis(probs_map, vsp)
    all_time = time.time() - start_t
    print("wsi need time:", all_time)
    return probs_map, vsp


def probs_cls_maps(model_path, tif_pth, mask, batch_size, patch_size, num_workers, vis_pth):

    model = torch.jit.load(model_path)
    model = model.cuda()

    tif = tif_pth.split('/')[-1].split('.tif')[0]
    npy, vsp = run(model, tif_pth, mask, tif, batch_size, patch_size, num_workers, vis_pth)
    return npy, vsp


