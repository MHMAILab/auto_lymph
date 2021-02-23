from probs_classification_map import probs_cls_maps
from probs_segmentation_map import prob_map
from data.make_slide_cutting import save_slide_cutting
from compute_area import compute_area
import os
import sys
import argparse
import cv2
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')


parser = argparse.ArgumentParser(description='Get the probability map of tumor'
                                             ' patch predictions given a WSI')
parser.add_argument('--wsi_path', default='./tiff/',
                    metavar='WSI_PATH', type=str,
                    help='Path to the input WSI file')

parser.add_argument('--cls_ckpt', default='./model/classification1.pt',
                    metavar='classification CKPT_PATH', type=str,
                    help='Path to the saved classification ckpt file of a pytorch model')
parser.add_argument('--seg_ckpt', default='./model/segmentation.pt',
                    metavar='segmentation CKPT_PATH', type=str,
                    help='Path to the saved bin ckpt file of a pytorch model')

parser.add_argument('--mlna_path', default='./result/mlna/',
                    metavar='PROBS_MAP_PATH',
                    type=str, help='Path to the output probs_map numpy file')
parser.add_argument('--visualize_path', default='./result/vis/',
                    metavar='PROBS_MAP_PATH',
                    type=str, help='Path to the output probs_map numpy file')
parser.add_argument('--GPU', default='0', type=str, help='which GPU to use'
                                                         ', default 0')
parser.add_argument('--num_workers', default=16, type=int, help='number of '
                    'workers to use to make batch, default 5')
parser.add_argument('--wsi_batch_size', default=16, type=int,
                    help='classification probs map batch size, default 26')
parser.add_argument('--patch_size', default=256, type=int,
                    help='patch size, default 256')

args = parser.parse_args()

if not os.path.exists('./result/'):
    os.mkdir('./result/')

if not os.path.exists(args.mlna_path):
    os.mkdir(args.mlna_path)

if not os.path.exists(args.visualize_path):
    os.mkdir(args.visualize_path)


def generate_hotmap(img, tumor_mask, vsp):
    img_copy = img.copy()

    print(tumor_mask.shape)
    w, h, _ = img_copy.shape
    print(img.shape)
    tumor_mask = np.transpose(tumor_mask)
    print(tumor_mask.shape)
    tumor = cv2.resize(tumor_mask, (h, w))
    tumor = np.array(tumor) * 255
    tumor_vis = cv2.applyColorMap(np.uint8(tumor), cv2.COLORMAP_JET)
    print(img_copy.shape, tumor_vis.shape)
    img_weighted = cv2.addWeighted(img_copy, 0.5, tumor_vis, 0.6, 0)
    img_new = np.hstack((img_weighted, img_copy))
    cv2.imwrite(vsp, img_new)


def run(tif_path, save_path):
    print('generate img')
    img_ori = save_slide_cutting(tif_path, 40)
    print('generate lymph mask')
    lymph_mask = prob_map(args.seg_ckpt, img_ori)
    print('generate tumor mask')
    tumor_mask, vsp = probs_cls_maps(args.cls_ckpt,
                                     tif_path,
                                     lymph_mask,
                                     args.wsi_batch_size,
                                     args.patch_size,
                                     args.num_workers,
                                     args.visualize_path)
    print('generate hotmap')
    generate_hotmap(img_ori, tumor_mask, vsp)
    print('compute area')
    compute_area(lymph_mask, tumor_mask, img_ori, save_path)


if __name__ == '__main__':
    for root, dirs, files in os.walk(args.wsi_path):
        for tif in files:
            img_name = tif.split('.tif')[0]
            save_path = os.path.join(args.mlna_path, img_name + '.png')
            tif_path = os.path.join(root, tif)
            run(args.wsi_path, save_path)
