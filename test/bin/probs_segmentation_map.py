# -*-coding:utf-8-*-
import cv2
import numpy as np
import torch

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')


def prob_map(model_path, img_ori):
    model = torch.jit.load(model_path)
    model = torch.nn.DataParallel(model).cuda()

    with torch.no_grad():
        model.eval()
        try:
            img, _ = cv2.decolor(img_ori)
            img = np.array(img, dtype=np.float32)
            img = (img - 128.0) / 128.0
            img = torch.from_numpy(img.copy())
            print("img:", img.size())
            img = img.unsqueeze(0)
            input = img.unsqueeze(0)
            input = input.cuda()
            output = model(input)
            img_size_1, img_size_2 = input.size(2), input.size(3)
            output = output.squeeze()
            output = output.cpu().numpy()
            mask = np.zeros((img_size_1, img_size_2))
            x, y = np.where(output > 0.5)
            mask[x, y] = 255
            print('end')

        except:

            img, _ = cv2.decolor(img_ori)
            w, h = img.shape
            img = np.array(img, dtype=np.float32)
            img = (img - 128.0) / 128.0
            img = torch.from_numpy(img.copy())
            img = img.unsqueeze(0)
            img = img.unsqueeze(0)
            print("img:", img.size())
            img_slice1 = img[:, :, 0:w // 2, 0:h // 2]
            img_slice2 = img[:, :, w // 2:, 0:h // 2]
            img_slice3 = img[:, :, 0:w // 2, h // 2:]
            img_slice4 = img[:, :, w // 2:, h // 2:]
            # print("img slice1:", img_slice1.size())
            input1 = img_slice1.cuda()
            output1 = model(input1)
            input2 = img_slice2.cuda()
            output2 = model(input2)
            input3 = img_slice3.cuda()
            output3 = model(input3)
            input4 = img_slice4.cuda()
            output4 = model(input4)
            output = torch.cat(
                (torch.cat((output1, output2), 2), torch.cat((output3, output4), 2)), 3)
            output = output.squeeze()
            output = output.cpu().numpy()
            mask = np.zeros((w, h))
            x, y = np.where(output > 0.5)
            mask[x, y] = 255
            print('end oversize')
    return mask
