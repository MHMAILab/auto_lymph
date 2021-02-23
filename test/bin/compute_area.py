import numpy as np
import cv2
import sys
import os
np.set_printoptions(threshold=np.inf)
font = cv2.FONT_HERSHEY_SIMPLEX


def make_tumor_mask(mask_shape, contours):

    wsi_empty = np.zeros(mask_shape[:2])

    wsi_empty = wsi_empty.astype(np.uint8)

    cv2.drawContours(wsi_empty, contours, -1, 255, cv2.FILLED)
    return wsi_empty


def load_npy_tumor(tumor, img_shape):

    np_file = tumor
    np_file[np_file > 0.5] = 255
    np_file[np_file < 0.5] = 0
    np_file = np_file.astype(np.uint8)
    mask = np.transpose(np_file)
    mask_40 = cv2.resize(mask, img_shape, interpolation=cv2.INTER_LINEAR)
    return mask_40


def load_npy_lymph(lymph):
    np_file = lymph

    np_file = np_file.astype(np.uint8)
    return np_file


def get_contours_index(hierarchy, length):
    list_index = []
    if len(hierarchy.shape) != 1:
        for i in range(length):
            if hierarchy[i][3] != -1:
                list_index.append(i)
        return list_index


def compute_area(lymph, tumor, jpg, save_path):

    lymph = load_npy_lymph(lymph)

    (h, w) = lymph.shape
    tumor = load_npy_tumor(tumor, (w, h))

    img = jpg
    img_copy = img.copy()
    img_shape = img_copy.shape
    contours_lymph, hierarchy_l = \
        cv2.findContours(
            lymph,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE)
    # contours_tumor, hierarchy_t = \
    #     cv2.findContours( \
    #         tumor, \
    #         cv2.RETR_TREE, \
    #         cv2.CHAIN_APPROX_NONE)

    for i, contour in enumerate(contours_lymph):

        if len(contour) < 250:
            continue
        try:
            M = cv2.moments(contours_lymph[i])
            x = int(M['m10'] / M['m00'])
            y = int(M['m01'] / M['m00'])
        except:
            continue
        cv2.drawContours(img_copy, contours_lymph, i, (0, 255, 255), 3)
        lymph_mask = make_tumor_mask(img_shape, [contours_lymph[i]])

        tumor_in_lymph = cv2.bitwise_and(lymph_mask, tumor)

        contours_and, hierarchy_and = \
            cv2.findContours(
                tumor_in_lymph,
                cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_NONE)
        for i_a, contour in enumerate(contours_and):
            if len(contour) > 20:

                cv2.drawContours(img_copy, contours_and,
                                 i_a, (255, 105, 65), 2)

        if hierarchy_and is None:
            # print('NoneType')
            continue
        else:

            hierarchy = np.squeeze(hierarchy_and)
            list_index = get_contours_index(hierarchy, len(contours_and))

            if (list_index is not None) and (len(list_index) > 0):
                for i_l in list_index:
                    if len(contours_and[i_l]) < 20:
                        continue
                    cv2.drawContours(
                        img_copy, contours_and[i_l], -1, (87, 139, 46), 2)
        tumor_in_lymph_img = make_tumor_mask(img_shape, contours_and)
        area = np.sum(tumor_in_lymph_img == 255) / \
            np.sum(lymph_mask == 255) * 100
        area = '{:.1f}'.format(area)

        # print('MLNA:' + area + '%')
        img_copy = cv2.putText(img_copy, area + '%',
                               (x, y), font, 1, (0, 0, 0), 2)
    cv2.imwrite(save_path, img_copy)
