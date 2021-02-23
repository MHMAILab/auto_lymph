import cv2
import numpy as np
from openslide import OpenSlide


def save_slide_cutting(
        file_path,
        multiple,
        # save_path
):
    slide = OpenSlide(file_path)
    slide_downsamples = slide.get_best_level_for_downsample(multiple)
    downsample = slide.level_downsamples[slide_downsamples]
    w_lv_, h_lv_ = slide.level_dimensions[slide_downsamples]
    wsi_pil_lv_ = slide.read_region(
        (0, 0),
        slide_downsamples,
        (w_lv_, h_lv_))
    wsi_ary_lv_ = np.array(wsi_pil_lv_)
    wsi_bgr_lv_ = cv2.cvtColor(wsi_ary_lv_, cv2.COLOR_RGBA2BGR)

    downsample = multiple / downsample
    w = int(w_lv_ / downsample)
    h = int(h_lv_ / downsample)
    img = cv2.resize(wsi_bgr_lv_, (w, h), interpolation=cv2.INTER_LINEAR)
    # cv2.imwrite(img, save_path)
    return img
