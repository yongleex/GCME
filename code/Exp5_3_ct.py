#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Experiment 5.3: Test the AGT-ME with Abdominal CT images
Dataset link: https://www.kaggle.com/kmader/ct-scans-before-and-after
Author: Yong Lee
E-Mail: yongli.cv@gmail.com
C-Data: 2019.04.11
"""
import os
import cv2
import glob
import pydicom
import numpy as np

from methods.BIGC import BIGC
from methods.CAB import CAB
from methods.GCME import GCME
from methods.GCMP import GCMP
from methods.GCMV import GCMV
from methods.metrics import m_entropy, m_contrast, m_std, m_EMEG, m_power

def ORIG(img, mask):
    return 1.0, img



def exp():
    image_dir = r"../images/medical_image_sets/CT"
    out_dir = r"./temp_out"
    [os.remove(path) for path in glob.glob(out_dir + "/*")]
    image_names = glob.glob(image_dir + "/*")

    methods = ['ORIG', 'BIGC', 'CAB', 'GCMV', 'GCMP', 'GCME']

    for k, img_path in enumerate(image_names):
        name =img_path.split(os.sep)[-1].split('.')[0]
                
        print("processing " + img_path + "...")
        # Step 1. read CT images with pydicom package
        dcm = pydicom.read_file(img_path)
        dcm.image = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
        img_raw = dcm.image.copy()

        if img_raw is None:  # something wrong to read an image, or BGRA image
            print("Warning: path (" + img_path + ") is not valid, we will skip this path...")
            continue

        # Step 2. Get the mask with a simple threshold strategy and normalize it to 0-1
        mask_threshold = -150
        mask = ((img_raw > mask_threshold) * 255).astype(np.uint8)
        img = 255 * (img_raw - mask_threshold) / (np.max(img_raw[:]) - mask_threshold)
        img[img < 0] = 0
        img = img.round().astype(np.uint8)

        for method in methods:
            gamma, crt_img = eval(method)(img, mask=mask)
            m_ent, m_pw = m_entropy(crt_img, mask=mask), m_power(crt_img, mask=mask)
            m_cont, m_e = m_contrast(crt_img, mask=mask), m_EMEG(crt_img, mask=mask)
            info = f"{name}_{method:4s}_{m_ent:.4f}_{m_pw:8.5f}_{m_cont:8.4f}_{m_e:7.4f}_{gamma:7.4f}"
            print(info)
            
            outpath = os.path.join(out_dir, info+'.png')
            cv2.imwrite(outpath, crt_img)
        print('\n')

    print("please check the results in dir:" + out_dir)
    os.system("nautilus " + out_dir)


if __name__ == '__main__':
    exp()
