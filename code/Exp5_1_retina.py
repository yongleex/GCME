# !/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Experiment 5.1: Test the AGT-ME with color images of the retina
Dataset link: http://www.isi.uu.nl/Research/Databases/DRIVE/
    The result images are saved in out_dir.
Author: Yong Lee
E-Mail: yongli.cv@gmail.com
C-Data: 2019.04.11
"""
import os
import cv2
import glob
import numpy as np

from methods.BIGC import BIGC
from methods.CAB import CAB
from methods.GCME import GCME
from methods.GCMP import GCMP
from methods.GCMV import GCMV
from methods.metrics import m_entropy, m_contrast, m_std, m_EMEG, m_power


def ORIG(img, mask=None):
    return 1.0, img


def exp():
    methods = ['ORIG', 'BIGC', 'CAB', 'GCMV', 'GCMP', 'GCME']

    image_dir = r"../images/medical_image_sets/DRIVE/images"
    mask_dir = r"../images/medical_image_sets/DRIVE/mask"
    out_dir = r"./temp_out/"
    [os.remove(path) for path in glob.glob(out_dir + "/*")]
    image_names = glob.glob(image_dir + "/*")

    gamma_list = []
    image_path_list = []
    for k, path in enumerate(image_names):
        # Step 1. read images
        name = path.split(os.sep)[-1].split('.')[0]
        mask_path = os.path.join(mask_dir, path.split(os.sep)[-1])
        img = cv2.imread(path)
        # img = np.round(255*(img-np.min(img[:])/(np.max(img[:])-np.min(img[:]))))
        # img = img.astype(np.uint8)
        mask = cv2.imread(mask_path, 0)
        if img is None or img.shape[2] == 4:  # something wrong to read an image, or BGRA image, or other files
            print("Warning: path (" + path + ") is not valid, we will skip this path...")
            continue
        _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        outpath = os.path.join(out_dir, f'{name}_v.png')
        cv2.imwrite(outpath, HSV[:,:,2])

        # Step 2. conduct gamma estimation and image restoration with different methods or config
        for method in methods:
            gamma, crt_img= eval(method)(img, mask=mask)
            m_ent, m_pw = m_entropy(crt_img, mask=mask), m_power(crt_img, mask=mask)
            m_cont, m_e = m_contrast(crt_img, mask=mask), m_EMEG(crt_img, mask=mask)

            info = f"{name}_{method:8s}_{m_ent:.4f}_{m_pw:8.5f}_{m_cont:8.4f}_{m_e:7.4f}_{gamma:6.4f}"
            print(info)

            outpath = os.path.join(out_dir, info+'.png')
            cv2.imwrite(outpath, crt_img)
            
    os.system("nautilus " + out_dir)


if __name__ == '__main__':
    exp()
