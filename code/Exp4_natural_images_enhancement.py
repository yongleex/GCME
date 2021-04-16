#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Experiment 4: Test the AGT-ME on the task of natural image contrast enhancement
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

def ORIG(img):
    return 1.0, img

def exp():
    # image_dir = r"../images/natural_image_sets" # full test
    image_dir = r"../images/natural_image" # test for paper images
    out_dir = r"./temp_out/"
    [os.remove(path) for path in glob.glob(out_dir + "/*")]
    image_names = glob.glob(image_dir + "/*/*")
    image_names = sorted(image_names)

    methods = ['ORIG', 'BIGC', 'CAB', 'GCMV', 'GCMP', 'GCME']

    for k, path in enumerate(image_names):
        # Step 1. read images
        img = cv2.imread(path)
        name = path.split(os.sep)[-1].split('.')[0]
        if img is None or img.shape[2] == 4:  # something wrong to read an image, or BGRA image, or other files
            print("Warning: path (" + path + ") is not valid, we will skip this path...")
            continue

        # Step 2. conduct gamma estimation and image restoration with different methods or config
        for method in methods:
            gamma, crt_img = eval(method)(img)
            m_ent, m_pw = m_entropy(crt_img), m_power(crt_img)
            m_cont, m_e = m_contrast(crt_img), m_EMEG(crt_img)
            info = f"{name}_{method:4s}_{m_ent:.4f}_{m_pw:8.5f}_{m_cont:8.4f}_{m_e:7.4f}"
            # print(gamma)
            print(info)
            outpath = os.path.join(out_dir, info+'.png')
            cv2.imwrite(outpath, crt_img)

        print('\n')

    os.system("nautilus " + out_dir)

if __name__ == '__main__':
    exp()
