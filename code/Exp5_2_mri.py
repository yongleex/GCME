#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Experiment 5.2: Test the AGT-ME with spinal MRI images
Dataset link: https://www.kaggle.com/dutianze/mri-dataset
Author: Yong Lee
E-Mail: yongli.cv@gmail.com
C-Data: 2019.04.11
"""
import os
import cv2
import glob
import numpy as np
import matplotlib
from methods.BIGC import BIGC
from methods.CAB import CAB
from methods.GCME import GCME
from methods.GCMP import GCMP
from methods.GCMV import GCMV
from methods.metrics import m_entropy, m_contrast, m_std, m_EMEG, m_power

matplotlib.use('TkAgg')
from matplotlib import pylab as plt
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D


def read_show():
    example_filename = '../images/medical_image_sets/MRI/Case196.nii'

    img = nib.load(example_filename)
    print(img)
    print(img.header['db_name'])

    width, height, queue = img.dataobj.shape

    OrthoSlicer3D(img.dataobj).show()

    img_arr = img.dataobj[:, :, 5]
    plt.imshow(img_arr, cmap='gray')
    plt.show()


def ORIG(img,mask):
    return 1.0, img


def exp():
    image_dir = r"../images/medical_image_sets/MRI/"
    out_dir = r"./temp_out"
    [os.remove(path) for path in glob.glob(out_dir + "/*")]
    image_names = glob.glob(image_dir + "/*")

    for k, img_path in enumerate(image_names):
        name = img_path.split(os.sep)[-1].split('.')[0]

        # Step 1. read images
        img_raw = nib.load(img_path).dataobj[200:675, :, 5]
        img_raw = cv2.rotate(img_raw, cv2.ROTATE_90_COUNTERCLOCKWISE)

        mask = (img_raw > 100).astype(np.uint8) * 255
        img = (255 * (img_raw - np.min(img_raw[:])) / (np.max(img_raw[:]) - np.min(img_raw[:])))
        img = img.round().astype(np.uint8)

        methods = ['ORIG', 'BIGC', 'CAB', 'GCMV', 'GCMP', 'GCME']

        for method in methods:
            gamma, crt_img = eval(method)(img, mask=mask)
            m_ent, m_pw = m_entropy(crt_img, mask=mask), m_power(crt_img, mask=mask)
            m_cont, m_e = m_contrast(crt_img, mask=mask), m_EMEG(crt_img, mask=mask)
            info = f"{name}_{method:4s}_{m_ent:.4f}_{m_pw:8.5f}_{m_cont:8.4f}_{m_e:7.4f}_{gamma:7.4f}"
            # print(gamma)
            print(info)
            outpath = os.path.join(out_dir, info+'.png')
            cv2.imwrite(outpath, crt_img)

    os.system("nautilus " + out_dir)


if __name__ == '__main__':
    exp()
    # read_show() # show the MRI image sequence
