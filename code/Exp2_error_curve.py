#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Experiment 2: Test the gamma estimation accuracy in comparison with BIGC method
Author: Yong Lee
E-Mail: yongli.cv@gmail.com
C-Data: 2019.04.11
______________________________
version 2
M-Data: 2020.09.03
    1. Correct the bugs of AGT-ME
    2. Add CAB algorithm
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from methods.BIGC import BIGC
from methods.CAB import CAB
from methods.GCME import GCME
from methods.GCMP import GCMP
from methods.GCMV import GCMV

MAX_NUMBER = -1 # -1: all image employed; 2: accelerate for you to test it


# support function, add gamma distortion to an image
def gamma_trans(image, gamma):
    """
    Add gamma to an image
    """
    # Step 0. Check the inputs
    # -image
    if np.ndim(image) == 3 and image.shape[-1] == 3:  # color image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        img = hsv[:, :, 2]  # Add gamma distortion on V-channel of HSV colorspace
        color = True
    elif np.ndim(image) == 2:  # gray image
        img = image
        color = False
    else:
        print("ERROR：check the input image of AGT function...")
        return None

    # Step 1. Normalised the image to (0,1), and apply gamma transformation
    img = (img + 0.5) / 256
    img = np.power(img, gamma)
    img = np.clip(img * 256 - 0.5, 0, 255).round().astype(np.uint8)
    if color:
        hsv[:, :, 2] = img
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


# exp main function
def exp():
    # Step.0 Set the image dataset dir and a dir to store the results
    image_dir = r"../images/BSD68/"
    image_names = os.listdir(image_dir)[:MAX_NUMBER]  # Get the file name of the test images

    out_dir = r"./temp_out"

    # Step.1 the gamma set definition
    gamma_set = np.linspace(0.1, 3.0, 30)
    for gamma in gamma_set:  # save the distorted image for fun, Fig. 4 in the paper
        # path = os.path.join(image_dir, image_names[0])
        path = os.path.join(image_dir, 'test001.png')
        img = cv2.imread(path, -1)
        distorted_img = gamma_trans(img, gamma)
        cv2.imwrite(out_dir + os.sep + str(gamma) + ".png", distorted_img)

    # Step.2 get the estimated gamma with different methods
    methods = ['BIGC', 'CAB', 'GCMV', 'GCMP', 'GCME']
    gamma_estimated_list = []
    for gamma in gamma_set:
        gamma_estimated_list.append([])
        for k, name in enumerate(image_names):
            gamma_estimated_list[-1].append([])
            
            print(f"gamma:{gamma:.2f}; number:{k}-{name}")
            path = os.path.join(image_dir, name)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # read as gray image
            distorted_img = gamma_trans(img, gamma)  # Add gamma distortion

            for method in methods:
                gamma_origin, _ = eval(method)(img)  # Estimate the original gamma value
                gamma_estimated, _ =eval(method)(distorted_img)  # distorted gamma value
                gamma_estimated_list[-1][-1].append(gamma_origin / gamma_estimated)

    gamma_estimated_list = np.array(gamma_estimated_list)  # (3,2,5) with shape

    err = (gamma_estimated_list - gamma_set.reshape(-1,1,1))**2
    err[err>0.5] = np.NaN
    rmse = np.sqrt(np.nanmean(err, axis=1))
    rmse[rmse< 1e-3] = 1e-3

    # figure 1. actual gamma VS estimated gamma (GCME method)
    plt.figure()
    plt.subplots_adjust(bottom=0.14, left=0.14)
    plt.grid()
    gamma = np.repeat(gamma_set[:, np.newaxis], gamma_estimated_list.shape[1], axis=1)
    plt.scatter(gamma, gamma_estimated_list[:,:,-1], s=20, c='b')
    plt.xlim([-0.2, 3.2])
    plt.ylim([-0.2, 3.2])
    plt.xlabel("Gamma bias $\gamma_b$", fontsize=16)
    plt.ylabel("Recognized gamma $\gamma_r$", fontsize=16)
    plt.tick_params(labelsize=16)
    # plt.legend(fontsize=20)
    foo_fig = plt.gcf()
    foo_fig.savefig("figs/Fig5_a.pdf", format='pdf', dpi=1200)
    plt.title("set-estimate")

    # figure 2. RMSE error curve for different methods
    plt.figure()
    plt.grid()
    plt.subplots_adjust(bottom=0.14, left=0.18)
    print(gamma_set.shape, rmse.shape)
    # methods = ['BIGC', 'CAB', 'GCMV', 'GCMP', 'GCME']
    styles = ["-.", "--", ":", "--", "-"]
    colors = ['limegreen', 'mediumpurple','magenta','maroon','mediumblue']
    for k, method in enumerate(methods):
        plt.plot(gamma_set, rmse[:,k], styles[k], color=colors[k], label=method)
    plt.xlim([0.0, 3.0])
    plt.ylim([0.9e-3, 1.0])
    plt.semilogy()
    plt.xlabel("Gamma bias $\gamma_b$", fontsize=16)
    plt.ylabel("RMSE", fontsize=16)
    plt.tick_params(labelsize=16)
    plt.legend(fontsize=16)

    foo_fig = plt.gcf()
    foo_fig.savefig("figs/Fig5_b.pdf", format='pdf', dpi=1200)

    plt.show()


if __name__ == '__main__':
    exp()

