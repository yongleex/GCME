#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
This is the implementation of a baseline method GCMV.
Ref: Mahamdioua, Meriama, and Mohamed Benmohammed. "New mean-variance gamma method for automatic gamma correction." International Journal of Image, Graphics and Signal Processing 9, no. 3 (2017) 
Author: Yong Lee
E-Mail: yongli.cv@gmail.com
C-Data: 2021.4.8
"""
import cv2
import time
import numpy as np
import scipy.optimize as op


def GCMV(image, mask=None):
    """
    :param image:  input image, color (3 channels) or gray (1 channel);
    :param mask:  calc gamma value in the mask area, default is the whole image;
    :return: gamma, and output
    """

    # Step 1. Check the inputs: image
    if np.ndim(image) == 3 and image.shape[-1] == 3:  # color image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        img = hsv[:, :, 2]
        color_flag = True
    elif np.ndim(image) == 2:  # gray image
        img = image
        color_flag = False
    else:
        print("ERROR：check the input image of AGT function...")
        return 1, None

    if mask is not None:
        mask = mask<255
    else:
        mask = np.ones_like(img)
        
    # Step 2. Main steps of GCMV
    n_img = img/255.0
    mean = np.mean(n_img) 
    gamma_list = np.arange(0.01,1.01,0.01) if mean<=0.5 else np.arange(1.1,10.1,0.1)
    
    score = np.zeros_like(gamma_list)
    for k, gamma in enumerate(gamma_list):
        t_img = np.power(n_img, gamma)
        m1, v1 = np.mean(t_img, axis=0), np.var(t_img, axis=0)
        m2, v2 = np.mean(t_img, axis=1), np.var(t_img, axis=1)
        score[k] = np.mean(np.power(m1-0.5077,2)) + np.mean(np.power(m2-0.5077,2))+np.mean(np.power(v1-0.0268,2)) + np.mean(np.power(v2-0.0268,2))

    # grid search for the optimal gamma
    ind = np.argmin(score)
    best_gamma =gamma_list[ind]
    # print(best_gamma)
    
    # Step 2.4 apply gamma transformation
    n_img = (img+0.5)/256
    output = np.power(n_img, best_gamma)

    # Step 3.0 stretch back and post-process
    # if mask is not None:
    #     output = (output * 256 - 0.5) * mask / 255.0
    # else:
    output = (output * 256 - 0.5)
    output = output.round().astype(np.uint8)
    if color_flag:
        hsv[:, :, 2] = output
        output = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return best_gamma, output


# test function
def simple_example():
    image = cv2.imread(r"../../images/natural_image_sets/CBSD68/14037.png")
    visual = False

    start_time = time.time()
    gamma, output = GCMV(image)
    end_time = time.time()

    print("Estimated gamma =" + str(gamma) + ", with time cost=" + str(end_time - start_time) + "s")
    # cv2.namedWindow("input", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    # cv2.imshow("input", image)
    # cv2.imshow("output", output)
    # cv2.waitKey()
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(image[:, :, ::-1])
    plt.title("Before: GCMV")
    plt.figure()
    plt.imshow(output[:, :, ::-1])
    plt.title("After: GCMV")
    plt.show()


if __name__ == '__main__':
    simple_example()
