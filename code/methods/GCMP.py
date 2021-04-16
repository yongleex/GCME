#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
This is the implementation of a baseline variant GCMP.
       gamma correction with minimum power
Author: Yong Lee
E-Mail: yongli.cv@gmail.com
C-Data: 2021.04.8
"""
import cv2
import time
import numpy as np
import scipy.optimize as op


def GCMP(image, mask=None):
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
        mask = mask > 128
        
    # Step 2. Main steps of GCMP
    # Step 2.1 obtain the pdf
    hist = np.zeros(256)
    for i in range(256):
        if mask is not None:
            hist[i] = np.sum((img[:]==i)*mask[:])
        else:
            hist[i] = np.sum(img[:]==i)

    pdf = hist/np.prod(img.shape[0:2])
    # print(pdf)

    # Step 2.2 calculate the gamma
    u = (np.arange(256)+0.5)/256
    
    pdf =pdf.reshape(-1, 1)
    u = u.reshape(-1, 1)
    def loss(gamma): # calculate the power loss with novel computing pipeline (using change of variable rule)
        return np.sum(np.power(pdf,2)*np.power(u, 1-gamma)/gamma, axis=0)

    # grid search for the optimal gamma
    gamma = np.arange(0.05, 8.0, 0.0005).reshape(1, -1)
    res = loss(gamma)
    ind = np.argmin(res)
    best_gamma = gamma[0, ind]
    # print(best_gamma)
    
    # Step 2.4 apply gamma transformation
    n_img = (img+0.5)/256
    output = np.power(n_img, best_gamma)

    # Step 3.0 stretch back and post-process
    # if mask is not None:
    #     output = (output * 256 - 0.5) * mask / 255.0
    # else:
    output = (output * 256 - 0.5)
    if mask is not None:
        output[~mask] = 0        
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
    gamma, output = GCMP(image)
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
    plt.title("Before: GCMP")
    plt.figure()
    plt.imshow(output[:, :, ::-1])
    plt.title("After: GCMP")
    plt.show()


if __name__ == '__main__':
    simple_example()
