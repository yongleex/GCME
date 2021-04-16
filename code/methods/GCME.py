#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
This is the implementation of our method AGT-ME, renamed as GCME.
Ref: https://arxiv.org/pdf/2007.02246.pdf
Author: Yong Lee
E-Mail: yongli.cv@gmail.com
C-Data: 2019.04.10
______________________________
version 2
M-Data: 2020.09.03
    1. Correct the bugs of AGT-ME
    2. Add CAB algorithm
"""
import cv2
import time
import numpy as np


def GCME(image, mask=None, normalize=False):
    """
    :param image:  input image, color (3 channels) or gray (1 channel);
    :param mask:  calc gamma value in the mask area, default is the whole image;
    :param normalize: normalize the input with max/min or not
    :param visual: for better visualization, we divide the  maximized entropy gamma with a constant 2.2
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

    # Step 2. pre-processing (optional)
    if normalize:  # max-min normalization
        img = img.astype(np.float)
        img = (255 * (img - np.min(img[:])) / (np.max(img[:]) - np.min(img[:]) + 0.1)).astype(np.float)

    # Step 3. Main steps of AGT-ME/ GCME
    # Step 3.1 to range (0,1)
    img = (img + 0.5) / 256

    # Step 3.2 calculate the gamma
    img_log = np.log(img)
    if mask is not None:
        mask = mask.copy()
        mask = mask<128
        img_log[mask] = np.NaN
    gamma = -1 / np.nanmean(img_log[:])
    # gamma = np.clip(gamma,0,10.0)

    # Step 3.3 apply gamma transformation
    output = np.power(img, gamma)

    # Step 4.0 stretch back and post-process
    output = (output*256-0.5)
    
    if mask is not None:
        output[mask] = 0
    output = output.round().astype(np.uint8)
    if color_flag:
        hsv[:, :, 2] = output
        output = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return gamma, output


# test function
def simple_example():
    image = cv2.imread(r"../../images/natural_image_sets/CBSD68/14037.png")

    start_time = time.time()
    gamma, output =GCME(image)
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
    plt.title("Before: GCME(AGT-ME)")
    plt.figure()
    plt.imshow(output[:, :, ::-1])
    plt.title("After: GCME(AGT-ME)")
    plt.show()


if __name__ == '__main__':
    simple_example()
