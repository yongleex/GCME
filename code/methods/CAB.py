#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
This is the implementation of CAB method
Ref: Babakhani, Pedram, and Parham Zarei. "Automatic gamma correction based on average of brightness."
     Advances in Computer Science: an International Journal 4.6 (2015): 156-159.
Author: Yong Lee
E-Mail: yongli.cv@gmail.com
C-Data: 2019.04.10
"""
import cv2
import time
import numpy as np


def CAB(image, mask=None):
    """
    :param image:  input image, color (3 channels) or gray (1 channel);
    :param mask:  calc gamma value in the mask area, default is the whole image;
    :param normalize: normalize the input with max/min or not
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
        print("ERROR：check the input image of CAB function...")
        return None

    # Step 3. Main steps of CAB
    # Step 3.1 image normalization to range (0,1)
    img_n = (img + 0.5) / 256

    # Step 3.2 calculate the gamma
    if mask is not None:
        mask = mask < 128 
        img_n[mask] = np.NaN

    gamma = -np.log(2.0) / np.log(np.nanmean(img_n[:]))

    # Step 3.3  weather optimize for human visual system

    # Step 3.4 apply gamma transformation
    output = np.power(img_n, gamma)

    # Step 4.0 stretch back and post-process
    output = (output * 256 - 0.5).round().astype(np.uint8)
    if mask is not None:
        output[mask] = 0

    if color_flag:
        hsv[:, :, 2] = output
        output = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return gamma, output


# test function
def simple_example():
    image = cv2.imread(r"../../images/natural_image_sets/CBSD68/14037.png")
    visual = True

    start_time = time.time()
    gamma, output = CAB(image)
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
    plt.title("Before: CAB")
    plt.figure()
    plt.imshow(output[:, :, ::-1])
    plt.title("After: CAB")
    plt.show()


if __name__ == '__main__':
    simple_example()
