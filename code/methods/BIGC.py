#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
This is our implementation of ref method BIGC.
Ref: H. Farid, “Blind inverse gamma correction,” IEEE TIP, vol. 10, no. 10, pp. 1428–1433, 2001.
Author: Yong Lee
E-Mail: yongli.cv@gmail.com
C-Data: 2019.04.10
"""
import cv2
import time
import numpy as np


# Funcs: add gamma to an image
def gamma_trans(image, gamma):
    # Step 0. Check the inputs
    if np.ndim(image) == 3 and image.shape[-1] == 3:  # color image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        img = hsv[:, :, 2]
        color = True
    elif np.ndim(image) == 2:  # gray image
        img = image
        color = False
    else:
        print("ERROR：check the input image of AGT function...")
        return 1, None

    # Step 1. apply gamma transform
    img = (img + 0.5) / 256
    img = np.power(img, gamma)
    img = np.clip(img * 256 - 0.5, 0, 255).astype(np.uint8)
    if color:
        hsv[:, :, 2] = img
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def get_bi_coherence_cost(image, scanline_flag=False):
    """
        Calc the cost in Eq.(12) of "Blind Inverse Gamma Correction"

    """
    if np.ndim(image) != 2:  # only for gray image
        print("The input image is wrong")
        return False

    # Step.1 get scan lines. total lines is 16*((w-64)/32)
    if scanline_flag:
        scan_lines = image
    else:
        scan_lines = image[::np.floor(image.shape[0] / 16).astype(np.int), :64]
        for i in range(1, np.floor((image.shape[1] - 65) / 32).astype(np.int)):
            scan_lines = np.concatenate(
                (scan_lines, image[::np.floor(image.shape[0] / 16).astype(np.int), 32 * i:32 * i + 64]), axis=0)

    # Step.2 fft transform for all lines
    fft_lines = np.fft.fft(scan_lines)

    # Step.3 get the cost for the given image
    cost = 0
    for w1 in range(-3, 3):  # w1 in range [-pi,pi]
        for w2 in range(-3, 3):  # w2 in range [-pi,pi]
            # Calc the b(w1,w2)
            s1, s2 = [], []
            for fft in fft_lines:
                Yk_1 = fft[np.abs(w1)]
                Yk_2 = fft[np.abs(w2)]
                Yk_12 = fft[np.abs(w1 + w2)]
                s1.append(Yk_1 * Yk_2)
                s2.append(np.conjugate(Yk_12))
            s1 = np.array(s1)
            s2 = np.array(s2)
            b_w1_w2 = np.abs(np.mean(s1 * s2)) / (
                    np.sqrt(np.mean(np.abs(s1) * np.abs(s1))) * np.sqrt(np.mean(np.abs(s2) * np.abs(s2))))
            cost = cost + b_w1_w2  # add the bi-coherence in the frequency range
    return cost


def BIGC(image, method="fast", mask=None):
    """
    This is our main implementation of work Farid, "Blind Inverse Gamma Correction".
    :param image:  input image, color (3 channels) or gray (1 channel);
    :param method:  "fast" and "bruteforce" available
    :return: gamma, and result
    """
    # Check the inputs
    if np.ndim(image) == 3 and image.shape[-1] == 3:  # color image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        img = hsv[:, :, 2]
        color = True
    elif np.ndim(image) == 2:  # gray image
        img = image
        color = False
    else:
        print("ERROR：check the input image of AGT function...")
        return 1, None

    # Brute force search in the gamma_list
    if method == "bruteforce":
        gamma_list = np.linspace(0.1, 10.0, 100)  # 0.01 error due to search policy
        cost = np.zeros_like(gamma_list)
        for k, gamma in enumerate(gamma_list):
            img_temp = gamma_trans(img, gamma)
            cost[k] = get_bi_coherence_cost(img_temp)  # calc the cost for each gamma
        gamma = gamma_list[np.argmin(cost)]  # get the gamma with the minimal cost
        # plt.figure()
        # plt.plot(gamma_list, cost)
        # plt.show()

    # fast search method
    if method == "fast":
        # get the scan-lines at first step
        scan_lines = img[::np.floor(img.shape[0] / 16).astype(np.int), :64]
        for i in range(1, np.floor((img.shape[1] - 65) / 32).astype(np.int)):
            scan_lines = np.concatenate(
                (scan_lines, img[::np.floor(img.shape[0] / 16).astype(np.int), 32 * i:32 * i + 64]), axis=0)

        gamma0 = 0.01
        gamma1 = 4.5
        gamma2 = 9

        for i in range(11):  # 0.003 error due to search policy
            if get_bi_coherence_cost(gamma_trans(scan_lines, 0.5 * gamma0 + 0.5 * gamma1),
                                     True) < get_bi_coherence_cost(gamma_trans(scan_lines, 0.5 * gamma1 + 0.5 * gamma2),
                                                                   True):
                gamma0, gamma1, gamma2 = gamma0, 0.5 * (gamma0 + gamma1), gamma1
            else:
                gamma0, gamma1, gamma2 = gamma1, 0.5 * (gamma1 + gamma2), gamma2
        gamma = gamma1
        # print(gamma)

    # Step 3.0 apply gamma transformation
    result = gamma_trans(img, gamma)
    if color:
        hsv[:, :, 2] = result
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return gamma, result


# test function
def simple_example():
    image = cv2.imread(r"../../images/natural_image_sets/CBSD68/14037.png")

    start_time = time.time()
    gamma, output = BIGC(image, "fast")
    end_time = time.time()

    print("Estimated gamma =" + str(gamma) + ", with time cost=" + str(end_time - start_time) + "s")
    # cv2.namedWindow("input", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    # cv2.imshow("input", image)
    # cv2.imshow("output", result)
    # cv2.waitKey()

    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(image[:, :, ::-1])
    plt.title("Before:BIGC")
    plt.figure()
    plt.imshow(output[:, :, ::-1])
    plt.title("After:BIGC")
    plt.show()


if __name__ == '__main__':
    simple_example()
