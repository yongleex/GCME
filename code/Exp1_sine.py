#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Experiment 1: A Simple 1D example to analysis the AGT-ME cost function
Author: Yong Lee
E-Mail: yongli.cv@gmail.com
C-Data: 2019.04.10
"""
import numpy as np
import matplotlib.pyplot as plt
from methods.BIGC import BIGC
from methods.CAB import CAB
from methods.GCME import GCME
from methods.GCMP import GCMP
from methods.GCMV import GCMV

# Add gamma distortion to an image or signal
def gamma_trans(image, gamma):
    img = (image.astype(np.float32) + 0.5) / 256  # Normalized to range (0,1)
    img = np.power(img, gamma)  # Apply gamma distortion
    img = np.clip(img * 256 - 0.5, 0, 255)  # change the range to [0,255]
    return img


# Get the negative entropy cost / loss for a given signal
# def get_j(img):
def negative_entropy_cost(img):
    p = np.zeros(256, dtype=np.float)
    img = img.astype(np.uint8)
    cost = 0
    for i in range(256):
        p[i] = np.sum(img == i) / np.size(img)  # Get the histogram frequency with intensity level i
        cost = cost + p[i] * np.log(p[i] + 1e-10)  # Get the negative entropy loss of intensity value i
    return cost


# Get the negative entropy cost J(\gamma) value in AGT-ME method
# def get_cost(img, gamma):
def predict_negative_entropy_cost(img, gamma):
    p = np.zeros(256, dtype=np.float)
    img = img.astype(np.uint8)
    cost = 0
    for i in range(256):
        p[i] = np.sum(img == i) / np.size(img)  # Get the histogram frequency
        # Get the negative entropy loss, ref Eq.(7)
        cost = cost + p[i] * np.log(1e-10 + p[i] * ((i + 0.5) / 256.0) ** (1 - gamma) / gamma)
    return cost


# Get the bi-coherence cost value in BIGC method
def get_bi_coherence_cost(image):
    if np.ndim(image) != 2:
        print("The input image is wrong")
        return False

    # Step.1 get scan lines
    scan_lines = image[::10, :64]
    for i in range(np.floor((image.shape[1] - 65) / 32).astype(np.int)):
        scan_lines = np.concatenate((scan_lines, image[::10, 32 * i:32 * i + 64]), axis=0)

    # Step.2 perform DFT operation
    fft_lines = np.fft.fft(scan_lines)

    # Step.3 calc the cost, eq(12) in the paper
    cost = 0
    for i1, w1 in enumerate(range(-3, 3)):  # w1 in range [-pi,pi]
        for i2, w2 in enumerate(range(-3, 3)):  # w2 in range [-pi,pi]
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
            s1 = s1[np.newaxis, :]
            s2 = s2[np.newaxis, :]
            b_w1_w2 = np.abs(np.mean(s1 * s2)) / (
                    np.sqrt(np.mean(np.abs(s1) * np.abs(s1))) * np.sqrt(np.mean(np.abs(s2) * np.abs(s2))))
            cost = cost + b_w1_w2  # add the bi-coherence for all frequency
    return cost


def exp():
    # Step 1. synthesis the 1-d signal
    n = np.linspace(0, 511, 512)
    y = 75 * np.sin(2 * n * 2 * np.pi / 64) - 55 * np.sin(1.3 * n * 2 * np.pi / 64) + 127
    y_float = np.clip(y, 0, 255).astype(np.float32)
    y_uint8 = np.clip(y, 0, 255).astype(np.uint8)

    # Step 2. apply gamma distorsion
    gamma = 1.5
    y_float_d = gamma_trans(y_float, gamma).astype(np.float32)
    y_uint8_d = gamma_trans(y_uint8, gamma).astype(np.uint8)

    # Step 3. gamma estimation for 4 method, except BIGC  
    methods = ['CAB','GCMV','GCMP','GCME']
    g_res = []
    for method in methods:
        gamma_e, _ = eval(method)(y_uint8_d.reshape(-1,1))
        info = f"{method}: Truth:{gamma:.4f}\t {1/gamma_e:.4f}"
        print(info)
        g_res.append(gamma_e)

    # Step 4. get the gamma value with BIGC method (brute force search)
    gamma_list = np.linspace(0.1, 3.0, 300)
    BIGC_cost = np.zeros_like(gamma_list)
    for k, g in enumerate(gamma_list):
        img = (gamma_trans(y_uint8_d, 1 / g)[np.newaxis, :]).astype(np.uint8)
        BIGC_cost[k] = get_bi_coherence_cost(img)
    gamma_BIGC = gamma_list[np.argmin(BIGC_cost)]
    info = f"BIGC: Truth:{gamma:.4f}\t {gamma_BIGC:.4f}"
    print(info)
    g_res.append(1/gamma_BIGC)

    # Step 5. The negative entropy change with different gamma
    gamma_list = np.linspace(0.1, 3.0, 300)
    discrete_entropy_cost = np.zeros_like(gamma_list)
    continuous_entropy_cost = np.zeros_like(gamma_list)
    entropy_smooth_cost = np.zeros_like(gamma_list)
    for k, gamma in enumerate(gamma_list):
        img = gamma_trans(y_uint8_d, 1 / gamma)[np.newaxis, :]
        discrete_entropy_cost[k] = negative_entropy_cost(img)
        img = (gamma_trans(y_float_d, 1 / gamma)[np.newaxis, :]).astype(np.uint8)
        continuous_entropy_cost[k] = negative_entropy_cost(img)
        entropy_smooth_cost[k] = predict_negative_entropy_cost(y_uint8_d, 1.0 / gamma)

    # Figure 1a. 1-D signal and the distorted curve
    plt.figure()
    plt.subplots_adjust(bottom=0.14)
    plt.plot(n, (y + 0.5) / 256, "k-", label="y(n)")
    plt.plot(n, (y_uint8_d + 0.5) / 256, "b:", label="$y^{1.5}(n)$")
    plt.xlim([0, 120])
    plt.ylim([0, 1])
    plt.xlabel("n", fontsize=16)
    plt.ylabel("y", fontsize=16)
    plt.tick_params(labelsize=16)
    plt.legend(fontsize=16)
    foo_fig = plt.gcf()
    foo_fig.savefig("figs/Fig3_a.pdf", format='pdf', dpi=1200)
    plt.title("Synthetic curve")

    # Figure 1b. 1-D signal and restored curves with BIGC and AGT-ME
    plt.figure()
    plt.subplots_adjust(bottom=0.14)
    plt.plot(n, (y + 0.5) / 256, "k-", label="y(n)")
    plt.plot(n, (y_uint8_d + 0.5) / 256, "b:", label="$y^{1.5}(n)$")
    plt.plot(n, ((y_uint8_d + 0.5) / 256) ** g_res[-1], "-.", color='limegreen', label="BIGC")
    plt.plot(n, ((y_uint8_d + 0.5) / 256) ** g_res[0], "--", color='mediumpurple', label="CAB")
    plt.plot(n, ((y_uint8_d + 0.5) / 256) ** g_res[3], "-", color='mediumblue', label="GCME")
    plt.xlim([70, 110])
    plt.ylim([0.4, 0.8])
    plt.xlabel("n", fontsize=16)
    plt.ylabel("y", fontsize=16)
    plt.tick_params(labelsize=16)
    plt.legend(fontsize=16)
    foo_fig = plt.gcf()
    foo_fig.savefig("figs/Fig3_b.pdf", format='pdf', dpi=1200)
    plt.title("Restore curve")

    # Figure 1c. BIGC loss curve
    plt.figure()
    plt.subplots_adjust(bottom=0.14)
    plt.plot(gamma_list, np.log(BIGC_cost), "-.", color='limegreen', label="BIGC")
    plt.xlim([0, 3])
    plt.xlabel("Reciprocal of correction gamma", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.tick_params(labelsize=16)
    plt.yticks([])
    plt.grid()
    plt.legend(fontsize=16)
    foo_fig = plt.gcf()
    foo_fig.savefig("figs/Fig3_c.pdf", format='pdf', dpi=1200)
    plt.title("BIGC Loss")

    # Figure 1d. AGT-ME loss curve
    plt.figure()
    plt.subplots_adjust(bottom=0.14)
    # plt.plot(gamma_list, discrete_entropy_cost, "--", color="midnightblue", label="$L^*$")
    # plt.plot(gamma_list, continuous_entropy_cost, label="$-H_c(y^{1.5\gamma}(n))$")
    plt.plot(gamma_list, entropy_smooth_cost, "-", color='mediumblue', label="$J(\gamma)$ from Eq.(7)")
    plt.xlim([0.5, 3])
    plt.ylim([-5.3, -4.3])
    plt.xlabel("Reciprocal of correction gamma", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.tick_params(labelsize=16)
    plt.yticks([])
    plt.grid()
    plt.legend(fontsize=16)
    foo_fig = plt.gcf()
    foo_fig.savefig("figs/Fig3_d.pdf", format='pdf', dpi=1200)
    plt.title("GCME Loss")

    plt.yticks([])
    plt.show()


if __name__ == '__main__':
    exp()
