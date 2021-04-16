#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

from methods.BIGC import BIGC
from methods.CAB import CAB
from methods.GCME import GCME
from methods.GCMP import GCMP
from methods.GCMV import GCMV
from methods.metrics import m_entropy, m_contrast, m_std, m_EMEG, m_power

def ORIG(img):
    return 1.0, img


def spectrum(image):
    y = np.fft.fftshift(np.fft.fft(image[line_index, 250:850].astype(np.float) - np.mean(image[line_index, 250:850])))
    spec = np.abs(np.sqrt(y * np.conjugate(y)))
    return spec


def line_value(image):
    return image[line_index, 250:850].astype(np.float)


# Step.0 path setting
image_dir = r"../images/slm image"
out_dir = r"./temp_out"
[os.remove(path) for path in glob.glob(out_dir + "/*")]

# Step.1 read images
path = glob.glob(image_dir + "/*")[0]
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # read as gray image

line_index = 50  # 200
name='fpp'
methods = ['ORIG', 'BIGC', 'CAB', 'GCMV', 'GCMP', 'GCME']
results = []
# Step 2. conduct gamma estimation and image restoration with different methods or config
info = f"name_method_entropy_power_contrast_EMEG_gamma"
print(info)
for method in methods:
    gamma, crt_img = eval(method)(img)
    results.append(crt_img)
    m_ent, m_pw = m_entropy(crt_img), m_power(crt_img)
    m_cont, m_e = m_contrast(crt_img), m_EMEG(crt_img)
    info = f"{name}_{method:4s}_{m_ent:.4f}_{m_pw:8.5f}_{m_cont:8.4f}_{m_e:7.4f}_{gamma:7.4f}"
    # print(gamma)
    print(info)
    temp = cv2.cvtColor(crt_img, cv2.COLOR_GRAY2RGB)
    temp = cv2.line(temp, (250, line_index), (850, line_index), (255, 0, 0), thickness=5)
    outpath = os.path.join(out_dir, info+'.png')
    cv2.imwrite(outpath,temp)

print('\n')

# Step.2 power spectrum analysis
power_spec = [spectrum(temp) for temp in results]
line_values = [line_value(temp) for temp in results]


# Step.3 display the curves (intensity curve and power spectrum)
styles = ["-", "-.", "--", ":", "--", "-"]
colors = ['lime', 'limegreen', 'mediumpurple','magenta','maroon','mediumblue']
plt.figure(figsize=(8, 3))
for line, style,color, label in zip(line_values, styles,colors, methods):
    plt.plot(np.arange(250, 850), line, style, color=color, label=label)
plt.xlim(250, 850)
plt.xlabel("n")
plt.ylabel("Intensity value")
plt.legend(loc='upper right')
plt.grid()
plt.subplots_adjust(bottom=0.16)
foo_fig = plt.gcf()
foo_fig.savefig("figs/Fig9_b.pdf", format='pdf', dpi=1200)

plt.figure(figsize=(8, 3))
for ps, style, color, label in zip(power_spec, styles,colors, methods):
    x = np.linspace(-len(ps) / 2, len(ps) / 2 - 1, len(ps))
    plt.plot(x, ps, style, color=color, label=label)
plt.xlim([0, 150])
plt.ylim([-1, 7E3])
plt.xlabel("frequency (Hz)")
plt.ylabel("power/frequency (dB/Hz)")
plt.legend(loc='upper right')
plt.subplots_adjust(bottom=0.16)
foo_fig = plt.gcf()
foo_fig.savefig("figs/Fig9_c.pdf", format='pdf', dpi=1200)
plt.show()

os.system("nautilus " + out_dir)
