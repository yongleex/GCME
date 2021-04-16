#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Experiment 3: Test the gamma estimation efficiency in comparison with BIGC method
Author: Yong Lee
E-Mail: yongli.cv@gmail.com
C-Data: 2019.04.11
"""
import os
import cv2
import time
import numpy as np
# from methods.AGT import adaptive_gamma_transform
# from methods.BIGC import blind_inverse_gamma_correction
# from methods.CAB import correct_average_brightness
from methods.BIGC import BIGC
from methods.CAB import CAB
from methods.GCME import GCME
from methods.GCMP import GCMP
from methods.GCMV import GCMV


methods = ['BIGC', 'CAB', 'GCMV', 'GCMP', 'GCME']
# exp main function
def exp():
    # Step.0 Set the image path
    image_dir = r"../images/BSD68"
    file_list = os.listdir(image_dir)[0:10] # test on 10 images 

    size_list = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]
    time_list = []
    for k, name in enumerate(file_list):
        time_list.append([])
        image = cv2.imread(os.path.join(image_dir, name), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        for siz in size_list:
            time_list[-1].append([])
            img = cv2.resize(image, siz)  # Resize the images

            for method in methods:
                start_time = time.time()
                _, _ =eval(method)(img)
                end_time = time.time()
                cost = end_time - start_time
                time_list[-1][-1].append(cost)
                print(f"{k}/{len(file_list)}:{name}, {siz}\t{method}:\t{1000*cost:8.2f} ms")

    time_list = np.array(time_list) # 2*4*5 shape 
    
    # show the results
    for j, siz in enumerate(size_list):
        for k, method in enumerate(methods):
            c_mean = np.mean(time_list[:,j,k])*1000
            c_std = np.std(time_list[:,j,k])*1000
            info = f"size:{siz[0]:4d}, method:{method:5s}, time:({c_mean:8.2f},{c_std:8.2f}) ms"
            print(info)


if __name__ == '__main__':
    exp()
