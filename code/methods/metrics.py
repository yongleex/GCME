import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def m_entropy(image,smooth=True, mask=None):
    # check the input image type 
    if np.ndim(image) == 3 and image.shape[-1] == 3:  # color image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        img = hsv[:, :, 2]
    elif np.ndim(image) == 2:  # gray image
        img = image
    else:
        print("ERROR：check the input image")
        return 1, None

    if mask is not None:
        mask = mask.copy()
        mask = mask>128

    if smooth:
        img = cv2.GaussianBlur(img, (3,3), 1.0)
        # img = cv2.blur(img,(3,3))

    hist = np.zeros(256)
    for i in range(256):
        if mask is not None:
            hist[i] = np.sum((img[:]==i)*(mask[:]))
            pdf = hist/np.sum(mask[:])
        else:
            hist[i] = np.sum((img[:]==i))
            pdf = hist/np.prod(img.shape[0:2])

    ent = -np.sum(np.log(pdf+1e-8)*pdf)
    return ent

def m_power(image,smooth=True, mask=None):
    """ Need a small smooth to overcome the entropy decrease barrier
    """
    # check the input image type 
    if np.ndim(image) == 3 and image.shape[-1] == 3:  # color image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        img = hsv[:, :, 2]
    elif np.ndim(image) == 2:  # gray image
        img = image
    else:
        print("ERROR：check the input image")
        return 1, None
     
    if mask is not None:
        mask = mask.copy()
        mask[mask < 255] = 0
        mask[mask > 0] = 1.0

    if smooth:
        img = cv2.GaussianBlur(img, (3,3), 1.0)

    hist = np.zeros(256)
    for i in range(256):
        if mask is not None:
            hist[i] = np.sum((img[:]==i)*(mask[:]))
        else:
            hist[i] = np.sum((img[:]==i))
    pdf = hist/np.prod(img.shape[0:2])
    # print(pdf)
    power= np.sum(np.power(pdf,2))
    return power

def m_contrast(image, mask=None):
    """ The Michelson contrast
    """
    # check the input image type 
    if np.ndim(image) == 3 and image.shape[-1] == 3:  # color image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        img = hsv[:, :, 2]
    elif np.ndim(image) == 2:  # gray image
        img = image
    else:
        print("ERROR：check the input image")
        return 1, None
    
    if mask is not None:
        mask = mask.copy()
        mask[mask < 255] = 0
        mask[mask > 0] = 1.0
        
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img_min = cv2.erode(img, kernel)
    img_max = cv2.dilate(img, kernel)

    c_michelson = (img_max-img_min)/(img_max+img_min+1e-3)
    if mask is not None:
        c_michelson[mask] = np.NaN 
    return np.nanmean(c_michelson)
    # return np.mean(np.log(c_michelson+1e-3))

def m_std(image, mask=None):
    # check the input image type 
    if np.ndim(image) == 3 and image.shape[-1] == 3:  # color image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        img = hsv[:, :, 2]
    elif np.ndim(image) == 2:  # gray image
        img = image
    else:
        print("ERROR：check the input image")
        return 1, None

    if mask is not None:
        mask = mask.copy()
        mask = mask<128
        img[mask] = np.NaN

    return np.nanstd(img[:])

def m_EMEG(image, mask=None):
    """ Metric from: Celik, Turgay. "Spatial entropy-based global and local image contrast enhancement." IEEE Transactions on Image Processing 23, no. 12 (2014): 5298-5308.
    """
    # check the input image type 
    if np.ndim(image) == 3 and image.shape[-1] == 3:  # color image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        img = hsv[:, :, 2]
    elif np.ndim(image) == 2:  # gray image
        img = image
    else:
        print("ERROR：check the input image")
        return 1, None

    if mask is not None:
        mask = mask.copy()
        mask = mask <128
        
    sobelx = np.abs(cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3))
    sobely = np.abs(cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8,8))
    x_min = cv2.erode(sobelx, kernel)
    x_max = cv2.dilate(sobelx, kernel)
    y_min = cv2.erode(sobely, kernel)
    y_max = cv2.dilate(sobely, kernel)

    emeg = np.maximum(x_max/(x_min+1.),y_max/(y_min+1.)) /255.0
    if mask is not None:
        emeg[mask] = np.NaN

    return np.nanmean(emeg[:])

# test function
def simple_example():
    image = cv2.imread(r"../../images/natural_image_sets/CBSD68/14037.png")

    # image = np.uint8(image/2)
    contrast = m_contrast(image)
    print(contrast)


if __name__ == '__main__':
    simple_example()

