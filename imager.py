import cv2
import os
import pathlib

import matplotlib
import matplotlib.pyplot as plt

import io
import numpy as np

#tf.get_logger().setLevel('ERROR')
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
def create_gaborfilter():
    # from https://www.freedomvc.com/index.php/2021/10/16/gabor-filter-in-edge-detection/
    #Function is designed to produce a set of GaborFilters
    #an even distribution of theta values equally distributed amongst pi rad / 180 degrees.

    filters = []
    num_filters = 8
    ksize = 35
    sigma = 3.0
    lambd = 10.0
    gamma = 0.5
    psi = 0
    for theta in np.arange(0, np.pi, np.pi / num_filters):
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
        kern /= 1.0 * kern.sum() #Brightness normalization
        filters.append(kern)
    return filters

def apply_filter(img, filters):
    #General function to apply filters to our image
    #New Image
    newimage = np.zeros_like(img)

    #Starting with blank, loop through and apply gabor filt
    #on each it, take highest value until have max
    #final image is returned
    depth = -1
    for kern in filters: #Loop through the kernels in GaborFilter
        image_filter = cv2.filter2D(img, depth, kern)
        #Using numpy maximum to comp filter and cumulative image taking max
        np.maximum(newimage, image_filter, newimage)
    return newimage
    
imgPath = 'F:\Forza Images\OffRoad5.png'

img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
cv2.imshow('Forza',img)
grayIm = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grey Forza', grayIm)
gfilters = create_gaborfilter()
gabor_color = apply_filter(img, gfilters)
gabor_gray  = apply_filter(grayIm, gfilters)

cv2.imshow('Gabor Color', gabor_color)
cv2.imshow('Gabor Gray', gabor_gray)

gabor_color_grey = cv2.cvtColor(gabor_color, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gabor Color2Gray', gabor_color_grey)
cv2.waitKey(0)