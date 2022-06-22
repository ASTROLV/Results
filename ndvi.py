from tkinter import image_names
import cv2
import numpy as np
from sklearn.feature_extraction import image
from fastiecm import fastiecm

park = cv2.imread('/home/pi/Desktop/015.png') # load image

#Function to modify the contrast of the image
def contrast_stretch(im):
    in_min = np.percentile(im, 5)
    in_max = np.percentile(im, 95)

    out_min = 0.0
    out_max = 255.0

    out = im - in_min
    out *= ((out_min - out_max) / (in_min - in_max))
    out += in_min

    return out

#Function to calculate the NDVI
def calc_ndvi(image):
    b, g, r = cv2.split(image)
    bottom = (r.astype(float) + b.astype(float))
    bottom[bottom==0] = 0.01
    ndvi = (b.astype(float) - r) / bottom
    #ndvi = (r.astype(float) - b) / bottom
    return ndvi


#Save the photos in the computer
contrasted = contrast_stretch(park)
cv2.imwrite('contrasted.png', contrasted)
ndvi = calc_ndvi(contrasted)
ndvi_contrasted = contrast_stretch(ndvi)
cv2.imwrite('ndvi_contrasted.png', ndvi_contrasted)
color_mapped_prep = ndvi_contrasted.astype(np.uint8)
color_mapped_image = cv2.applyColorMap(color_mapped_prep, fastiecm)
cv2.imwrite('color_mapped_image.png', color_mapped_image)
cv2.imwrite('ndvi.png', ndvi)
contrasted = contrast_stretch('original')
cv2.imwrite('contrasted.png', contrasted)
