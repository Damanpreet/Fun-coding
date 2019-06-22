'''Find connected components using Python OpenCV'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

connectivity = 4

rgbimg = cv2.imread('specify_img_path')
grayimg = cv2.cvtColor(rgbimg, cv2.COLOR_BGR2GRAY)
__, img = cv2.threshold(grayimg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imshow('original', rgbimg)
cv2.waitKey(0)
cv2.imshow('binary', img)
cv2.waitKey(0)

'''
return values:
1. total_labels - number of connected components in the image.
2. labels - label assigned to each component (0-background)
3. stats - In order: start of the bounding box around the object in horizontal direction, start of the bounding box around the object in vertical direction,
horizontal size of the bounding box, vertical size of the bounding box, total area (in pixels) of that component.
4. centroids - Centroid of each component.
'''
total_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)

print("Number of connected components in the image: ", total_labels-1)

print("Center of each component: ", centroids[1:])

print("Total area(in pixels): ", stats[1:, 4])
