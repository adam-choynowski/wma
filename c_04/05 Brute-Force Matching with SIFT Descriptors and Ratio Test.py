# more information https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
# reading images in grayscale
img1 = cv.imread("saw4.jpg", cv.IMREAD_GRAYSCALE) # queryImage
img1 = cv.resize(img1, (0, 0), fx = 0.5, fy = 0.5)
img2 = cv.imread("sawtraining1.png", cv.IMREAD_GRAYSCALE) # trainImage
img2 = cv.resize(img2, (0, 0), fx = 0.5, fy = 0.5)

# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# Brute-Force matcher 'BFMatcher' is simple.
# It takes the descriptor of one feature in first set
# and is matched with all other features in second set using some distance calculation.
# And the closest one is returned.
# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# If the distance to the best match is less than 75% of the distance to the second-best match, the match is considered good.
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()
cv.imwrite("05_BFMatcher_img3.jpg", img3)
"""
TASKS:
- find out what is the ratio test explained by D.Lowe in his paper (sec.7.1)
https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
The ratio test involves comparing the distance between the best and second-best matches for each descriptor.
If the distance to the best match is significantly smaller than the distance to the second-best match, the match is considered good.

"""

