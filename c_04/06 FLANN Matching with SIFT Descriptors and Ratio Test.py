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
# FLANN parameters
# LANN (Fast Library for Approximate Nearest Neighbors) is a library used for quickly finding approximate nearest neighbors in large datasets. 
FLANN_INDEX_KDTREE = 1#This line defines the type of FLANN index to be used, indicating the use of a k-d tree structure for accelerating nearest neighbor searches.
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)#creates a dictionary containing FLANN index parameters. The trees parameter determines the number of trees in the k-d tree structure.
search_params = dict(checks=50)#FLANN search parameters. The checks parameter controls the number of checks FLANN performs during the search. A higher number of checks may increase accuracy but can also slow down the process.   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params,search_params) #initializes a FLANN-based Matcher object with the specified index and search parameters.
matches = flann.knnMatch(des1,des2,k=2)#to search for the two best matches for each feature described by descriptors des1 and des2 from the images.
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()
cv.imwrite("06_FLANNMatcher_img3.jpg", img3)
"""
TASKS:
- compare BFMatcher and FLANN. Try to optimize methods by changing parameters.
- as the queryImage import man.jpg, and as the trainImage import people.jpg.
- try man_cold.jpg and man_rotated.jpg as the query image
- try messi.jpg and footballers.jpg. How to improve the matching?
- see more examples : https://datahacker.rs/feature-matching-methods-comparison-in-opencv/
"""

sift = cv.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

index_params = dict(algorithm=1, trees=8)
search_params = dict(checks=100)

flann = cv.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append([m])

# sortowanie
good = sorted(good, key=lambda x: x[0].distance)[:50]