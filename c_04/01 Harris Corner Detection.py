
"""
introduction:
https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html
Lecture 05 (ENG), pp 1-9 + 15
Corners are regions in the image with large variation in intensity in all the directions.
cv.cornerHarris basically finds the difference in intensity for a displacement of (u,v) in all directions.
A score determines if a window can contain a corner or not.
So the result of Harris Corner Detection is a grayscale image with these scores. 
Thresholding for a suitable score gives you the corners in the image.
"""
import numpy as np
import cv2 as cv

img = cv.imread('chessboard.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray = np.float32(gray)


"""
cornerHarris parameters:
Lecture 05 (ENG), pp 26-49
img - Input image. It should b-e grayscale and float32 type.
blockSize - It is the size of neighbourhood considered for corner detection - integer, more than 1
ksize - Aperture parameter of the Sobel derivative used -odd number
k - Harris detector free parameter in the equation - less than 0.2

dst is the Harris detector responses
"""
dst = cv.cornerHarris(gray,2,3,0.001)
dst = cv.dilate(dst,None)
#result is dilated for marking the corners, dst - score map with the height and width of the image

# Threshold for an optimal value - below 0.2 - it may vary depending on the image.
Threshold=0.01
img[dst>Threshold*dst.max()]=[0,255,0]
# if the x,y pixel of image has dst biggest then e.g. 0.01 max score, this is a corner, so we change the colour to e.g. green

cv.imshow('dst',img)
cv.waitKey()
cv.destroyAllWindows()
cv.imwrite("01_cornerHarris_img.jpg", img)
"""
TASKS:
- try a different Threshold below 0.2. With a small threshold, are there fewer or more corners?
- import chessboard.jpg and play with cornerHarris parameters to find corners only between bright and dark squares
*for saw1.jpg resize img: img_res = cv2.resize(img, (0, 0), fx = 0.1, fy = 0.1)
"""


img = cv.imread('chessboard.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = np.float32(gray)

dst = cv.cornerHarris(gray, 5, 3, 0.04)

dst = cv.dilate(dst, None)

Threshold = 0.002
img[dst > Threshold * dst.max()] = [0, 255, 0]

cv.imshow('Corners', img)
cv.waitKey(0)
cv.destroyAllWindows()

cv.imwrite("cornerHarris_result.jpg", img)