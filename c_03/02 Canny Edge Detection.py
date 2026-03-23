
import numpy as np
import cv2 as cv
import sys
from matplotlib import pyplot as plt

#import image
img = cv.imread('warsaw.jpg', cv.IMREAD_GRAYSCALE)
if img is None:
    sys.exit("Could not read the image.")

#https://en.wikipedia.org/wiki/Canny_edge_detector
edges = cv.Canny(img, threshold1 = 100, threshold2 = 200, apertureSize = 3)#The aperture size defines the dimensions of the kernel used for convolution

plt.subplot(1,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()


"""
TASKS:

- change parameters:
threshold1 (first threshold for the hysteresis procedure),
and threshold2 (second threshold for the hysteresis procedure).
note : apertureSize is aperture size for the Sobel operator (3, 5 or 7).

edges_v1 = cv.Canny(img, threshold1 = 100, threshold2 = 200)
edges_v2 = cv.Canny(img, threshold1 = 100, threshold2 = 300)
edges_v3 = cv.Canny(img, threshold1 = 200, threshold2 = 400)
edges_v4 = cv.Canny(img, threshold1 = 200, threshold2 = 600)
plt.subplot(2,2,1),plt.imshow(edges_v1,cmap = 'gray')
plt.title('edges_v1'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(edges_v2,cmap = 'gray')
plt.title('edges_v2'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(edges_v3,cmap = 'gray')
plt.title('edges_v3'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(edges_v4,cmap = 'gray')
plt.title('edges_v4'), plt.xticks([]), plt.yticks([])
plt.show()
"""
edges_v1 = cv.Canny(img, threshold1=100, threshold2=200)
edges_v2 = cv.Canny(img, threshold1=100, threshold2=300)
edges_v3 = cv.Canny(img, threshold1=200, threshold2=400)
edges_v4 = cv.Canny(img, threshold1=200, threshold2=600)

plt.figure(figsize=(8,8))

plt.subplot(2,2,1)
plt.imshow(edges_v1, cmap='gray')
plt.title('100 / 200')
plt.xticks([]); plt.yticks([])

plt.subplot(2,2,2)
plt.imshow(edges_v2, cmap='gray')
plt.title('100 / 300')
plt.xticks([]); plt.yticks([])

plt.subplot(2,2,3)
plt.imshow(edges_v3, cmap='gray')
plt.title('200 / 400')
plt.xticks([]); plt.yticks([])

plt.subplot(2,2,4)
plt.imshow(edges_v4, cmap='gray')
plt.title('200 / 600')
plt.xticks([]); plt.yticks([])

plt.show()

#change aperture size

edges3 = cv.Canny(img, 100, 200, apertureSize=3)
edges5 = cv.Canny(img, 100, 200, apertureSize=5)
edges7 = cv.Canny(img, 100, 200, apertureSize=7)

plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.imshow(edges3, cmap='gray')
plt.title('aperture=3')

plt.subplot(1,3,2)
plt.imshow(edges5, cmap='gray')
plt.title('aperture=5')

plt.subplot(1,3,3)
plt.imshow(edges7, cmap='gray')
plt.title('aperture=7')

for i in range(1,4):
    plt.subplot(1,3,i)
    plt.xticks([]); plt.yticks([])

plt.show()


"""
- Smooth image with a Gaussian filter

src = cv.GaussianBlur(img, (7,7),0)
# https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1
edges_v1_src = cv.Canny(src, threshold1 = 100, threshold2 = 200)
edges_v1_img = cv.Canny(img, threshold1 = 100, threshold2 = 200)
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('img'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(src,cmap = 'gray')
plt.title('src'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(edges_v1_img,cmap = 'gray')
plt.title('edges_v1_img'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(edges_v1_src,cmap = 'gray')
plt.title('edges_v1_src'), plt.xticks([]), plt.yticks([])
plt.show()

"""
src = cv.GaussianBlur(img, (7,7), 0)

edges_img = cv.Canny(img, 100, 200)
edges_src = cv.Canny(src, 100, 200)

plt.figure(figsize=(8,8))

plt.subplot(2,2,1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.xticks([]); plt.yticks([])

plt.subplot(2,2,2)
plt.imshow(src, cmap='gray')
plt.title('Gaussian Blur')
plt.xticks([]); plt.yticks([])

plt.subplot(2,2,3)
plt.imshow(edges_img, cmap='gray')
plt.title('Edges (no blur)')
plt.xticks([]); plt.yticks([])

plt.subplot(2,2,4)
plt.imshow(edges_src, cmap='gray')
plt.title('Edges (blur)')
plt.xticks([]); plt.yticks([])

plt.show()
