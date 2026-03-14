#################################################################
# more info:
# https://www.geeksforgeeks.org/python-opencv-morphological-operations/
#################################################################
# %% '''complete the code'''
# import libraries
import cv2
import sys
import numpy as np

# %%
# imshow function
def imshow(title, image):
    cv2.imshow(title,image)
    k = cv2.waitKey(0)#0 infinity waiting time
    if k == ord("s"):
        cv2.imwrite(f"00_{title}_saved.jpg", image)
    cv2.destroyAllWindows()
    return

# %%
# Upload image using cv2.imread in GRAYSCALE
# we need grayscale for binarize the image
img = cv2.imread("letters2.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    sys.exit("Could not read the image.")
imshow("Original_img", img)

# %%
# binarize the image and invert the image
# cv2.threshold () a function used for thresholding grayscale images,
# separating regions of interest based on pixel intensity values.
# Pixels with intensity values above threshold are set to a maximum value (usually 255),
# rest are set to a minimum value (usually 0).
# Binary thresholding are set to a maximum value (usually 255), and a minimum value (usually 0).
# Otsu's thresholding is an automatic thresholding technique maximizing the inter-class variance
# more : https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
# binr = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)[1]
binr = cv2.threshold(img, 1, 255, cv2.THRESH_OTSU)[1]
# Consider an image with only two distinct image values (bimodal image),
# where the histogram would only consist of two peaks.
# A good threshold would be in the middle of those two values.
# Similarly, Otsu's method determines an optimal global threshold value from the image histogram.
imshow("binr", binr)

# %%
# # bitwise inversion of an image
invert = cv2.bitwise_not(binr)
imshow("invert", invert)

# note: white areas have value 255 (1), and black areas have 0 (0),
# thus the erosion operation erodes the white 255 (1) areas (makes them smaller).
# Dilatation works in opposite direction.
# %%
# Erosion
"""
Erosion primarily involves eroding the outer surface (the foreground) of the image.
As binary images only contain two pixels 0 and 255,
it primarily involves eroding the foreground of the image and it is suggested to have the foreground as white.
The thickness of erosion depends on the size and shape of the defined kernel.
"""
# define the kernel
kernel = np.ones((5, 5), np.uint8)
# erode the image
erosion = cv2.erode(binr,kernel,iterations=1)
# print the output
imshow("erosion", erosion)

# %%
"""
Dilation involves dilating the outer surface (the foreground) of the image.
As binary images only contain two pixels 0 and 255,
it primarily involves expanding the foreground of the image and it is suggested to have the foreground as white.
The thickness of erosion depends on the size and shape of the defined kernel. 
"""
# define the kernel
kernel = np.ones((5, 5), np.uint8)
# dilate the image
dilation = cv2.dilate(binr, kernel, iterations=1)
# print the output
imshow("dilation", dilation)
# %%
