#################################################################
# more info:
# https://www.geeksforgeeks.org/python-opencv-morphological-operations/
#################################################################
# %%
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
img = cv2.imread("dots.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    sys.exit("Could not read the image.")
imshow("Original_img", img)

# %%
# binarize the image
binr = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1]
imshow("binr", binr)

# %%
# Opening
"""
Opening involves erosion followed by dilation in the outer surface (the foreground) of the image.
All the above-said constraints for erosion and dilation applies here.
It is a blend of the two prime methods.
It is generally used to remove the noise in the image.
"""
# define the kernel
kernel = np.ones((3, 3), np.uint8)
# opening the image
opening = cv2.morphologyEx(binr, cv2.MORPH_OPEN,kernel, iterations=1)
imshow("opening", opening)

# %%
# Closing
"""
Closing involves dilation followed by erosion in the outer surface (the foreground) of the image.
All the above-said constraints for erosion and dilation applies here.
It is a blend of the two prime methods.
It is generally used to remove the noise in the image.
"""
# define the kernel
kernel = np.ones((3, 3), np.uint8)
# closing the image
closing = cv2.morphologyEx(binr, cv2.MORPH_CLOSE, kernel, iterations=1)
imshow("closing", closing)
# %%
