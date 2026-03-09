#################################################################
# more info:
# https://techvidvan.com/tutorials/detect-objects-of-similar-color-using-opencv-in-python/?fbclid=IwAR3X4TceQjEglgg9VPWyAn5DW34OExBbjrhlXMZ_wOcC6ZFNWo_HPGOFPZQ
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
# Upload image using cv2.imread in color
img = cv2.imread("hsv_circle.jpg") 
if img is None:
    sys.exit("Could not read the image.")
imshow("Original_img", img)

# %%
# convert to hsv colorspace
'''complete the code'''
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


# %%
# bounds
# https://math.hws.edu/graphicsbook/demos/c2/rgb-hsv.html
# lower bound and upper bound for Green color
# green HSV (120, somewhere 1-254,somewhere 1-254)
'''complete the code'''
lower_bound = np.array([60-15, 20 , 20])
upper_bound = np.array([60+15, 255 , 255])
# %%
# find the colors within the boundaries
'''complete the code'''
mask = cv2.inRange(hsv, lower_bound, upper_bound)
imshow('green mask', mask)

# in the frame wherever the green color is detected
# the mask shows that as white and the rest of the region as black.

# %%
# Remove unnecessary noise from mask
# define kernel size  
kernel = np.ones((7,7),np.uint8)
# np.ones((5,5),np.uint8) create a 5×5 8 bit integer matrix.
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
# cv2.MORPH_CLOSE removes unnecessary black noises from the white region.
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
# cv2.MORPH_OPEN removes white noise from the black region of the mask.
imshow("mask after removing noises", mask)

# %%
# Segment only the detected region
segmented_img = cv2.bitwise_and(img, img, mask=mask)
imshow("segmented_img", segmented_img)
# cv2.bitwise_and() applies mask on frame in only that region
# where the mask is true means white.

# %%
# Find contours from the mask
contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
temp_output = cv2.drawContours(segmented_img, contours, -1, 255, cv2.LINE_AA)
imshow("temp_output", temp_output)

# %%
# Draw contour on original image
output = cv2.drawContours(img, contours, -1, 255, cv2.LINE_AA)
imshow("output", output)