#################################################################
# more info:
# https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
#################################################################
# %%
# import libraries
import cv2
import sys
import numpy as np
from numpy.distutils.extension import cxx_ext_re


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
img = cv2.imread("blob.jpg") 
if img is None:
    sys.exit("Could not read the image.")
imshow("Original_img", img)

# %%
# convert image to grayscale image and to binary image
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray_image,127,255,0)

# %%
# calculate moments of binary image (dictionary)
'''complete the code'''
M = cv2.moments(thresh)
print(M)
cX = int(M['m10']/M['m00'])
cY = int(M['m01']/M['m00'])
# %%
# put text and highlight the center
'''complete the code'''
cv2.circle(img,(cX, cY), 5, (0,0,255), -1)
cv2.putText(img, 'centroid', (cX - 25, cY -25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
imshow("centroid", img)