#%%
import cv2 as cv

def imshow(title, image):
    cv.imshow(title,image)
    k = cv.waitKey(0)#0 infinity waiting time
    if k == ord("s"):
        cv.imwrite(f"00_{title}_saved.jpg", image)
    cv.destroyAllWindows()
    return

#%%
im = cv.imread('rectangle.png')
assert im is not None, "file could not be read, check with os.path.exists()"
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
imshow('imgray',imgray)

#%%
_, thresh = cv.threshold(imgray, 127, 255, cv.THRESH_BINARY)#cv.THRESH_BINARY_INV
# The first argument is the source image, which should be a grayscale image.
# The second argument is the threshold value which is used to classify the pixel values.
# The third argument is the maximum value which is assigned to pixel values exceeding the threshold.
imshow('thresh',thresh)

#%%
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#gadf1ad6a0b82947fa1fe3c3d497f260e0
# finding contours is like finding white object from black background
# first one is source image, second is contour retrieval mode, third is contour approximation method
print(hierarchy)
# hierarchy[0][i]: Next contour at the same hierarchical level.
# hierarchy[1][i]: Previous contour at the same hierarchical level.
# hierarchy[2][i]: Child contour.
# hierarchy[3][i]: Parent contour.

#%%
#To draw an individual contour:
cv.drawContours(im, contours, 0, (0,255,0), 2)
# 3rd parameter - contourIdx - the index of the contour to draw (negative-all contours).
imshow('im',im)

#To draw an individual contour:
cv.drawContours(im, contours, 1, (255,0,0), 2)
imshow('im',im)
#To draw an individual contour:
cv.drawContours(im, contours, 2, (255,255,0), 2)
imshow('im',im)
#To draw all the contours in an image (negative number):
cv.drawContours(im, contours, -1, (0,0,0), 3)
imshow('im',im)