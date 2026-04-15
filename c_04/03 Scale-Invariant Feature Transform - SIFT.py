"""
https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf page 2
https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html

Corner detectors like Harris etc. are rotation-invariant, corners remain corners in rotated image also.
But a corner may not be a corner if the image is scaled. So Harris corner is not scale invariant.
In 2004, D.Lowe, University of British Columbia, came up with a new algorithm, Scale Invariant Feature Transform (SIFT) in his paper.
There are mainly four steps involved in SIFT algorithm.
1. Scale-space Extrema Detection
From the image above, it is obvious that we can't use the same window to detect keypoints with different scale.
It is OK with small corner. But to detect larger corners we need larger windows. For this, scale-space filtering is used.
2. Keypoint Localization
Once potential keypoints locations are found, they have to be refined to get more accurate results.
They used Taylor series expansion of scale space to get more accurate location of extrema.
3. Orientation Assignment
Now an orientation is assigned to each keypoint to achieve invariance to image rotation.
A neighbourhood is taken around the keypoint location depending on the scale, and the gradient magnitude and direction is calculated in that region.
4. Keypoint Descriptor
Now keypoint descriptor is created. A 16x16 neighbourhood around the keypoint is taken.
"""
import numpy as np
import cv2 as cv
img = cv.imread('chessboard.jpg')
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()

#sift.detect() function finds the keypoint in the images.
kp = sift.detect(gray,None)

kp, des = sift.detectAndCompute(gray,None)

"""
A flag, cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, will draw a circle with size of keypoint and it will even show its orientation
"""
img=cv.drawKeypoints(gray,kp,img)
#img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

img_res = cv.resize(img, (0, 0), fx = 0.7, fy = 0.7)
cv.imshow('sift_keypoints.jpg',img_res)
cv.waitKey()
cv.destroyAllWindows()
cv.imwrite("03_sift_keypoints_img.jpg", img)
"""
TASKS:

- import chessboard.jpg, saw1.jpg, and dino.jpg and interpret the results.
*for saw1.jpg resize img: img_res = cv2.resize(img, (0, 0), fx = 0.1, fy = 0.1)

-find keypoints 'kp' and descriptors 'des' in a single step with the function, sift.detectAndCompute().
kp, des = sift.detectAndCompute(gray,None)
A computer also should describe the region around the feature so that it can find it in other images.
So called description is called Feature Description .

- what is the shape of the 'des'?
print(des.shape)
'des' is a numpy array of shape (Number of Keypoints) * 128.
"""


# lista obrazów
images = ['chessboard.jpg', 'saw1.jpg', 'messi.jpg']

for name in images:
    img = cv.imread(name)
    
    # resize tylko dla saw1
    if name == 'saw1.jpg':
        img = cv.resize(img, (0, 0), fx=0.1, fy=0.1)
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    sift = cv.SIFT_create()
    
    # wykrycie punktów i deskryptorów w jednym kroku
    kp, des = sift.detectAndCompute(gray, None)
    
    print(f"{name}:")
    print("Liczba keypointów:", len(kp))
    
    if des is not None:
        print("Shape des:", des.shape)
    else:
        print("Brak deskryptorów")
    
    print("----------------------")
    
    # rysowanie punktów
    img_kp = cv.drawKeypoints(gray, kp, img, 
        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    cv.imshow(name, img_kp)
    cv.waitKey(0)

cv.destroyAllWindows()