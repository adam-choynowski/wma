#https://docs.opencv.org/4.x/d4/d8c/tutorial_py_shi_tomasi.html
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('chessboard.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
"""
cv.goodFeaturesToTrack()
finds e.g. N=25 strongest corners in the image by Shi-Tomasi method (default) or Harris Corner Detection.
As usual, image should be a grayscale image.
maxCorners - specify number of corners you want to find.
qualityLevel - specify the quality level, which is a value between 0-1,
which denotes the minimum quality of corner below which everyone is rejected.
minDistance - the minimum euclidean distance between corners detected.
useHarrisDetector=True,k=0.04 - using Harris Corner Detection method with specified, e.g. 0.04, free parameter 'k' of Harris Corner Detection.
"""
corners = cv.goodFeaturesToTrack(gray,maxCorners = 4, qualityLevel=0.01,minDistance=10,useHarrisDetector=True,k=0.04)
#cv.goodFeaturesToTrack(gray,maxCorners = 5, qualityLevel=0.01,minDistance=100,useHarrisDetector=True,k=0.04)
corners = np.int8(corners)
print(corners[0].ravel())
for i in corners:
    x,y = i.ravel()
    #ravel return a contiguous flattened array.
    cv.circle(img,(x,y),10,255,-1)
plt.imshow(img),plt.show()
cv.imwrite("02_goodFeaturesToTrack_img.jpg", img)
"""
TASKS
-play with the goodFeaturesToTrack parameters
-import chessboard.jpg, saw1.jpg, and dino.jpg and check five of the strongest corners, how changing parameters influences on the results?
*for saw1.jpg resize img: img_res = cv2.resize(img, (0, 0), fx = 0.1, fy = 0.1)
-compare two methods : Shi-Tomasi method (default) and Harris Corner Detection """


images = ['chessboard.jpg', 'saw1.jpg', 'dino.jpg']

for name in images:
    img = cv.imread(name)
    
    # resize tylko dla saw1
    if name == 'saw1.jpg':
        img = cv.resize(img, (0, 0), fx=0.1, fy=0.1)
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Shi-Tomasi (domyślnie)
    corners = cv.goodFeaturesToTrack(
        gray,
        maxCorners=20,
        qualityLevel=0.01,
        minDistance=20
    )




    corners = np.int16(corners)

    img_copy = img.copy()
    for i in corners:
        x, y = i.ravel()
        cv.circle(img_copy, (x, y), 8, (0, 255, 0), -1)

    plt.imshow(cv.cvtColor(img_copy, cv.COLOR_BGR2RGB))
    plt.title(f"{name} - Shi-Tomasi")
    plt.show()

    corners = np.float32(corners)
    corners = cv.goodFeaturesToTrack(
    gray,
    maxCorners=5,
    qualityLevel=0.01,
    minDistance=20,
    useHarrisDetector=True,
    k=0.04
)

