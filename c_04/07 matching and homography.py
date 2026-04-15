#%%
# https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html
# https://www.geeksforgeeks.org/python-opencv-object-tracking-using-homography/
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

#%%
# find SIFT features in images
# apply the ratio test to find the best matches.
MIN_MATCH_COUNT = 10
# a condition that at least 10 matches are to be there to find the object,
# otherwise simply show a message saying not enough matches are present.
img1 = cv.imread("man_rotated.jpg", cv.IMREAD_GRAYSCALE) # queryImage
img2 = cv.imread("people.jpg", cv.IMREAD_GRAYSCALE) # trainImage
# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)
# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)


#%%
# If enough matches are found,
# we extract the locations of matched keypoints in both the images.
# They are passed to find the perspective transformation by findHomography.
# Once we get this 3x3 transformation matrix,
# we use it to transform the corners of queryImage to corresponding points in trainImage.

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

#%%
# draw our inliers (if successfully found the object) or matching keypoints (if failed)
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
plt.imshow(img3, 'gray'),plt.show()
cv.imwrite("07_findHomography_img3.jpg", img3)
"""
TASKS:
- import saw1.jpg and sawtraining1.png and run the code
- write a code for feature matching for dino.jpg and dinovideo.mp4
- read about the object tracker:
https://learnopencv.com/the-complete-guide-to-object-tracking-in-computer-vision/
https://broutonlab.com/blog/opencv-object-tracking
https://docs.opencv.org/4.x/d2/d0a/tutorial_introduction_to_tracker.html
"""

MIN_MATCH_COUNT = 10

# Load reference image (dino)
img1 = cv.imread("saw4.jpg", cv.IMREAD_GRAYSCALE)

sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)

# FLANN matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)

# Open video
cap = cv.VideoCapture("sawmovie.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    kp2, des2 = sift.detectAndCompute(gray, None)

    if des2 is None:
        continue

    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

        if M is not None:
            h, w = img1.shape
            pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
            dst = cv.perspectiveTransform(pts, M)

            frame = cv.polylines(frame, [np.int32(dst)], True, (0,255,0), 3)

    cv.imshow("Saw Tracking", frame)

    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()