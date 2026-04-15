# https://docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html
import cv2
import numpy as np 
import matplotlib.pyplot as plt

def plot_image(ax, image, title):
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.axis('off')

# Read the image and convert to grayscale
ori = cv2.imread('04_simple.jpg')#footballers.jpg
gray = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)
fig, axes = plt.subplots(2, 3, figsize=(12, 10))
plot_image(axes[0,0], ori, 'Original')

# Apply Harris corner detection
# https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html
image = ori.copy()
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
dst = cv2.dilate(dst, None)
image[dst > 0.1 * dst.max()] = [0, 0, 255]
plot_image(axes[0,1], image, 'Harris')

# Apply Shi-Tomasi corner detection
# https://docs.opencv.org/4.x/d4/d8c/tutorial_py_shi_tomasi.html
image = ori.copy()
corners = cv2.goodFeaturesToTrack(gray,20,0.01,10)
corners = np.intp(corners)
for i in corners:
    x,y = i.ravel()
    cv2.circle(image,(x,y),3,255,-1)
plot_image(axes[0,2], image, 'Shi-Tomasi')


'''
detector classes: cv2.SIFT_create(), cv2.ORB_create(), ...
more: https://docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html
detector.detect(image, None) You can pass a mask if you want to search only a part of image.
detector.compute(image, kp) computes the descriptors from the keypoints kp we have found.
detector.detectAndCompute(image, None) directly find keypoints and descriptors in a single step

cv2.drawKeypoints(
    The input image,
    an object of the cv2.KeyPoint class,
    Optional output image,
    color a single scalar value or a tuple of three values representing the color in BGR format,
    Flags setting drawing behavior e.g. cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS: For each keypoint, the circle around the keypoint with keypoint size and orientation will be drawn)
'''

# Apply Scale-Invariant Feature Transform
# https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html
image = ori.copy()
sift = cv2.SIFT_create()
kp = sift.detect(gray,None)
image=cv2.drawKeypoints(image,kp,image)#flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
plot_image(axes[1,0], image, 'SIFT')

# Apply ORB (Oriented FAST and Rotated BRIEF)
# https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html
image = ori.copy()
orb = cv2.ORB_create()
kp = orb.detect(image,None)
image=cv2.drawKeypoints(image,kp,image)
plot_image(axes[1,1], image, 'ORB')

# FAST Algorithm for Corner Detection
# https://docs.opencv.org/4.x/df/d0c/tutorial_py_fast.html
image = ori.copy()
fast = cv2.FastFeatureDetector_create()
kp = fast.detect(image,None)
image=cv2.drawKeypoints(image,kp,image)
plot_image(axes[1,2], image, 'FAST')

plt.show()