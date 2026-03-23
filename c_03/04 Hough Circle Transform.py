import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('circles.jpg', cv.IMREAD_GRAYSCALE)

img = cv.medianBlur(img,5)
cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)

#The function finds circles in a grayscale image using a modification of the Hough transform.
circles = cv.HoughCircles(img,method = cv.HOUGH_GRADIENT,dp = 1,minDist = 20,param1=50,param2=30,minRadius=0,maxRadius=100)

circles = np.uint16(np.around(circles))

for i in circles[0,:]:
    # draw the outer circle
    cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
cv.imshow('detected circles',cimg)
cv.waitKey(0)
cv.destroyAllWindows()

"""
TASKS:
- play with "cv.HoughCircles" parameters:

method -the available detection methods are #HOUGH_GRADIENT and #HOUGH_GRADIENT_ALT.

dp Inverse ratio of the accumulator resolution to the image resolution,
dp=1 , the accumulator has the same resolution as the input image,
dp=2 , the accumulator has half as big width and height,
for #HOUGH_GRADIENT_ALT the recommended value is dp=1.5, unless some small very circles need to be detected.

minDist Minimum distance between the centers of the detected circles.
param1 the higher threshold of the two passed to the Canny edge detector.
param2 Second method-specific parameter.
In case of #HOUGH_GRADIENT, it is the accumulator threshold for the circle centers at the detection stage.
The smaller it is, the more false circles may be detected.
Circles, corresponding to the larger accumulator values, will be returned first.
In the case of #HOUGH_GRADIENT_ALT algorithm, this is the circle "perfectness" measure.
The closer it to 1, the better shaped circles algorithm selects.
In most cases 0.9 should be fine.
If you want get better detection of small circles, you may decrease it to 0.85, 0.8 or even less.
But then also try to limit the search range [minRadius, maxRadius] to avoid many false circles.

minRadius Minimum circle radius.

maxRadius Maximum circle radius. If <= 0, uses the maximum image dimension.
If < 0, #HOUGH_GRADIENT returns centers without finding the radius. #HOUGH_GRADIENT_ALT always computes circle radiuses.

- play with "cv.circle" parameters

- import 'eyes.jpg' and detect eyes
img = cv.imread('eyes.jpg', cv.IMREAD_GRAYSCALE)
cimg = cv.medianBlur(img,13)
circles = cv.HoughCircles(cimg,method = cv.HOUGH_GRADIENT,dp = 1,minDist = 200,param1=60,param2=30,minRadius=40,maxRadius=100)
"""

#houghcircles params

tests = [
    ("default", 1, 20, 50, 30, 0, 100),
    ("more circles", 1, 20, 50, 20, 0, 100),
    ("less circles", 1, 20, 50, 50, 0, 100),
    ("bigger dp", 2, 20, 50, 30, 0, 100),
    ("far centers", 1, 50, 50, 30, 0, 100),
]

plt.figure(figsize=(12,8))

for i, (title, dp, minDist, p1, p2, rmin, rmax) in enumerate(tests):
    img = cv.imread('circles.jpg', cv.IMREAD_GRAYSCALE)
    img_blur = cv.medianBlur(img,5)
    cimg = cv.cvtColor(img_blur, cv.COLOR_GRAY2BGR)

    circles = cv.HoughCircles(img_blur, cv.HOUGH_GRADIENT,
                              dp=dp, minDist=minDist,
                              param1=p1, param2=p2,
                              minRadius=rmin, maxRadius=rmax)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0,:]:
            cv.circle(cimg,(c[0],c[1]),c[2],(0,255,0),2)
            cv.circle(cimg,(c[0],c[1]),2,(0,0,255),3)

    plt.subplot(2,3,i+1)
    plt.imshow(cv.cvtColor(cimg, cv.COLOR_BGR2RGB))
    plt.title(title)
    plt.xticks([]); plt.yticks([])

plt.tight_layout()
plt.show()

#cv circle params

#cv.circle(cimg,(c[0],c[1]),c[2],(255,0,0),4)  # thicker

cv.circle(cimg,(c[0],c[1]),5,(0,255,255),-1)  # filled
plt.imshow(cv.cvtColor(cimg, cv.COLOR_BGR2RGB))
plt.title("Detected circles (blue edge + yellow center)")
plt.xticks([]); plt.yticks([])
plt.show()

# eye detection

img = cv.imread('eyes.jpg', cv.IMREAD_GRAYSCALE)
blur = cv.medianBlur(img,13)
cimg = cv.cvtColor(blur, cv.COLOR_GRAY2BGR)

circles = cv.HoughCircles(blur, cv.HOUGH_GRADIENT,
                          dp=1,
                          minDist=200,
                          param1=60,
                          param2=30,
                          minRadius=40,
                          maxRadius=100)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for c in circles[0,:]:
        cv.circle(cimg,(c[0],c[1]),c[2],(0,255,0),2)
        cv.circle(cimg,(c[0],c[1]),2,(0,0,255),3)

plt.imshow(cv.cvtColor(cimg, cv.COLOR_BGR2RGB))
plt.title("Detected Eyes")
plt.xticks([]); plt.yticks([])
plt.show()