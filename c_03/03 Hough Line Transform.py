import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('sudoku.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

edges = cv.Canny(gray,50,150,apertureSize = 3)

lines = cv.HoughLines(edges,1.,np.pi/180*1/2,150)


"""The Hough transform is a technique which can be used to isolate features of a particular shape within an image.
Because it requires that the desired features be specified in some parametric form,
the classical Hough transform is most commonly used for the detection of regular curves such as lines, circles, ellipses, etc.
https://homepages.inf.ed.ac.uk/rbf/HIPR2/hough.htm
"""

for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)

plt.subplot(1,3,1)
plt.imshow(gray,cmap = 'gray'),plt.title('gray'),plt.xticks([]),plt.yticks([])

plt.subplot(1,3,2)
plt.imshow(edges,cmap = 'gray'),plt.title('edges'),plt.xticks([]),plt.yticks([])

plt.subplot(1,3,3)
plt.imshow(img,cmap = 'gray'),plt.title('img'),plt.xticks([]),plt.yticks([])

plt.show() #Display all open figures.

"""
TASK:
- play with 'HoughLines' parameters. What value gives diagonals or doesn't give all horizontal and vertical lines?
2nd - rho - Distance resolution of the accumulator in pixels - from 0.1 to 2.0 is enough.
3rd - theta - Angle resolution of the accumulator in radians - from pi/180/2 to pi/180*10 is enough.
4th - threshold - Accumulator threshold parameter. Only those lines are returned that get enough votes.
"""

results = [
    ("default", 1, np.pi/180, 150),
    ("more lines", 1, np.pi/180, 100),
    ("less lines", 1, np.pi/180, 300),
    ("better diagonals", 1, np.pi/180*2, 120),
    ("coarse angle", 1, np.pi/90, 150),
    ("rho 0.5", 0.5, np.pi/180, 150),
    ("rho 1.5", 1.5, np.pi/180, 150),
    ("rho 2.0", 2.0, np.pi/180, 150),
]

plt.figure(figsize=(12,8))

for i, (title, rho, theta, thresh) in enumerate(results):
    img_copy = cv.imread('sudoku.jpg')
    
    lines = cv.HoughLines(edges, rho, theta, thresh)
    
    if lines is not None:
        for line in lines:
            rho_l, theta_l = line[0]
            a = np.cos(theta_l)
            b = np.sin(theta_l)
            x0 = a * rho_l
            y0 = b * rho_l
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv.line(img_copy,(x1,y1),(x2,y2),(255,0,0),2)

    plt.subplot(3,3,i+1)
    plt.imshow(cv.cvtColor(img_copy, cv.COLOR_BGR2RGB))
    plt.title(title)
    plt.xticks([]); plt.yticks([])

plt.tight_layout()
plt.show()