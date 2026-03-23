import numpy as np
import cv2 as cv
img = cv.imread('tray1.jpg', 0)
img = cv.medianBlur(img,3)
cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
#%%
#detect 5 zl coins (the bigger one)
circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,100, param1=20,param2=60,minRadius=35,maxRadius=100)
circles = np.uint16(np.around(circles))
#print(circles)
#print(circles[0,:])
for i in circles[0,:]:
    # draw the outer circle
    cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
print(len(circles[0,:]))
cv.imshow('detected 5 zl coins = ',cimg)
cv.waitKey(0)
cv.destroyAllWindows()
"""
TASK:
-count 5 zł and 5 gr separately
-validate the code with diffrent sources: tray2.jpg, ..., tray8.jpg
"""