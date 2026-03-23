#%%
#To determine if a point is inside, outside, or on the edge of a shape you can check if the point is within a contour using cv2.pointPolygonTest().
import cv2

image = cv2.imread('rectangle.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 120, 255, 1)
cv2.imshow('canny', canny)
cv2.waitKey()
cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv.RETR_EXTERNAL: Retrieves only the external contours.
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

point1 = (25, 50)
point2 = (200, 250)
point3 = (200, 350)

# Perform check if point is inside contour/shape
for c in cnts:
    cv2.drawContours(image, [c], -1, (36, 255, 12), 2)
    result1 = cv2.pointPolygonTest(c, point1, False)
    result2 = cv2.pointPolygonTest(c, point2, False)
    result3 = cv2.pointPolygonTest(c, point3, False)
"""
pointPolygonTest the third argument is measureDist:
True, it finds the shortest distance between a point in the image and a contour.
False, it finds whether the point is inside, outside, or on the contour.
"""

# Draw points
cv2.circle(image, point1, 8, (100, 100, 255), -1)
cv2.putText(image, 'point1', (point1[0] -10, point1[1] -20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), lineType=cv2.LINE_AA)
cv2.circle(image, point2, 8, (200, 100, 55), -1)
cv2.putText(image, 'point2', (point2[0] -10, point2[1] -20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), lineType=cv2.LINE_AA)
cv2.circle(image, point3, 8, (150, 50, 155), -1)
cv2.putText(image, 'point3', (point3[0] -10, point3[1] -20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), lineType=cv2.LINE_AA)

print('point1:', result1)
print('point2:', result2)
print('point3:', result3)
#The function returns +1, -1, or 0 to indicate if a point is inside, outside, or on the contour, respectively

cv2.imshow('image', image)
cv2.waitKey()
#%%