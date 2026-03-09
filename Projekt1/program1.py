
import cv2
import numpy as np
import sys

def imshow(title, image):
    cv2.imshow(title, image)
    k = cv2.waitKey(0)
    if k == ord("s"):
        cv2.imwrite(f"00_{title}_saved.jpg", image)
    cv2.destroyAllWindows()
    return

img = cv2.imread("red_ball.jpg")
if img is None:
    sys.exit("Could not read the image.")
imshow("Original_img", img)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 100, 100])
upper_red2 = np.array([179, 255, 255])

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

mask = cv2.bitwise_or(mask1, mask2)
imshow("red_mask_before_morphology", mask)

kernel = np.ones((7, 7), np.uint8)

mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

imshow("red_mask_after_morphology", mask)


M = cv2.moments(mask)


if M['m00'] != 0:
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])
else:
    sys.exit("Nie znaleziono czerwonej pilki na obrazie.")


cv2.circle(img, (cX, cY), 5, (0, 0, 255), -1)
cv2.putText(img, "czerwona pilka", (cX - 50, cY - 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

imshow("red_ball_detected", img)