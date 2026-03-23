import cv2 as cv
import numpy as np
import sys



def analyze_tray_and_coins(image_path):
    img_color = cv.imread(image_path)
    if img_color is None:
        sys.exit("Error loading image")

    img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
    img_blur = cv.medianBlur(img_gray, 5)

    _, thresh = cv.threshold(img_blur,90,255,cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    tray_contour = max(contours, key=cv.contourArea)
    tray_area = cv.contourArea(tray_contour)
    print(f"Area of the tray: {tray_area}\n")

    cv.drawContours(img_color, [tray_contour], 0, (0,255,0),3)

    circles = cv.HoughCircles(img_blur, cv.HOUGH_GRADIENT, dp=1, minDist=35, param1=65, param2=40, minRadius=15, maxRadius=50)

    large_on_tray = 0
    large_off_tray = 0
    small_on_tray = 0
    small_off_tray = 0

    total_coins = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        total_coins = len(circles[0])
        RADIUS_THRESHOLD = 33

        for i in circles[0,:]:
            x, y, r = i[0], i[1], i[2]

            is_inside = cv.pointPolygonTest(tray_contour, (int(x), int(y)), False)

            if r >= RADIUS_THRESHOLD:
                if is_inside >= 0:
                    large_on_tray += 1
                else:
                    large_off_tray += 1

                cv.circle(img_color, (x,y), r, (0, 0, 255), 2)
            else:
                if is_inside >= 0:
                    small_on_tray += 1
                else:
                    small_off_tray += 1

                cv.circle(img_color, (x,y), r, (255, 0, 0), 2)

            cv.circle(img_color, (x,y),2, (0, 255, 255), 5)


    print(f"----{image_path}----")
    print(f"Number of coins: {total_coins}")
    print(f"Large on the tray: {large_on_tray}")
    print(f"Large off the tray: {large_off_tray}")
    print(f"Small on the tray: {small_on_tray}")
    print(f"Small off the tray: {small_off_tray}\n")

    cv.imshow("Image", img_color)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    trays_to_test = ['tray8.jpg', 'tray3.jpg', 'tray7.jpg']
    for img_path in trays_to_test:
        analyze_tray_and_coins(img_path)




