import numpy as np
import cv2 as cv

def main():
    img = cv.imread("photo_1.jpg")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    corners = cv.goodFeaturesToTrack(
        gray,
        4,
        0.01,
        10,
        useHarrisDetector=True,
        k=0.04
    )
    corners = np.intp(corners)

    img_harris = img.copy()
    for i in corners:
        x, y = i.ravel()
        cv.circle(img_harris, (int(x), int(y)), 5, 255, -1)

    cv.imshow("Harris", img_harris)
    cv.waitKey(0)
    cv.destroyAllWindows()

    img2 = cv.imread("photo_1.jpg")
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(gray2, None)

    img_sift = cv.drawKeypoints(
        gray2,
        kp,
        img2,
        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    cv.imshow("SIFT keypoints", img_sift)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()