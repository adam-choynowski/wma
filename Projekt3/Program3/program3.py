import numpy as np
import cv2 as cv

def main():
    MIN_MATCH_COUNT = 8

    img1 = cv.imread("photo_3_train.jpg", cv.IMREAD_GRAYSCALE)

    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    cap = cv.VideoCapture("video_3_query.mp4")

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
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

            if M is not None:
                h, w = img1.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv.perspectiveTransform(pts, M)
                frame = cv.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3)

        cv.imshow("Dino Tracking", frame)
        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()
if __name__ == "__main__":
    main()