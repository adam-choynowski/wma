import numpy as np
import cv2 as cv

# Capture video
cap = cv.VideoCapture('rgb_ball_720.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    # Check if frame is read correctly; ret is True if successful
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Convert frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display grayscale frame; press 'q' to quit
    cv.imshow('frame, click q to quit', gray)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()