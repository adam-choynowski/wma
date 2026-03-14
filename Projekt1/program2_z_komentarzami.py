
import cv2
import numpy as np
import sys


cap = cv2.VideoCapture('rgb_ball_720.mp4')

ret, frame = cap.read()
if not ret:
    sys.exit("Nie mozna otworzyc pliku wideo.")

while True:

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    """
    tutaj widełki są bardziej rygorystyczne bo na filmie występuje więcej szumów
    """

    lower_red1 = np.array([0, 220, 80])
    upper_red1 = np.array([8, 255, 255])
    lower_red2 = np.array([172, 220, 80])
    upper_red2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    """
    cv2.findContours spaceruje po czarno-białej masce i szuka "wysp" – ciągłych, zamkniętych kształtów białych pikseli.
    cv2.RETR_EXTERNAL oznacza, że interesują Cię tylko zewnętrzne granice tych wysp (bez ewentualnych dziur w środku), a cv2.CHAIN_APPROX_SIMPLE kompresuje te kontury, żeby zużywały mniej pamięci operacyjnej.
    
    hierarchy - to tablica realacji krawedzi zewnętrznych do wewnętrznych
    cv2.RETR_EXTERNAL mówi ze interesuje mnie i tak tylko zewnętrzne wiec nie uzywamy hierarchy
    
    """

    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # bierzemy największy kontur zebyc nie przeskakiwalo w razie szumów
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)

        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])

            cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
            cv2.putText(frame, "czerwona pilka", (cX - 50, cY - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('red ball (q) - exit', frame)
    if cv2.waitKey(1) == ord('q'):
        break

    ret, frame = cap.read()
    if not ret:
        print("Nie mozna otworzyc pliku wideo.")
        break

cap.release()
cv2.destroyAllWindows()