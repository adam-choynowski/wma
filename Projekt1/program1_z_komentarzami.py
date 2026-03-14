
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
    sys.exit("Nie mozna załadować obrazku.")
imshow("Original_img", img)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

"""
h - w normalnie kolory na hsv sa w 360° ale openCv,
używa kanałów 8 bitowych 0-255 więc dziel kąt na 2 i
powstaje zakres h (0-179) czerwony na diagramie
jest na poczatku i konca okregu wiec potrzebujemy
dwoch mask ktore potem łączymy

s - saturacja nasycenie koloru od 0-255 ja daje od 100 zeby uciac malo nasycone kolory ktore nie bede czerownym 

v - jasnosc analogicznie ucinam zeby byly czerowne tylko
"""


lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 100, 100])
upper_red2 = np.array([179, 255, 255])

""" 
analizuje czy kazdy piskel jest w zakresie jesli tak to zmienia
jego wartosc na 255 idalnie bialy jesli nie to zmienia wartosc 
na 0 idealnie czarny 
"""

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

#łączenie białych i czarnych piskeli z obu obrazków

mask = cv2.bitwise_or(mask1, mask2)
imshow("red_mask_before_morphology", mask)

kernel = np.ones((7, 7), np.uint8)
"""
cv2.MORPH_CLOSE (Zamknięcie). Skupia się na naprawie obiektu głównego.
Najpierw rozszerza białe piksele, żeby "zalać" i 
zaszpachlować czarne dziury wewnątrz piłki, a potem lekko ją kurczy do oryginalnego rozmiaru. Efekt:
Masz już pełną, solidną piłkę bez dziur, ale śmieci w tle wciąż tam są
"""

mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

"""
cv2.MORPH_OPEN (Otwarcie). Skupia się na posprzątaniu tła. Najpierw erozja bezlitośnie
ściera białe obszary. Ponieważ Twoja piłka jest teraz wielką i litą 
bryłą (dzięki poprzedniemu krokowi), przetrwa to ścieranie. Natomiast małe kropki szumu w tle 
znikną bezpowrotnie. Potem dylatacja przywraca piłce jej pierwotny rozmiar.
"""

mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

imshow("red_mask_after_morphology", mask)


M = cv2.moments(mask)

#czy znaleziono biale piksele

"""
m10: moment pierwszego rzędu dla osi X 
(można to rozumieć jako sumę współrzędnych X wszystkich punktów należących do piłki).

m01: moment pierwszego rzędu dla osi Y 
(suma współrzędnych Y wszystkich punktów należących do piłki).

Aby znaleźć punkt, który leży dokładnie na samym środku tego obiektu,
musisz podzielić te sumy współrzędnych przez masę obiektu (czyli jego pole powierzchni m00)

"""
if M['m00'] != 0:
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])
else:
    sys.exit("Nie znaleziono czerwonej pilki na obrazie.")


cv2.circle(img, (cX, cY), 5, (0, 0, 255), -1)
cv2.putText(img, "czerwona pilka", (cX - 50, cY - 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

imshow("red_ball_detected", img)