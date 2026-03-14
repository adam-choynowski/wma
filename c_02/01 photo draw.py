# %%
# import libraries
import cv2
import sys

# %%
# imshow function
def imshow(title, image):
    cv2.imshow(title,image)
    k = cv2.waitKey(0)#0 infinity waiting time
    if k == ord("s"):
        cv2.imwrite(f"00_{title}_saved.jpg", image)
    cv2.destroyAllWindows()
    return

# %%
# upload image
img = cv2.imread("robo.jpg")
if img is None:
    sys.exit("Could not read the image.")

# %%
# img.shape the number of rows, columns, and channels (if the image is color)
num_rows , num_cols = img.shape[:2] #img.shape = (768, 1024, 3) tuple

# %%
# draw line and cicle
cv2.line(img,(0,0),(num_cols,num_rows),(255,0,0),5)
cv2.circle(img,(num_cols//2,num_rows//3*2), 63, (0,0,255), -1)#filled with color instead of having only countour
imshow('Line_and_circle',img)