#################################################################
# more info:
# https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html
#################################################################
# %%
# import libraries
import cv2
import sys
import numpy

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
imshow("Original img", img)

# %%
# img.shape the number of rows, columns, and channels (if the image is color)
num_rows , num_cols = img.shape[:2] #img.shape = (768, 1024, 3) tuple

# %%
# 1 Translations - right and bottom
translation_matrix=numpy.float32([[1,0,70],[0,1,110]])
img_translation=cv2.warpAffine(img,translation_matrix,(num_cols+70,num_rows+110))
imshow("Translation_1",img_translation)

# %%
# 2 Translations - left and top
translation_matrix=numpy.float32([[1,0,-30],[0,1,-50]])
img_translation=cv2.warpAffine(img_translation,translation_matrix,(num_cols+70+30,num_rows+110+50))
imshow("Translation_1_2",img_translation)

# example of perspective transformation to study at home
# https://theailearner.com/tag/cv2-getperspectivetransform/