# %%
# import libraries
import cv2
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
# Upload image using cv2.imread
# cv2.imread() - load an image from a file specified by its file path
# by default, it reads images in the BGR (Blue, Green, Red) color space
img = cv2.imread("OpenCV_Logo.png", cv2.IMREAD_COLOR)
if img is None:
    sys.exit("Could not read the image.")
imshow("Original_img", img)

# %%
# BGR colors channels
b , g , r = cv2.split(img) 
bgr_img = cv2.merge(( b , g , r ))
imshow("Original",img)
imshow("bgr",bgr_img)
imshow("B",b) #white areas have max value, black min value, white, #FFFFFF, rgb(255,255,255)
imshow("G",g)
imshow("R",r)

# %%
# Gray
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imshow("img_gray ",img_gray)

# %%
# Display the image using pyplot.
# Convert the image from the BGR format (used by OpenCV)
# to the RGB format (expected by Matplotlib) before displaying it
image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.title('Image using cvtColor and pyplot with matplotlib')
plt.axis('off')  # Turn off axis
plt.show()

# %%
# upload image using mpimg.imread 
image = mpimg.imread("OpenCV_Logo.png")
plt.imshow(image_rgb)
plt.title('Image using mpimg.imread and pyplot with matplotlib')
plt.axis('off')  # Turn off axis
plt.show()

# %%
# HSV
# RGB and HSV Color Model Demo : https://math.hws.edu/graphicsbook/demos/c2/rgb-hsv.html
img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
h,s,v=cv2.split(img_hsv)
hsv_img = cv2.merge(( h , s , v ))
imshow("hsv_img no converts",hsv_img)
# Images are always displayed as they are BGR. 
# If you want to show your HSV image, you need to convert to BGR first
out = cv2.cvtColor(hsv_img , cv2.COLOR_HSV2BGR)
imshow("hsv_image after converts ",out)

imshow("H",h)
imshow("S",s)
imshow("V",v)
# %%
