#%% jupyter
# See 04 WMA lecture
# Find Image gradients, edges etc
import sys
import cv2 as cv
import matplotlib.pyplot as plt
#%%
#import image
img = cv.imread('sudoku.jpg', cv.IMREAD_GRAYSCALE)
if img is None:
    sys.exit("Could not read the image.")
plt.imshow(img,cmap = 'gray')

#%%
"""
Sobel derivatives :
a joint Gaussian smoothing plus differentiation operation, it is more resistant to noise.
You can specify the direction of derivatives to be taken, vertical or horizontal (by yorder and xorder respectively).
You can also specify the size of kernel by the argument ksize.
more : https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html
"""
sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
plt.imshow(sobelx,cmap = 'gray')
"""
TASK:
- change ksize 2*n+1, n from 1 to e.g. 10, and check the result
- ddepth: Depth of the destination image. Change CV_64F to: CV_8U or CV_16U or CV_16S or CV_32F 

note:
CV_8U - 8-bit unsigned integers ( 0..255 ) ie a pixel can have values 0-255 like in grayscale
CV_16U - 16-bit unsigned integers ( 0..65535 )
CV_16S - 16-bit signed integers ( -32768..32767 )
CV_32F is float - the pixel can have any value between 0-1.0. To display by multiplying each pixel by 255.
CV_64F - 64-bit floating-point
"""

#k-sizes
plt.figure(figsize=(10,8))

for i, k in enumerate([1,3,5,7,9]):
    sob = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=k)
    
    plt.subplot(2,3,i+1)
    plt.imshow(sob, cmap='gray')
    plt.title(f"Sobel ksize={k}")
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.show()

#ddepths
depths = [
    (cv.CV_8U, "CV_8U"),
    (cv.CV_16U, "CV_16U"),
    (cv.CV_16S, "CV_16S"),
    (cv.CV_32F, "CV_32F"),
    (cv.CV_64F, "CV_64F")
]

plt.figure(figsize=(10,8))

for i, (d, name) in enumerate(depths):
    sob = cv.Sobel(img, d, 1, 0, ksize=5)

    # float scaling
    if d in [cv.CV_32F, cv.CV_64F]:
        sob = cv.normalize(sob, None, 0, 255, cv.NORM_MINMAX)

    plt.subplot(2,3,i+1)
    plt.imshow(sob, cmap='gray')
    plt.title(name)
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.show()

#%%
#gradient (high-pass) filters
"""Laplacian derivatives : 
the Laplacian of the image given by the relation
Δsrc=∂2src∂x2+∂2src∂y2 where each derivative is found using Sobel derivatives.
If ksize = 1, then following kernel [[0,1,0],[1,-4,1],[0,1,0]]
more : https://docs.opencv.org/3.4/d5/db5/tutorial_laplace_operator.html
"""
laplacian =  cv.Laplacian(img,cv.CV_64F,ksize=3)
plt.imshow(laplacian,cmap = 'gray')
"""
TASK:
- Remove noise by blurring with a Gaussian filter
src = cv.GaussianBlur(img, (3, 3), 0)
laplacian_Gauss =  cv.Laplacian(src,dst,cv.CV_64F,ksize=3)
"""
# blur
blur = cv.GaussianBlur(img, (5,5), 0)

# laplacian no blur
laplacian = cv.Laplacian(img, cv.CV_64F, ksize=3)

# laplacian with blur
laplacian_blur = cv.Laplacian(blur, cv.CV_64F, ksize=3)

plt.figure(figsize=(10,6))

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.xticks([]); plt.yticks([])

plt.subplot(1,3,2)
plt.imshow(laplacian, cmap='gray')
plt.title("Laplacian (no blur)")
plt.xticks([]); plt.yticks([])

plt.subplot(1,3,3)
plt.imshow(laplacian_blur, cmap='gray')
plt.title("Laplacian + Gaussian")
plt.xticks([]); plt.yticks([])

plt.show()


#%%
"""
module matplotlib.pyplot as plt is a state-based interface to matplotlib.
It provides an implicit, MATLAB-like, way of plotting.
It also opens figures on your screen, and acts as the figure GUI manager.

subplot(m,n,k)
m number of rows
n number of columns
k order number (from left to right, form up to down like reading)
"""
plt.subplot(2,2,1)
plt.imshow(img,cmap = 'gray')
plt.title('Original')
plt.xticks([])#Get or set the current tick locations and labels of the x-axis. The list is empty to get rid of ticks.
plt.yticks([])

plt.subplot(2,2,2)
plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian')
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,3)
plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X')
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,4)
plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y')
plt.xticks([])
plt.yticks([])

plt.show() #Display all open figures.

""" TASKS: - change cmap colormaps, e.g. https://matplotlib.org/stable/tutorials/colors/colormaps.html, """

cmaps = ['gray', 'hot', 'jet', 'viridis']

plt.figure(figsize=(10,8))

for i, cmap in enumerate(cmaps):
    plt.subplot(2,2,i+1)
    plt.imshow(sobelx, cmap=cmap)
    plt.title(f"cmap = {cmap}")
    plt.xticks([])
    plt.yticks([])

plt.show()