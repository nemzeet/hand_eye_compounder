import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

plot_grid_size = (4, 3) # color 이미지에 대한 map추가

def plot_img_histcdf(index1, index2, index3, img, img_rgb, title):
    plt.subplot(plot_grid_size[0], plot_grid_size[1], index1)
    plt.imshow(img, cmap='gray')
    plt.axis('off'), plt.title(title) 
    
    plt.subplot(plot_grid_size[0], plot_grid_size[1], index3)
    plt.imshow(img_rgb)
    plt.axis('off'), plt.title(title + ' rgb') 

    plt.subplot(plot_grid_size[0], plot_grid_size[1], index2)
    hist,bins = np.histogram(img.flatten(), 256, [0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    plt.plot(cdf_normalized, color = 'b')
    plt.hist(img.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256]), plt.title(title) 
    plt.legend(('cdf','histogram'), loc = 'upper left')


img_rgb = cv.imread('house.jpg')
img_rgb = cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB)
img_yuv = cv.cvtColor(img_rgb, cv.COLOR_RGB2YUV)

img = cv.imread("house.jpg", cv.IMREAD_GRAYSCALE)
plot_img_histcdf(1, 2, 3, img, img_rgb,"original")

img_he = cv.equalizeHist(img)
img_he2 = img_yuv.copy()
img_he2[:, :, 0] = cv.equalizeHist(img_he2[:,:,0])
img_rgb_he = cv.cvtColor(img_he2, cv.COLOR_YUV2RGB)
plot_img_histcdf(4, 5, 6, img_he, img_rgb_he, "HE")

# herein, 255(무한대) implies no limit contrast!, which uses AHE
ahe = cv.createCLAHE(clipLimit=255, tileGridSize=(8,8))
img_ahe = ahe.apply(img)
img_ahe2 = img_yuv.copy()
img_ahe2[:, :, 0] = ahe.apply(img_ahe2[:,:,0])
img_rgb_ahe = cv.cvtColor(img_ahe2, cv.COLOR_YUV2RGB)
plot_img_histcdf(7, 8, 9, img_ahe, img_rgb_ahe, "AHE")

clahe = cv.createCLAHE(clipLimit = 2.0, tileGridSize=(8,8))
img_clahe = clahe.apply(img)
img_clahe2 = img_yuv.copy()
img_clahe2[:, :, 0] = clahe.apply(img_clahe2[:,:,0])
img_rgb_clahe = cv.cvtColor(img_clahe2, cv.COLOR_YUV2RGB)

plot_img_histcdf(10, 11, 12, img_clahe, img_rgb_clahe, "CLAHE")


plt.show()