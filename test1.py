import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# return gaussian pyramid    
def generate_gaussian_pyramid(img, levels):
    GP = [img]
    for i in range(1, levels): # 1 ~ level-1
        img = cv.pyrDown(img)
        GP.append(img)
    return GP

# return laplacian pyramid
def generate_laplacian_pyramid(GP):
    levels = len(GP)
    LP = []
    for i in range(levels - 1, 0, -1): # level 4~0
        upsample_img = cv.pyrUp(GP[i], None, GP[i-1].shape[:2]) # 3번째 position -> 이미지의 사이즈를 명시적으로 적어줌
        laplacian_img = cv.subtract(GP[i-1], upsample_img) 
        LP.append(laplacian_img)
    LP.reverse() # 순서 0~4로 바꿔주기
    return LP

# GP, LP Filtering
def stitchFliter(P_center, P_side, filter1, filter2):
    
    P_stitch = []
    
    for lc, ls, lf, lf2 in zip(P_center, P_side, filter1, filter2):
        
        lf = (lf/float(lf.max()))
        lf2 = (lf2/float(lf2.max()))
        
        lc[:,:,2] = lc[:,:,2]*lf
        lc[:,:,1] = lc[:,:,1]*lf
        lc[:,:,0] = lc[:,:,0]*lf
        
        ls[:,:,2] = ls[:,:,2]*lf2
        ls[:,:,1] = ls[:,:,1]*lf2
        ls[:,:,0] = ls[:,:,0]*lf2
            
        l = (ls+lc)
        
        P_stitch.append(l)
        
    return P_stitch

# Running
img_hand = cv.imread('hand.jpg')
img_eye = cv.imread('eye.jpg')
img_filter1 = cv.imread('filter.jpg', cv.IMREAD_GRAYSCALE) # eye
img_filter2 = 255 - img_filter1 # hand

img_hand = cv.resize(img_hand, img_filter1.shape[:2]) # resizing
img_eye = cv.resize(img_eye, img_filter1.shape[:2])

GP_hand = generate_gaussian_pyramid(img_hand, 6)
GP_eye = generate_gaussian_pyramid(img_eye, 6)
GP_filter1 = generate_gaussian_pyramid(img_filter1, 6)
GP_filter2 = generate_gaussian_pyramid(img_filter2, 6)

LP_hand = generate_laplacian_pyramid(GP_hand)
LP_eye = generate_laplacian_pyramid(GP_eye)

GP_stitch = stitchFliter(GP_hand, GP_eye, GP_filter1, GP_filter2)
LP_stitch = stitchFliter(LP_hand, LP_eye, GP_filter1, GP_filter2)


# plot image setting
def plot_img(index, img, title):
    plt.subplot(3, 6, index) # 3행 6열
    plt.imshow(img[...,::-1], cmap='gray') # a same as img[...,::-1]), RGB imgae is displayed withcout cv.cvtColor
    plt.axis('off'), plt.title(title)

recon_img = GP_stitch[-1] # array의 마지막 element, 가장 작은 img
lp_maxlev = len(LP_stitch) - 1

plot_img(6, recon_img.copy(), 'level: '+str(5))

for i in range(lp_maxlev, -1, -1): # maxlev ~ 0
    recon_img = cv.pyrUp(recon_img, None, LP_stitch[i].shape[:2])
    plot_img(i+1+12, recon_img.copy(), 'level: '+str(i))
    recon_img = cv.add(recon_img, LP_stitch[i])
    plot_img(i+1, recon_img.copy(), 'level: '+str(i))
    plot_img(i+1+6, LP_stitch[i].copy(), 'level: '+str(i))

cv.imshow('result', recon_img)
plt.show()
