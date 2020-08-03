import cv2 as cv

img=cv.imread('p.png')

flip_img_y=cv.flip(img,1)
flip_img_x=cv.flip(img,0)
flip_img_xy=cv.flip(img,-1)

cv.imshow('original img',img)
cv.imshow('flip around the y-axis',flip_img_y)
cv.imshow('flip around the x-axis',flip_img_x)
cv.imshow('flip around the xy-axis',flip_img_xy)
cv.waitKey(0)
