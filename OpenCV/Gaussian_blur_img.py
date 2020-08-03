import cv2 as cv

#загружаем изобр
img=cv.imread('p.png')
gaus=cv.GaussianBlur(img,(9,9),10)""" (9,9)- размер ядра фильтра, 10- отклонение по оси x"""
cv.imshow('Original img',img)
cv.imshow('Gaussian blur',gaus)

cv.waitKey(0)
