import cv2 as cv

cap=cv.VideoCapture(0)

while(cap.isOpened()):
    ret,img=cap.read()
    if cv.waitKey(1) & 0xFF==ord('q') or ret==False:
        break
    flip_img_y=cv.flip(img,1)
    flip_img_x=cv.flip(img,0)
    flip_img_xy=cv.flip(img,-1)
    cv.imshow('img',img)
    cv.imshow('y-axis',flip_img_y)
    cv.imshow('x-axis',flip_img_x)
    cv.imshow('xy-axis',flip_img_xy)
cap.release()#освобождает оперативную память, занятую переменной cap.
cv.destroyAllWindows()#(закрывает все открытые в скрипте окна).
