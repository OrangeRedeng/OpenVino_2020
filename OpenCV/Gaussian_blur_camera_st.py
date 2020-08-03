import cv2 as cv

cap=cv.VideoCapture(0)

while(cap.isOpened()):
    ret,img=cap.read()
    if cv.waitKey(1) & 0xFF==ord('q') or ret==False:
        break
    gaus=cv.GaussianBlur(img,(9,9),10)
    cv.imshow('source img',img)
    cv.imshow('blur',gaus)
    
cap.release()#освобождает оперативную память, занятую переменной cap.
cv.destroyAllWindows()#(закрывает все открытые в скрипте окна).
