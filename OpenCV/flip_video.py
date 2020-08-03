import cv2 as cv

cap = cv.VideoCapture('video.mp4')#создаем экземпляр клаас и присваивае его перменной,
ret, img = cap.read()#нужна ли воодбще эта строчка здесь

while (cap.isOpened()):#функция изопенд будет возварщать каждый раз тру,пока не дойдет до конца файла/выход из цикла будет,когда функция вернет фолс,либо по брейку
    ret, img = cap.read()#функция рид возвращает лбо тру,либо фолс/это значение запишем в переменнуб ret, а текущий кадр запишем в перменную фрейм
    if cv.waitKey(1) & 0xFF == ord('q') or ret==False:#функция рид возвращает лбо тру,либо фолс/это значение запишем в переменнуб ret, а текущий кадр запишем в перменную фрейм
        break
    flip_img_y=cv.flip(img,1)
    flip_img_x=cv.flip(img,0)
    flip_img_xy=cv.flip(img,-1)
    cv.imshow('Source image', img)
    cv.imshow('flip around the y-axis',flip_img_y)
    cv.imshow('flip around the x-axis',flip_img_x)
    cv.imshow('flip around the xy-axis',flip_img_xy)
cap.release()#освобождает оперативную память, занятую переменной cap.
cv.destroyAllWindows()#(закрывает все открытые в скрипте окна).

