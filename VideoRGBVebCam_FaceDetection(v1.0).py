import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
while(cap.isOpened()):
    ret, frame = cap.read()
    if cv.waitKey(1) & 0xFF == ord('q') or ret == False:
        break

    net = cv.dnn_DetectionModel('face-detection-adas-0001.xml',
                                'face-detection-adas-0001.bin')
  
    # Perform an inference.
    _, confidences, boxes = net.detect(frame, confThreshold=0.5)
    
    # Draw detected faces on the frame.
    for confidence, box in zip(list(confidences), boxes):
        cv.rectangle(frame, box, color=(0, 255, 0),thickness=(2))
        text="Artyom"
        l = len(text)*14+2
        cv.rectangle(frame,(box[0],box[1]-26),(box[0]+l,box[1]),(0,255,0),-1)
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(frame,text,(box[0],box[1]-4), font, 0.8,(0,0,0),1,cv.LINE_AA)
    
    cv.imshow('frame', frame)

cap.release()
cv.destroyAllWindows()
