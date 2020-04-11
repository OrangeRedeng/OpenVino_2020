import cv2 as cv

cap = cv.VideoCapture(0)
while(cap.isOpened()):
    ret, frame = cap.read()
    if cv.waitKey(1) & 0xFF == ord('q') or ret == False:
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    net = cv.dnn_DetectionModel('face-detection-adas-0001.xml',
                                'face-detection-adas-0001.bin')
  
    # Perform an inference.
    _, confidences, boxes = net.detect(frame, confThreshold=0.5)
    
    # Draw detected faces on the frame.
    for confidence, box in zip(list(confidences), boxes):
        cv.rectangle(gray, box, color=(0, 255, 0))   
    cv.imshow('frame', gray)

cap.release()
cv.destroyAllWindows()
