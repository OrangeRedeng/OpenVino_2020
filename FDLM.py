import cv2 as cv
import numpy as np
from openvino.inference_engine import IENetwork, IECore
import argparse
import time

parser = argparse.ArgumentParser(description='Run face-detection and landmarks-regression with OpenVINO')
parser.add_argument('-m1', dest='model_1', default='face-detection-adas-0001', help='Path to the model1')
parser.add_argument('-m2', dest='model_2', default='landmarks-regression-retail-0009', help='Path to the model2')
args = parser.parse_args()

# Setup networks
net_1 = cv.dnn_DetectionModel(args.model_1 + '.xml', args.model_1 + '.bin')
net_2 = IENetwork(args.model_2 + '.xml', args.model_2 + '.bin')
    
# Load network to device
ie = IECore()
exec_net_2 = ie.load_network(net_2, 'GPU')

img_eye = cv.imread('eye5.png',cv.IMREAD_UNCHANGED)
height,width,depth = img_eye.shape  

# Reading the video
cap = cv.VideoCapture(0)
while(cap.isOpened()):

    # Reading the frame
    ret, frame = cap.read()
    if cv.waitKey(1) & 0xFF == ord('q') or ret == False:
        break

    start_time = time.time()
        
    # Perform an inference.
    ID, confidences, boxes = net_1.detect(frame, confThreshold=0.5)
    # Draw detected faces on the frame.
    for confidence, box in zip(list(confidences), boxes):
        cv.rectangle(frame, box, color=(0, 255, 0),thickness=(2))

        print("--- %s seconds ---" % (time.time() - start_time))

    # Getting the frame size
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        #Сhecking for faces in the frame
        if box[0] != 0:
    
            # Select face ROI 
            x10 = box[0]        
            y10 = box[1]
            x11 = box[0]+box[2]
            y11 = box[1]+box[3]
            k = y11 - y10
            k1 = x11 - x10

            if x10 < frame_width/2:
                if k > k1:
                    x11 = x10 + k
                else:
                    y11 = y10 + k1
            else:
                if k > k1:
                    x10 = x11 - k
                else:
                    y10 = y11 - k1
      
            img = frame[y10:y11, x10:x11]
        
            # Getting the image size
            img_height_width = img.shape[0]
 

            # Prepare input
            final_wide_shape = 48
            dim = (final_wide_shape, final_wide_shape)
            resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    
           
    
            inp = resized.transpose(2, 0, 1)  # interleaved to planar (HWC -> CHW)
            start_time = time.time()
            outs = exec_net_2.infer({'0':inp})
            out = next(iter(outs.values()))

            print("--- %s seconds ---" % (time.time() - start_time))

            # Eye1
            x = int(out[0][0][0][0] * img.shape[0])
            y = int(out[0][1][0][0] * img.shape[1])

        
            # Eye2
            x1 = int(out[0][2][0][0] * img.shape[0])
            y1 = int(out[0][3][0][0] * img.shape[1])

            # Select Eye1 ROI 
            if ((x > 18) and (y > 18) and ( x < img_height_width-18) and ( y < img_height_width-18)):
                eye1 = img[y-18:y+18, x-18:x+18]
                np.multiply(eye1, np.atleast_3d(255-img_eye[:, :, 3])/255.0, out=eye1, casting="unsafe")
                np.add(eye1, 255-img_eye[:, :, 0:3]* 255-np.atleast_3d(img_eye[:, :, 3]), out=eye1)   
                
                img[y-18:y+18, x-18:x+18] = eye1
                


            # Select Eye2 ROI 
            if ((x1 > 18) and (y1 > 18) and ( x1 < img_height_width-18) and ( y1 < img_height_width-18)):
                eye2 = img[y1-18:y1+18, x1-18:x1+18] 

                np.multiply(eye2, np.atleast_3d(255-img_eye[:, :, 3])/255.0, out=eye2, casting="unsafe")
                np.add(eye2, 255-img_eye[:, :, 0:3] * 255-np.atleast_3d(img_eye[:, :, 3]), out=eye2)   
                
                img[y1-18:y1+18, x1-18:x1+18] = eye2   

      
            #The other 3 points of the face
            x2 = int(out[0][4][0][0] * img.shape[0])
            y2 = int(out[0][5][0][0] * img.shape[1])

            cv.circle(img, (x2,y2),5,(0, 0, 255),-1)

            x3 = int(out[0][6][0][0] * img.shape[0])
            y3 = int(out[0][7][0][0] * img.shape[1])

            cv.circle(img, (x3,y3),5,(0, 0, 255),-1)

            x4 = int(out[0][8][0][0] * img.shape[0])
            y4 = int(out[0][9][0][0] * img.shape[1])

            cv.circle(img, (x4,y4),5,(0, 0, 255),-1)

            #Put text on the frame
            text="ID"+str(ID[0])
            l = len(text)*14+2
            cv.rectangle(frame,(box[0],box[1]-26),(box[0]+l,box[1]),(0,255,0),-1)
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(frame,text,(box[0],box[1]-4), font, 0.8,(0,0,0),1,cv.LINE_AA)
        
            cv.imshow('ROI image', img)
            cv.imshow('eye1', eye1)
            cv.imshow('eye2', eye2)
        
            box[0] = 0
            x = 0
            x1 = 0
            
    cv.imshow('frame', frame)
cap.release()
cv.destroyAllWindows()
