import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
#img = cv2.imread('DSC.jpg')
cap = cv2.VideoCapture(0)

while cap.isOpened():
    _,img = cap.read()
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor = 1.1,minNeighbors = 4)    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),3)
        roi = gray[y:y+h,x:x+w]
        roi_color = img[y:y+h,x:x+w]
        eye = eye_cascade.detectMultiScale(roi)
        for (ex,ey,ew,eh) in eye:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,255),3)


        
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()