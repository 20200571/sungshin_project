import numpy as np
import cv2
xml='C:/ass/haarcascade_frontalface_default.xml'
face_cascade=cv2.CascadeClassifier(xml)
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
while(True):
    ret,frame=cap.read()
    frame=cv2.flip(frame,1)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.05,5)
    print("Number of faces detected: "+str(len(faces)))
    if len(faces):
        for (x,y,w,h) in faces:
            face_img=frame[y:y+h,x:x+w]
            face_img=cv2.resize(face_img,dsize=(0,0),fx=0.04,fy=0.04)
            face_img=cv2.resize(face_img,(w,h),interpolation=cv2.INTER_AREA)
            frame[y:y+h,x:x+w]=face_img
    cv2.imshow('result',frame)
    k=cv2.waitKey(30)&0xff
    if k==27:
        break
cap.release()
cv2.destroyAllWindws()
