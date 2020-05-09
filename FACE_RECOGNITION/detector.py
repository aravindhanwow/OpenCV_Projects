import cv2
import numpy as np

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read('recognizer/trainingData.yml')
id=0
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1

fontColor = (255,255,255)
while (True):
    ret,img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        id,conf = rec.predict(gray[y:y+h,x:x+w])
        if id==1:
            id="Aravind"
        elif id==2:
            id="amma"
        elif id==3:
            id="dheeran"
        elif id==4:
            id="abi"
        elif id==5:
            id="Appa"
        elif id==6:
            id="vijay"
        elif id==7:
            id="ajith"
        elif id==8:
            id="dhanush"
        elif id==9:
            id="surya"
        elif id==10:
            id="yash"
        cv2.putText(img,str(id),(x,y+h),fontface, fontscale, fontColor)
    cv2.imshow("Faces",img)
    if(cv2.waitKey(1)==ord('q')):
        break

cam.release()
cv2.destroyAllWindows()
