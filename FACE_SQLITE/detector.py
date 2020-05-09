import cv2
import numpy as np
import sqlite3

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read('recognizer/trainingData.yml')
path='dataSets'

def getProfile(id):
    conn = sqlite3.connect("FaceBase.db")
    cmd = "SELECT * FROM People WHERE ID="+str(id)
    cursor = conn.execute(cmd)
    profile = None
    for row in cursor:
        profile=row
    conn.close()
    return profile

id=0
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontColor = (0,0,255)
while (True):
    ret,img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray,scaleFactor=1.3,
                                        minSize=(100,100))
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        id,conf = rec.predict(gray[y:y+h,x:x+w])
        profile = getProfile(id)
        if(profile!=None):
            cv2.putText(img,str(profile[1]),(x,y+h+30),fontface,fontscale,fontColor,2,cv2.LINE_AA)
            cv2.putText(img,str(profile[2]),(x,y+h+60),fontface,fontscale,fontColor,2,cv2.LINE_AA)
            cv2.putText(img,str(profile[3]),(x,y+h+90),fontface,fontscale,fontColor,2,cv2.LINE_AA)
            cv2.putText(img,str(profile[4]),(x,y+h+120),fontface,fontscale,fontColor,2,cv2.LINE_AA)
    cv2.imshow("Faces",img)
    if(cv2.waitKey(1)==ord('q')):
        break

cam.release()
cv2.destroyAllWindows()
