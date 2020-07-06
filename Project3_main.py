import numpy as np
import cv2
import pickle
#import urllib.request as ur
#import time

#URL = "http://192.168.1.102:8080"



face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {"persons-name":1}
with open("labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while True:
    #imgArr = np.array(bytearray(ur.urlopen(URL).read()), dtype=np.uint8)
    #img = cv2.imdecode(imgArr, -1)

    #cv2.imshow('IPWebcam', cv2.resize(img,(600,400)))
    
    ret, frame = cap.read()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=2)
    
    for (x,y,w,h) in faces:
        roiColor = frame[y:y+h, x:x+w]
        roiGray = gray[y:y+h, x:x+w]

        id_, conf = recognizer.predict(roiGray)
        if conf>=60 and conf<=85:
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
        imgItem = "myImage.png"
        cv2.imwrite(imgItem,roiGray)
    
        color = (255,0,0)
        stroke = 2
        endCordX=x+w
        endCordY=y+h
        cv2.rectangle(frame, (x,y), (endCordX, endCordY), color, stroke)

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()