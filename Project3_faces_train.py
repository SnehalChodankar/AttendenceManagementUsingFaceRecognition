import os
from PIL import Image
import numpy as np
import cv2
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "Images")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

currentID = 0
labelIDs = {}
yLabels = []
xTrain = []


for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
            path = os.path.join(root, file)
            
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()

            print(label, path)

            if not label in labelIDs:
                labelIDs[label] = currentID
                currentID += 1
            id_ = labelIDs[label]
            print(labelIDs)
            #yLabels.append(label)
            #xTrain.append(path)

            pilImage = Image.open(path).convert("L")
            size = (1000, 1000)
            finalImage = pilImage.resize(size, Image.ANTIALIAS)
            imageArray = np.array(finalImage, "uint8")
            #print(imageArray)

            faces = face_cascade.detectMultiScale(imageArray)

            for (x,y,w,h) in faces:
                roi = imageArray[y:y+h, x:x+w]
                xTrain.append(roi)
                yLabels.append(id_)

with open("labels.pickle", "wb") as f:
    pickle.dump(labelIDs, f)

recognizer.train(xTrain, np.array(yLabels))
recognizer.save("trainner.yml")
