import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# ---------------------------------------------------------------------------------------------
# myList will contain all the Images that are present in the 'ImagesAttendence' Folder
# path is a variable used to define the path to the folder where all the required images are stored
# images varible (list) will contain all the Images that mylist contains
# classnames variable (list) will contain only name of all the images that are present in the myList
# ---------------------------------------------------------------------------------------------
path = 'ImagesAttendence'
images = []
classNames = []
mylist = os.listdir(path)
print(mylist)

# ---------------------------------------------------------------------------------------------
# Loading images from folder 'ImgesAttendence'
# element will contain one image from mylist every time the loop executes
# ---------------------------------------------------------------------------------------------
for element in mylist:
    currentImage = cv2.imread(f'{path}/{element}')
    images.append(currentImage)
    classNames.append(os.path.splitext(element)[0])
print(classNames)

#------------------------------------------------------------------------------------------------------
# Encoding Images
# findEncodings method will generate the encodings for a image i.e. it will generate all the 128 unique values for a perticular image
#encodeList will be generated containing encodings for all the images that are available
#finally when the method is called, the values from encode list will be stored into encodeListOfKnownFaces
#encodeListOfKnownFaces - This variable only stores the encodings of those images whose encodings are generated from findEncodings method
#findEncodings method accesses the ImagesAttendence folder for the images
#-------------------------------------------------------------------------------------------------------
def findEncodings(images):
    encodeList = []
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(image)[0]
        encodeList.append(encode)
    return encodeList

encodeListOfKnownFaces = findEncodings(images)
print('Encoding Complete')

#---------------------------------------------------------------------------------------
#markAttendence function marks the attendence of the students whose faces are correctly determined in the webcam
#The attendence is stored in the csv file
#---------------------------------------------------------------------------------------
def markAttendence(name):
    with open('Attendence.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

#Initializing the webcam
capture = cv2.VideoCapture(0)

#------------------------------------------------------------------------------------------------
#Following while loop will help in comparing the webcam image encodings with the list of encodings of images that are available
#img is the image that is captured from webcam
#imgSmall is the resized webcam image which is done for faster execution
#facesCurrentFrame is used to detect all the images that are currently visible in the webcam
#encodesCurrentFrame is used to generate the encodis for all the images that are detected in the webcam
#the for loop will find the matches of the images that are present in the webcam with the images that are in ImagesAttendence folder with their respective encodings
#faceDistance is generated after comparing the images to find the best match possible
#minimum faceDistance is selected as the correct match
#----------------------------------------------------------------------------------------------------
while True:
    success, img = capture.read()
    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(imgSmall)
    encodesCurrentFrame = face_recognition.face_encodings(imgSmall, facesCurrentFrame)

    for encodeFace, faceLocation in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListOfKnownFaces, encodeFace)
        faceDistance = face_recognition.face_distance(encodeListOfKnownFaces, encodeFace)
        # print(faceDistance)
        matchIndex = np.argmin(faceDistance)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLocation
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendence(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
