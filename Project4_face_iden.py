import cv2, sys, numpy, os, time
import pickle
fn_haar = 'C:\\Users\\sneha\Desktop\\AttendenceProject\\cascades\\data\\haarcascade_frontalface_default.xml'
fn_dir = 'Images'
im_width= 224
im_height=224
faceDetect =cv2.CascadeClassifier(fn_haar)
video_capture=cv2.VideoCapture(0)
#rec= cv2.face_FisherFaceRecognizer()
rec=cv2.face.FisherFaceRecognizer_create()
x=rec.read('test.yml')

labels = {"persons-name":1}
with open("labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = og_labels.items()

print (x)
id=0
font=cv2.FONT_HERSHEY_COMPLEX
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #faces = faceDetect.detectMultiScale(gray,1.3,5)

    faces = faceDetect.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (im_width, im_height))
        #id,conf=rec.predict(gray[y:y+h,x:x+w])
        id, conf = rec.predict(face_resize)
        print(id)
        print (conf)
    #print the names
        if (id==4):
            id='sambit'
        

        cv2.putText(frame,str(id),(x,y+h),font,1,(50,205,50),2,cv2.LINE_4)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()