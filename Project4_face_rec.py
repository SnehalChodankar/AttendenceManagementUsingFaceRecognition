# facerec.py
#import cv2, sys, numpy, os, pyttsx, time
import cv2, numpy, os
from PIL import Image
import pickle
size = 4
fn_haar = 'haarcascade_frontalface_default.xml'
fn_dir = 'Images'

folder_pointer=0
folder_name='xxxxxx'

# Part 1: Create fisherRecognizer
print('Training...')
# Create a list of images and a list of corresponding names
(images, lables, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(fn_dir):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(fn_dir, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            #print(path)
            print(subdir)
            if (folder_name != subdir):
                folder_name=subdir
                folder_pointer+=2
                print(folder_pointer)
            lable = id
            mat_point=folder_pointer
            imgFile = cv2.imread ( path,0 )
            #showing the images and read in the numpy array as cv2 only accepts numpy array
            imread=Image.open(path)
            cv2.imshow("Training..",numpy.array(imread,'uint8'))
            cv2.waitKey(10)
	        #images.append(cv2.imread(path,0))
            imResize=cv2.resize(imgFile,(224,224))
            images.append(imResize)
            #lables.append(int(lable))
            lables.append(int(mat_point))
            id += 1
#(im_width, im_height) = (224, 224)

# Create a Numpy array from the two lists above
(images, lables) = [numpy.array(lis) for lis in [images, lables]]
print('X')
#print(lables)
#print(images)
# OpenCV trains a model from the images

#model = cv2.face.createFisherFaceRecognizer()
#model.train(images, lables)
with open("labels.pickle", "wb") as f:
    pickle.dump(names, f)

model = cv2.face.FisherFaceRecognizer_create()
#model = cv2.FisherFaceRecognizer_create()
model.train(images, lables)
model.write('test.yml')
print('Model Saved.......')
cv2.destroyAllWindows()