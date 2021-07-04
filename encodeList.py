import cv2
import face_recognition
import os
import pickle

path = 'Images'
images = []
myList = os.listdir(path)
personName = []

for curr in myList:
    curent_img = cv2.imread(f'{path}/{curr}')
    images.append(curent_img)
    #personName.append(os.path.splitext(curr)[0])

all_encodings = []
i=1
for img in images:
        #here cv2 read images so it shows in bgr format so convert into rgb format first
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        print(f"Done{i}!")
        i=i+1
        #picking first element of encodings and adding into encoding lists
        encode = face_recognition.face_encodings(img)[0]
        all_encodings.append(encode)
#print(all_encodings)

with open('dataset_faces.dat','wb') as f:
    pickle.dump(all_encodings,f)