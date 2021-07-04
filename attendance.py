import sys
#To get Camera application and stuff we use CV2
import cv2
#find reason for numpy???
import numpy as np
#face recognition for recognising face and all
import face_recognition
#to read images
import os
#to mark attendance we need datetime
from datetime import datetime
#to prepare dataset / to transfer data into files
import pickle


# to read out all images and bring here first
# and also to make extract names from image file name 
path = 'Images' #images path saved
#list to keep all images and names of persons
#Other words DataSet
images = []
personName = []
#to read components of the path given here
#i.e., extracting all the images from images folder(path stored in path variable)
myList = os.listdir(path)
#print(myList)



#looping through the list for each image/component
for cu in myList:
    #extracting the path of current image using cv2 and f-string
    current_img = cv2.imread(f'{path}/{cu}')
    #appending them into images list
    images.append(current_img)

    #Names extracting now
    #from os.path we get image full name i.e, shreya.jpg(cu) 
    #in which splittext makes it to be shreya-->0 and jpg-->1
    #we need name so taking [0] index
    personName.append(os.path.splitext(cu)[0])
#print(personName) checking list whether names appended correctly




### taking data from data set
with open('dataset_faces.dat','rb') as f:
    encodeListKnown = pickle.load(f)





#*************************
#Attendance Marking for tha faces found on camera
def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        
        #if name doesn't exists in list we add attendance now
        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')    #Time format
            dStr = time_now.strftime('%d/%m/%Y')    #Date Format
            f.writelines(f'\n{name},{tStr},{dStr}')


#**********************
#camera reading
#for laptop give id-->0
#for external camera use id-->1
cap = cv2.VideoCapture(0) 

while True:
    ret,frame = cap.read()
    faces = cv2.resize(frame, (0,0),None,0.25,0.25) #resizing the frame from camera input
    faces = cv2.cvtColor(faces,cv2.COLOR_BGR2RGB)  #converting bgr faces from camera to rgb , since using cv2 we are taking

    #face location finding
    facesCurrentFrame = face_recognition.face_locations(faces) #storing all faces
    encodesCurrentFrame = face_recognition.face_encodings(faces,facesCurrentFrame) #face encoding both faces and faces on frames

    #face comparisions and face distances finding
    for encodeFace,faceLoc in zip(encodesCurrentFrame,facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)   # already encoded faces and current encoded faces sending
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)


        matchIndex = np.argmin(faceDis) # minimum distance index values is extracted
        y1,x2,y2,x1 = faceLoc
        y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        if matches[matchIndex]:
            name = personName[matchIndex]
            #print(name)
            cv2.rectangle(frame,(x1, y2),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(frame,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.6,(0,0,255),1)
            markAttendance(name)    #calling attendance function
        else:
            cv2.rectangle(frame,(x1, y2),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(frame,"UN_KNOWN",(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.6,(0,0,255),1)

    cv2.imshow("Camera",frame)
    if cv2.waitKey(10) == 13:
        break
cap.release()
cv2.destroyAllWindows()


        