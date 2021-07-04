#**********************
#generating face encoding
#function for all the images
#dlib encode face into 128 unique points to detect the face
#HOG algorithm is used for encoding 
# def faceEncodings(images):
#     encodeList = []
#     for img in images:
#         #here cv2 read images so it shows in bgr format so convert into rgb format first
#         img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#         #picking first element of encodings and adding into encoding lists
#         encode = face_recognition.face_encodings(img)[0]
#         encodeList.append(encode)
#     return encodeList
