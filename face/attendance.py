import cv2
import numpy as np
import face_recognition
import os
path='Aimage'
images=[]
classNames=[]
mylist=os.listdir(path)

for cl in mylist:
    curimg=cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encode_list=[]
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list

encodeListKnown=findEncodings(images)
print("encoding complete")

p=[]
cap=cv2.VideoCapture(0)
while True:
    success,img=cap.read()
    imgs=cv2.resize(img,(0,0),None,0.25,0.25)
    imgs = cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)

    facesCurFrame=face_recognition.face_locations(imgs)
    encodesCurFrame=face_recognition.face_encodings(imgs,facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)
        matchIndex=np.argmin(faceDis)
        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            if name not in p:
                p.append(name)
                print(name)
                
            

    cv2.imshow('webcam',img)
    cv2.waitKey(1)



