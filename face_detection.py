import cv2
import os
import numpy as np
import json
from PIL import Image
from face_trainer import *

def write_json(new_data, filename='database.json'):
    with open(filename,'r+') as file:
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data["people"].append(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent = 4)

def face_detection():
    #file handling
    names = []
    with open('database.json') as f:
        data = json.load(f)
    #get all the names in database into a list called names
    for person in data['people']:
        names.append(person['name'])

    #face_detection
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # For each person, enter one numeric face id
    face_id = len(names)
    name=input('\n enter name of person to be blacklisted')
    add=input('\n address of blacklisted person')
    phoneno = int(input('\n phone number of blacklisted person'))
    adhar=int(input('\n adhar number of blacklisted person'))
    reason=input('\n reason for blacklisting?')

    y = {"name":name,
         "address":add,
         "phone number": phoneno,
         "adhar card number" :adhar,
         "reason for blacklist":reason
         }

    write_json(y)


    print("\n [INFO] Initializing face capture. Look the camera and wait ...")
    # Initialize individual sampling face count
    count = 0
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1
            # Save the captured image into the datasets folder
            cv2.imwrite("dataset/User." + str(face_id) + '.' +
                        str(count) + ".jpg", gray[y:y + h, x:x + w])
            cv2.imshow('image', img)
        k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 30:  # Take 30 face sample and stop video
            break
    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()
    face_trainer()