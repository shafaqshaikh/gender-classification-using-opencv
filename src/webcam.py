import cv2
import numpy as np
import os.path
import sys
from cv2 import WINDOW_NORMAL
from face_detector import find_faces

ESC = 27

def start_webcam(model_gender, window_size, window_name='live', update_time=1):
    cv2.namedWindow(window_name, WINDOW_NORMAL)
    if window_size:
        width, height = window_size
        cv2.resizeWindow(window_name, width, height)

    video_feed = cv2.VideoCapture(0)
    video_feed.set(3, width)
    video_feed.set(4, height)
    read_value, webcam_image = video_feed.read()

    delay = 0
    init = True
    while read_value:
        read_value, webcam_image = video_feed.read()
        for normalized_face, (x, y, w, h) in find_faces(webcam_image):
            if init or delay == 0:
                init = False
                gender_prediction = model_gender.predict(normalized_face)
            if (gender_prediction[0] == 0):
                cv2.rectangle(webcam_image, (x,y), (x+w, y+h), (0,0,255), 2)
            else:
                cv2.rectangle(webcam_image, (x,y), (x+w, y+h), (255,0,0), 2)
            text1="Male"
            text2="Female"
            if (gender_prediction[0] == 0):
                cv2.putText(webcam_image,text2,(x,y-10), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
            else:
                cv2.putText(webcam_image,text1,(x,y-10), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
        delay += 1
        delay %= 20
        cv2.imshow(window_name, webcam_image)
        key = cv2.waitKey(update_time)
        if key == ESC:
            break


    cv2.destroyWindow(window_name)

if __name__ == '__main__':
    fisher_face_gender = cv2.face.FisherFaceRecognizer_create()
    fisher_face_gender.read('models/gender_classifier_model.xml')
    choice = input("Use webcam?(y/n) n will exit program \n")
    if (choice == 'y'):
        window_name = "Webcam(press ESC to exit)"
        start_webcam(fisher_face_gender, window_size=(1280, 720), window_name=window_name, update_time=1)
    elif(choice == 'n'):
        sys.exit()
