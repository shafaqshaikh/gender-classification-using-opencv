import cv2
import numpy as np
import os.path
from cv2 import WINDOW_NORMAL
from face_detector import find_faces

ESC = 27

def analyze_picture(model_gender, path, window_size, window_name='static'):
    cv2.namedWindow(window_name, WINDOW_NORMAL)
    cv2.namedWindow(window_name, WINDOW_NORMAL)
    if window_size:
        width, height = window_size
        cv2.resizeWindow(window_name, width, height)

    image = cv2.imread(path, 1)
    for normalized_face, (x, y, w, h) in find_faces(image):
        gender_prediction = model_gender.predict(normalized_face)
        if (gender_prediction[0] == 0):
            cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,255), 2)
        else:
            cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
        text1="Male"
        text2="Female"
        if (gender_prediction[0] == 0):
            cv2.putText(image,text2,(x,y-10), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
        else:
            cv2.putText(image,text1,(x,y-10), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
    cv2.imshow(window_name, image)
    key = cv2.waitKey(0)
    if key == ESC:
        cv2.destroyWindow(window_name)

if __name__ == '__main__':
    fisher_face_gender = cv2.face.FisherFaceRecognizer_create()
    fisher_face_gender.read('models/gender_classifier_model.xml')
    run_loop = True
    window_name = "Photo Upload (press ESC to exit)"
    print("Default path is set to ../test_sample/")
    print("Type q or quit to end program")
    while run_loop:
        path = "../test_sample/"
        file_name = input("Specify image file: ")
        if file_name == "q" or file_name == "quit":
            run_loop = False
        else:
            path += file_name
            if os.path.isfile(path):
                analyze_picture(fisher_face_gender, path, window_size=(1280, 720), window_name=window_name)
            else:
                print("File not found!")
else:
    print("Invalid input, exiting program.")

