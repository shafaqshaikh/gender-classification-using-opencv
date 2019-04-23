from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap

from PyQt5 import QtCore,QtGui,QtWidgets,uic
from tkinter import filedialog
from tkinter import * 
import tkinter as tk
import keyboard
import cv2
import numpy as np
import os.path
from cv2 import WINDOW_NORMAL
from face_detector import find_faces

ESC = 27
name=""

class MyWindow(QtWidgets.QMainWindow):
	def __init__(self):
		super(MyWindow,self).__init__()
		uic.loadUi('upload.ui',self)
		self.upload.clicked.connect(self.open)
		self.webcam.clicked.connect(self.web)

	def open(self):
		root=Tk()
		root.withdraw()
		root.filename =  filedialog.askopenfilename(initialdir = "C:\\Users\\Salman\\Documents\\gender_rec_opencv\\test_sample",title = "Select file",filetypes = (("Image File","*.jpg"),("all files","*.*")))
		name=root.filename
		
		fisher_face_gender = cv2.face.FisherFaceRecognizer_create()
		fisher_face_gender.read('models/gender_classifier_model.xml')
		run_loop = True
		if os.path.isfile(name):
			print('true')
			image = cv2.imread(name, 1)
			
			image1 = cv2.imread(name, 1)
			image1=cv2.resize(image1,(361 ,301),cv2.COLOR_BGR2RGB) 
			image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
			height2, width2, channel2 = image1.shape
			step2 = channel2 * width2
			qImg2 = QImage(image1.data, width2, height2, step2, QImage.Format_RGB888)
			self.label.setPixmap(QPixmap.fromImage(qImg2))
		
			for normalized_face, (x, y, w, h) in find_faces(image):
				gender_prediction = fisher_face_gender.predict(normalized_face)
				if (gender_prediction[0] == 0):
					cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,255), 2)
				else:
					cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
				if (gender_prediction[0] == 0):
					self.textBrowser.setText("Female")
				else: 
					self.textBrowser.setText("Male")

			key = cv2.waitKey(0)
			if key == ESC:
				cv2.destroyWindow(window_name)

		else:
			print('false')

	def web(self):
            model_gender = cv2.face.FisherFaceRecognizer_create()
            model_gender.read('models/gender_classifier_model.xml')
            window_size=(1280, 720)
            window_name='live'
            update_time=1
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

if __name__=='__main__':
	import sys

	app=QtWidgets.QApplication(sys.argv)
	window=MyWindow()
	window.show()
	print(name)
	sys.exit(app.exec())

