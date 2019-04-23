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
if __name__=='__main__':
	import sys

	app=QtWidgets.QApplication(sys.argv)
	window=MyWindow()
	window.show()
	print(name)
	sys.exit(app.exec())

