import cv2
import numpy as np

cap = cv2.VideoCapture(0)
while (cap.isOpened()):
	ret,frame=cap.read()
	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces = cv2.CascadeClassifier('faces.xml')
	eyes=cv2.CascadeClassifier('eyes.xml')
	face_output=faces.detectMultiScale(gray) #work as predict
	eyes_output=eyes.detectMultiScale(gray)
	for (x, y, w, h) in face_output:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
		for (a, b, c, d) in eyes_output:
			cv2.rectangle(frame, (a, b), (a+c, b+d), (0, 255, 0), 2)

	cv2.imshow('title',frame)
	print(face_output)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows() 