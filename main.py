import cv2
import mediapipe as mp
import time

# import image from webcam

cap = cv2.VideoCapture(0)

# Create a mediapipe instance for hands 

mpHands = mp.solutions.hands

# Check parameters, add them if needed

hands = mpHands.Hands()

# create an instance of drawing method for detected objects

mpDraw = mp.solutions.drawing_utils


while True:
	success, img = cap.read()

	imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	# this is a method to process img in the instance we created
	results = hands.process(imgRGB)

	# print the location of a hand if detected

	# print(results.multi_hand_landmarks)

	# draw lines and dots
	if results.multi_hand_landmarks:
		for handLms in results.multi_hand_landmarks:
			# handLms shows dots, the second one shows lines
			mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

	cv2.imshow('Image', img)
	cv2.waitKey(1)