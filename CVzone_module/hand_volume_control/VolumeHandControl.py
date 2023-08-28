import cv2
import mediapipe as mp

wCam, hCam=640,480

cap=cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

while True:
    success, img =cap.read()
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break