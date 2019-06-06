
####
####    UNUSED FILE!!!!!!!!!
####

from time import sleep

import cv2
import cv2 as cv

cap1 = cv2.VideoCapture("./videos/5.MOV")
while not cap1.isOpened():
    cap1 = cv2.VideoCapture("./videos/4.MOV")
    cv2.waitKey(1000)
    print("Wait for the header")

cap2 = cv2.VideoCapture("traffic_light_detector/1.mp4")
while not cap2.isOpened():
    cap2 = cv2.VideoCapture("traffic_light_detector/1.mp4")
    cv2.waitKey(1000)
    print("Wait for the header")


while cap1.isOpened():
    flag1, frame1 = cap1.read()
    flag2, frame2 = cap2.read()
    if not flag1 or not flag2:
        break;

    if cv2.waitKey(10) == ord('s'):
        print('saving image')
        cv2.imwrite('images/' + str(i) + 'image.bmp', frame1)
        i = i + 1
    cv2.imshow('1', frame1)
    cv2.imshow('2', frame2)
    sleep(0.1)