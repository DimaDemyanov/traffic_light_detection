
####
####    UNUSED FILE!!!!!!!!!
####

from time import sleep

import numpy as np
#from cv2 import cv2
import cv2 as cv

import hmm_my
from hmm_my import get_model


def draw_contours(img):  # меняем цветовую модель с BGR на HSV
    min = (0, 0, 0)
    max = (160, 255, 255)
    thresh = cv.inRange(img, min, max)  # применяем цветовой фильтр
    # ищем контуры и складируем их в переменную contours
    _, contours, hierarchy = cv.findContours(thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # отображаем контуры поверх изображения
    cv.drawContours(img, contours, -1, (255, 0, 0), 3, cv.LINE_AA, hierarchy, 1)

# input img must be in HSV
def get_active_color(img, green_pos, yellow_pos, red_pos):
    colors = []
    min = (0, 0, 0)
    max = (180, 100, 255)
    thresh = cv.inRange(img, min, max)  # применяем цветовой фильтр
    if thresh[green_pos[1]][green_pos[0]] > 0:
        colors.append('green')
    if thresh[yellow_pos[1]][yellow_pos[0]] > 0:
        colors.append('yellow')
    if thresh[red_pos[1]][red_pos[0]] > 0 :
        colors.append('red')
    cv.imshow('sory', thresh)
    return colors

def draw_traffic_light(img, colors, x0, y0):
    width = 35
    lengh = 100
    cv.rectangle(img, (x0, y0), (x0 + width, y0 + lengh), (0,0,0), thickness=-1)
    if 'red' in colors:
        cv.circle(img, (x0 + int(width/2), y0 + int(lengh / 6)), int(lengh / 6), (0, 0, 255), thickness=-1)
    if 'yellow' in colors:
        cv.circle(img, (x0 + int(width / 2), y0 + int(lengh * 3 / 6)), int(lengh / 6), (0, 255, 255), thickness=-1)
    if 'green' in colors:
        cv.circle(img, (x0 + int(width / 2), y0 + int(lengh * 5 / 6)), int(lengh / 6), (0, 255, 0), thickness=-1)
    if 'blue' in colors:
        cv.circle(img, (x0 + int(width / 2), y0 + int(lengh * 5 / 6)), int(lengh / 6), (255, 0, 0), thickness=-1)

if __name__ == "__main__":
    cap = cv.VideoCapture("./videos/4.MOV")
    while not cap.isOpened():
        cap = cv.VideoCapture("./videos/4.MOV")
        cv.waitKey(1000)
        print("Wait for the header")

    pos_frame = cap.get(1)

    i = 0

    colors_n = []
    hmm = get_model()

    while cap.isOpened():
        flag, frame = cap.read()

        if flag:
            frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            # print(str(pos_frame)+" frames")

            # Is 's' was pressed, it saves current frame
            if cv.waitKey(10) == ord('s'):
                print('saving image')
                cv.imwrite('images/' + str(i) + 'image.bmp', frame)
                i = i + 1

            # Is 'a' was pressed, it exits
            # print(frame.type())


           # img = Image.fromarray(frame_hsv)
            frame_grayscale = cv.cvtColor(frame_hsv, cv.COLOR_BGR2GRAY)
            # cv.imshow('gray_scale', frame_grayscale)
            #
            # circles = cv.HoughCircles(frame_grayscale, cv.HOUGH_GRADIENT, 1, 60, param1=20, param2=30, minRadius=20, maxRadius=40)

            # if circles is not None:
            #     circles = np.uint16(np.around(circles))
            #     for i in circles[0, :]:
            #         cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
            #
            #
            # colors = []
            # get_active_color(frame_hsv, green_pos, yellow_pos, red_pos)
            # ищем контуры и складируем их в переменную contours
            #_, contours, hierarchy = cv.findContours(thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            # отображаем контуры поверх изображения
            #cv.drawContours(frame, contours, -1, (255, 0, 0), 3, cv.LINE_AA, hierarchy, 1)
            #cv.circle(frame, (20, 20), 15, )
            # draw_traffic_light(frame, colors, 0, 0)

            # if 'green' in colors:
            #     colors_n.append('green')
            # else:
            #     if 'yellow' in colors and 'red' in colors:
            #         colors_n.append('yellow_red')
            #     else:
            #         if 'yellow' in colors:
            #             colors_n.append('yellow')
            #         else:
            #             if 'red' in colors:
            #                 colors_n.append('red')
            #             else:
            #                 colors_n.append('black')
            # #if len(colors_n) > 20:
            #
            # colors1 = ['red', 'yellow', 'green', 'black', 'yellow_red']
            #
            # x = hmm.predict(np.array([colors1.index(o) for o in colors_n]).reshape(-1, 1))
            # draw_traffic_light(frame, hmm_my.state_color[hmm_my.states[x[0]]], 200, 0)
            # colors_n.pop(0)
            #
            # print(hmm_my.states[x[0]])
            # cv.circle(frame, green_pos, 5, (0, 255, 0), thickness=-1);
            # cv.circle(frame, yellow_pos, 5, (0, 255, 255), thickness=-1);
            # cv.circle(frame, red_pos, 5, (0, 0, 255), thickness=-1);
            cv.imshow('contours', frame)  # выводим итоговое изображение в окно
            #sleep(0.05)

        else:
            # The next frame is not ready, so we try to read it again
            break

