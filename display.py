import cv2
import colorsys
from config import *
from importer import import_image
import numpy as np


def hsv_to_bgr(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(b * 255), int(g * 255), int(r * 255)


def display_all_overlaid_landmarks(landmarks):
    imgs = []
    for tooth in landmarks:
        img = np.zeros((400, 250, 3), np.uint8)
        hue = 0
        for landmark in tooth:
            for point in landmark:
                cv2.circle(img, (int(point[0] * 100 + 125), int(point[1] * 100 + 200)), 1, hsv_to_bgr(hue, 1, 1), 2)
            hue += 1.0 / (len(tooth) + 1)
        imgs.append(img)
    cv2.imshow('img', np.vstack((
        np.hstack(imgs[0:TEETH_AMOUNT / 2]),
        np.hstack(imgs[(TEETH_AMOUNT / 2):TEETH_AMOUNT + 1])
         )))
    cv2.waitKey()
    cv2.destroyAllWindows()


def display_side_by_side_landmarks(landmarks):
    imgs = []
    for tooth in landmarks:
        img = np.zeros((400, 250, 3), np.uint8)
        for point in tooth:
            cv2.circle(img, (int(point[0] * 100 + 125), int(point[1] * 100 + 200)), 1, (0, 0, 255), 2)
        imgs.append(img)
    cv2.imshow('img', np.hstack(imgs))
    cv2.waitKey()
    cv2.destroyAllWindows()


def display_input():
    for image in range(1, TOTAL_RADIO_AMOUNT + 1):
        image_data = import_image(image)
        radiograph = image_data['radiograph']
        if image_data['full_data']:
            for tooth in image_data['original_landmarks']:
                for landmark in tooth:
                    cv2.circle(radiograph, (int(landmark[0]), int(landmark[1])), 2, (0, 0, 255), 2)
        cv2.imshow(str(image), cv2.resize(radiograph, (0, 0), fx=0.5, fy=0.5))
        cv2.waitKey()
        cv2.destroyWindow(str(image))