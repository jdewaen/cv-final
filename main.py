import cv2
import math
import numpy as np
import colorsys
from importer import import_image, import_all_landmarks, \
    PROCESSED_RADIO_AMOUNT, TOTAL_RADIO_AMOUNT, TEETH_AMOUNT

USE_MIRRORED = True


def normalize_landmarks(input_landmarks_set):
    result_set = []
    for input_landmarks in input_landmarks_set:
        # normalize position
        mean_x = 0
        mean_y = 0
        for landmark in input_landmarks:
            mean_x += landmark[0]
            mean_y += landmark[1]
        mean_x /= len(input_landmarks)
        mean_y /= len(input_landmarks)

        origin_landmarks = []
        for landmark in input_landmarks:
            new_x = landmark[0] - mean_x
            new_y = landmark[1] - mean_y
            origin_landmarks.append((new_x, new_y))


        # normalize scale
        scale = 0
        for landmark in origin_landmarks:
            scale += landmark[0]**2 + landmark[1]**2
        scale = math.sqrt(scale / len(origin_landmarks))
        scaled_landmarks = []
        for landmark in origin_landmarks:
            new_x = landmark[0] / scale
            new_y = landmark[1] / scale
            scaled_landmarks.append((new_x, new_y))

        # normalize rotation
        if len(result_set) == 0:
            result_set.append(scaled_landmarks)
        else:
            ref = result_set[0]
            norm_landmarks = calculate_landmark_rotation(ref, scaled_landmarks)
            result_set.append(norm_landmarks)

    return result_set


def calculate_landmark_rotation(ref_landmarks, other_landmarks):
    top = 0
    bottom = 0
    for index, ref in enumerate(ref_landmarks):
        other = other_landmarks[index]
        top += other[0]*ref[1] - other[1]*ref[0]
        bottom += other[0]*ref[0] + other[1]*ref[1]
    angle = math.atan(top/bottom)
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    result_landmarks = []
    for cur in other_landmarks:
        new_x = cur[0] * cos_angle - cur[1] * sin_angle
        new_y = cur[0] * sin_angle + cur[1] * cos_angle
        result_landmarks.append((new_x, new_y))
    return result_landmarks

def group_landmarks_by_tooth(input_data):
    grouped_landmarks = []
    for tooth in range(0, TEETH_AMOUNT):
        grouped_landmarks.append([])

    for landmark_data in input_data:
        for tooth in range(0, TEETH_AMOUNT):
            grouped_landmarks[tooth].append(landmark_data[0][tooth])
            if USE_MIRRORED:
                grouped_landmarks[tooth].append(landmark_data[1][tooth])

    return grouped_landmarks


def preprocess_landmarks():
    landmarks_input = []
    for image in range(1, PROCESSED_RADIO_AMOUNT + 1):
        landmarks_input.append(import_all_landmarks(image))
    grouped_landmarks = group_landmarks_by_tooth(landmarks_input)
    normalized_landmarks = []
    for tooth_group in grouped_landmarks:
        normalized_landmarks.append(normalize_landmarks(tooth_group))
    return normalized_landmarks


def hsv_to_bgr(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(b * 255), int(g * 255), int(r * 255)

def display_overlaid_landmarks(landmarks, image):
    img = np.zeros((400, 400, 3), np.uint8)
    hue = 0
    for landmark in landmarks:
        for point in landmark:
            cv2.circle(img, (int(point[0] * 100 + 200), int(point[1] * 100 + 200)), 1, hsv_to_bgr(hue, 1, 1), 2)
        hue += 1.0 / (len(landmarks) + 1)
    cv2.imshow("2 " + str(image), img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

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


landmarks = preprocess_landmarks()
for i, lm in enumerate(landmarks):
    display_overlaid_landmarks(lm, i)
cv2.waitKey()
