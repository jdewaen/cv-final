from config import *
import math


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


def process_landmarks(landmarks_input):
    grouped_landmarks = group_landmarks_by_tooth(landmarks_input)
    normalized_landmarks = []
    for tooth_group in grouped_landmarks:
        normalized_landmarks.append(normalize_landmarks(tooth_group))
    return normalized_landmarks
