from config import *
import numpy as np
import math


def normalize_landmarks(input_landmarks_set):
    result_set = []
    scales = []
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
        scales.append(scale)
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
            norm_landmarks, angle = calculate_landmark_rotation(ref, scaled_landmarks)
            result_set.append(norm_landmarks)
    print(np.mean(scales))
    print(np.std(scales))
    print("\n")
    return result_set


def normalize_landmark_with_data(rot_reference, input_landmarks):
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
    norm_landmarks, angle = calculate_landmark_rotation(rot_reference, scaled_landmarks)

    return norm_landmarks, (mean_x, mean_y), scale, angle


def restore_landmarks(landmarks, translation, scale, angle):
    result = []
    for landmark in landmarks:
        angle *= -1
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        new_landmark = np.array([landmark[0] * cos_angle - landmark[1] * sin_angle,
                        landmark[0] * sin_angle + landmark[1] * cos_angle])

        new_landmark *= scale
        new_landmark += translation
        result.append(new_landmark)
    return np.array(result)


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
    return result_landmarks, angle


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
        lm = normalize_landmarks(tooth_group)
        normalized_landmarks.append(lm)
    return normalized_landmarks
