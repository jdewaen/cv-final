import cv2
import os
from config import *

WORKING_DIR = os.path.dirname(__file__)


def import_landmarks(image, tooth_number, mirrored=False):
    directory = WORKING_DIR + LANDMARK_DIR
    if mirrored:
        directory += LANDMARK_MIRRORED_DIR
        image += PROCESSED_RADIO_AMOUNT
    else:
        directory += LANDMARK_DEFAULT_DIR
    file_loc = directory + LANDMARK_FILE_TEMPLATE
    file_loc = file_loc.replace("{image}", str(image))
    file_loc = file_loc.replace("{tooth_number}", str(tooth_number))
    input_file = open(file_loc, 'r')
    landmarks = []
    temp_x = 0
    for index, line in enumerate(input_file):
        if index % 2 == 0:
            temp_x = float(line)
        else:
            landmarks.append((temp_x, float(line)))
    return landmarks


def import_landmarks_for_image(image):
    original_landmarks = []
    mirrored_landmarks = []
    for tooth_number in range(1, TEETH_AMOUNT + 1):
        original_landmarks.append(import_landmarks(image, tooth_number))
        mirrored_landmarks.append(import_landmarks(image, tooth_number, mirrored=True))
    return original_landmarks, mirrored_landmarks


def import_radiograph(image):
    directory = WORKING_DIR + RADIO_DIR
    if image > 14:
        directory += RADIO_EXTRA_DIR
    image_str = str(image)
    if len(image_str) < 2:
        image_str = "0" + image_str
    file_loc = directory + RADIO_FILE_TEMPLATE
    file_loc = file_loc.replace("{image}", image_str)
    file_image = cv2.imread(file_loc)
    return file_image


def import_single_segmentation(image, tooth_number):
    directory = WORKING_DIR + SEGMENT_DIR
    image_str = str(image)
    if len(image_str) < 2:
        image_str = "0" + image_str
    file_loc = directory + SEGMENT_FILE_TEMPLATE
    file_loc = file_loc.replace("{image}", image_str)
    file_loc = file_loc.replace("{tooth_number}", str(tooth_number))
    file_image = cv2.imread(file_loc)
    return file_image


def import_segmentations(image):
    segmentations = []
    for tooth_number in range(1, TEETH_AMOUNT + 1):
        segmentations.append(import_single_segmentation(image, tooth_number))
    return segmentations


def import_image(image):
    if image < 1 or image > TOTAL_RADIO_AMOUNT:
        raise ValueError("Trying to import image with number " + str(image) +
                         ", needs to be between 1 and " + str(TOTAL_RADIO_AMOUNT) + ".")

    radiograph = import_radiograph(image)
    if image <= PROCESSED_RADIO_AMOUNT:
        segmentations = import_segmentations(image)
        original_landmarks, mirrored_landmarks = import_landmarks_for_image(image)
        result = {"radiograph": radiograph,
                  "segmentations": segmentations,
                  "original_landmarks": original_landmarks,
                  "mirrored_landmarks": mirrored_landmarks,
                  "full_data": True}
    else:
        result = {"radiograph": radiograph,
                  "segmentations": None,
                  "original_landmarks": None,
                  "mirrored_landmarks": None,
                  "full_data": False}
    return result


def import_all_landmarks():
    imported_landmarks = []
    for image in range(1, PROCESSED_RADIO_AMOUNT + 1):
        imported_landmarks.append(import_landmarks_for_image(image))
    return imported_landmarks
