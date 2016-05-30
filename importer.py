import cv2
import os

WORKING_DIR = os.path.dirname(__file__)

PROCESSED_RADIO_AMOUNT = 14
TOTAL_RADIO_AMOUNT = 30
TEETH_AMOUNT = 8

RADIO_DIR = "/Radiographs/"
RADIO_EXTRA_DIR = "extra/"
RADIO_FILE_TEMPLATE = "{image}.tif"

LANDMARK_DIR = "/Landmarks/"
LANDMARK_DEFAULT_DIR = "original/"
LANDMARK_MIRRORED_DIR = "mirrored/"
LANDMARK_FILE_TEMPLATE = "landmarks{image}-{tooth_number}.txt"

SEGMENT_DIR = "/Segmentations/"
SEGMENT_FILE_TEMPLATE = "{image}-{tooth_number}.png"


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
            temp_x = int(float(line))
        else:
            landmarks.append((temp_x, int(float(line))))
    return landmarks


def import_all_landmarks(image):
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
        original_landmarks, mirrored_landmarks = import_all_landmarks(image)
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