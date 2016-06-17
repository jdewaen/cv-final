import os


# General configuration variables
PROCESSED_RADIO_AMOUNT = 14
TOTAL_RADIO_AMOUNT = 30
TEETH_AMOUNT = 8

# Make use of the mirrored dataset?
USE_MIRRORED = True

# Importing variables
WORKING_DIR = os.path.dirname(__file__)
RADIO_DIR = "/Radiographs/"
RADIO_EXTRA_DIR = "extra/"
RADIO_FILE_TEMPLATE = "{image}.tif"

LANDMARK_DIR = "/Landmarks/"
LANDMARK_DEFAULT_DIR = "original/"
LANDMARK_MIRRORED_DIR = "mirrored/"
LANDMARK_FILE_TEMPLATE = "landmarks{image}-{tooth_number}.txt"

SEGMENT_DIR = "/Segmentations/"
SEGMENT_FILE_TEMPLATE = "{image}-{tooth_number}.png"

# Image output directory
OUTPUT_DIR = "/output/"
ALWAYS_SHOW_IMAGES = False

# Radiography processing variables
REPROCESS_IMAGES = False
IMAGE_BIT_DEPTH = 256

X_CROP_RATIO = 0.25
Y_CROP_RATIO = 0.55
Y_CROP_OFFSET = 0.07

HOMOMORPHIC_GAUSS_SIGMA = 10
HOMOMORPHIC_ALPHA = 0.9
HOMOMORPHIC_BETA = 1.5

DENOISE_H = 6
DENOISE_TEMPLATE_WINDOW = 7
DENOISE_SEARCH_WINDOW = 21

PROCESSED_DIR = WORKING_DIR + "/processed/"
CROPPED_DIR = PROCESSED_DIR + "cropped/"
HOMOMORPHIC_DIR = PROCESSED_DIR + "homomorphic/"
SOBEL_DIR = PROCESSED_DIR + "sobel/"
SOBEL_DENOISED_DIR = PROCESSED_DIR + "sobel_denoised/"
GRADIENT_DIR = PROCESSED_DIR + "gradient/"