
# General configuration variables
PROCESSED_RADIO_AMOUNT = 14
TOTAL_RADIO_AMOUNT = 30
TEETH_AMOUNT = 8

# Make use of the mirrored dataset?
USE_MIRRORED = True

# Importing variables
RADIO_DIR = "/Radiographs/"
RADIO_EXTRA_DIR = "extra/"
RADIO_FILE_TEMPLATE = "{image}.tif"

LANDMARK_DIR = "/Landmarks/"
LANDMARK_DEFAULT_DIR = "original/"
LANDMARK_MIRRORED_DIR = "mirrored/"
LANDMARK_FILE_TEMPLATE = "landmarks{image}-{tooth_number}.txt"

SEGMENT_DIR = "/Segmentations/"
SEGMENT_FILE_TEMPLATE = "{image}-{tooth_number}.png"
