import cv2
import math
import numpy as np
import input
import landmarks
import radiograph
from output import *
import pca
from config import *

# import raw landmarks from file
raw_landmarks = input.import_all_landmarks()

# show landmarks on input files
# display_input("input")

# normalize landmarks
landmarks = landmarks.process_landmarks(raw_landmarks)

# show normalized landmarks
# display_all_overlaid_landmarks(landmarks, "landmarks")


# generate PCA and standard deviations
pca_data = pca.pca_all(landmarks)
stds = pca.calculate_all_std(landmarks, pca_data)

# show modes of variation
# for tn in range(0, TEETH_AMOUNT):
#     display_side_by_side_landmarks(pca.vary_pca_parameter(0, stds[tn], pca_data[tn]), "modelvar-"+str(tn) + "-0")
#     display_side_by_side_landmarks(pca.vary_pca_parameter(1, stds[tn], pca_data[tn]), "modelvar-"+str(tn) + "-1")

radiograph.process_radiographs()
