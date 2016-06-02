import cv2
import math
import numpy as np
import input
import landmarks
import radiograph
from output import *
import pca
import time
from config import *
cx = 0
cy = 0
tn = 0

def onclick(event, x, y, flags, param):
    global cx, cy
    if event == cv2.EVENT_LBUTTONUP:
        cx = x
        cy = y


def update(homo_image, gradients, lm, window):
    homo_copy = np.copy(homo_image)
    gradients_copy = np.copy(gradients)
    for point in lm:
        cv2.circle(homo_copy, (int(point[0]), int(point[1])), 1, (0, 0, 255), 2)
        cv2.circle(gradients_copy, (int(point[0]), int(point[1])), 1, (0, 0, 255), 2)
    cv2.imshow(window, np.hstack([homo_copy, gradients_copy]))

def point_dist(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def matrix_point_dist(point, matrix_center, d):
    x_coords = np.ones((2*d+1, 2*d+1)) * matrix_center[0]
    x_coords[:] += np.array(range(-d, d+1))
    y_coords = np.ones((2*d+1, 2*d+1)) * matrix_center[1]
    y_coords[:] += np.array(range(-d, d+1))
    y_coords = np.transpose(y_coords)
    ref_matrix_x = np.ones((2*d+1, 2*d+1)) * point[0]
    ref_matrix_y = np.ones((2*d+1, 2*d+1)) * point[1]
    diff_x = x_coords - ref_matrix_x
    diff_y = y_coords - ref_matrix_y
    result = np.sqrt(np.square(diff_x) + np.square(diff_y))
    return result


def average_dist(points):
    last_point = points[-1]
    dist = 0.0
    for point in points:
        dist += point_dist(point,last_point)
        last_point = point
    dist /= len(points)
    return dist


def curvature(last_point, next_point, matrix_center, d):
    x_coords = np.ones((2*d+1, 2*d+1)) * matrix_center[0]
    x_coords[:] += np.array(range(-d, d+1))
    y_coords = np.ones((2*d+1, 2*d+1)) * matrix_center[1]
    y_coords[:] += np.array(range(-d, d+1))
    y_coords = np.transpose(y_coords)
    next_point_x = np.ones((2*d+1, 2*d+1)) * next_point[0]
    next_point_y = np.ones((2*d+1, 2*d+1)) * next_point[1]
    last_point_x = np.ones((2*d+1, 2*d+1)) * last_point[0]
    last_point_y = np.ones((2*d+1, 2*d+1)) * last_point[1]
    curv_x = next_point_x - 2 * x_coords + last_point_x
    curv_y = next_point_y - 2 * y_coords + last_point_y
    result = np.square(curv_x) + np.square(curv_y)
    return result


# def calculate_shape_cost2(points, index, pca_data, matrix_center, d):
#     x_coords = np.ones(2*d+1) * matrix_center[0]
#     x_coords += np.array(range(-d, d+1))
#     y_coords = np.ones(2*d+1) * matrix_center[1]
#     y_coords += np.array(range(-d, d+1))
#     result = np.zeros((2*d+1, 2*d+1))
#     _, eigv, mu = pca_data
#     for y_i, y in enumerate(y_coords):
#         for x_i, x in enumerate(x_coords):
#             points[index] = np.array([x, y])
#             vect = pca.project(eigv, points, mu)
#             result[y_i, x_i] = np.linalg.norm(vect)
#     return result

def fix_shape(points, pca_data, stds):
    _, eigv, mu = pca_data
    default = pca.reconstruct(eigv, np.zeros(len(eigv[0])), mu)
    norm_points, translation, scale, angle = landmarks.normalize_landmark_with_data(default, points)
    vect = pca.project(eigv, norm_points, mu)
    new_vec = vect.copy()
    fixed = 0
    for index, val in enumerate(vect):
        # if index == 0:
        #     continue
        if abs(val) > 3 * stds[index]:
            new_vec[index] = np.sign(val) * 3 * stds[index]
            fixed += 1

    result = pca.reconstruct(eigv, new_vec, mu)
    if tn == 0 or tn == 3:
        scale = 90
    if tn == 1 or tn == 2:
        scale = 100
    if tn == 4 or tn == 7:
        scale = 85
    if tn == 5 or tn == 6:
        scale = 80
    result = landmarks.restore_landmarks(result, translation, scale, angle) # TODO: dont hardcode scale
    # print("fixed: " + str(fixed) + "\n")
    return result


def calculate_direction(vects, prev, next, matrix_center, d, nbh, point):
    perp_dir = 1  # INPUT IS CCW, points inwards if == 1
    #
    # if point == 12:
    #     display_single_image(nbh)

    prev_all = np.zeros((2*d+1, 2*d+1, 2))
    prev_all[:, :, 0] = prev[0]
    prev_all[:, :, 1] = prev[1]

    next_all = np.zeros((2*d+1, 2*d+1, 2))
    next_all[:, :, 0] = next[0]
    next_all[:, :, 1] = next[1]

    coords = np.zeros((2*d+1, 2*d+1, 2))
    x_coords = np.ones((2*d+1, 2*d+1)) * matrix_center[0]
    x_coords[:] += np.array(range(-d, d+1))
    y_coords = np.ones((2*d+1, 2*d+1)) * matrix_center[1]
    y_coords[:] += np.array(range(-d, d+1))
    y_coords = np.transpose(y_coords)
    coords[:, :, 0] = x_coords
    coords[:, :, 1] = y_coords

    diff = coords - prev_all
    norm = np.linalg.norm(diff, axis=2)
    zeros = np.uint8(norm == np.zeros((2*d+1, 2*d+1)))  # all places where the norm is zero are one here
    norm += zeros  # prevent division by zero
    diff[:, :, 0] = diff[:, :, 0] / norm
    diff[:, :, 1] = diff[:, :, 1] / norm
    perp_prev = np.zeros((2*d+1, 2*d+1, 2))
    perp_prev[:, :, 0] = -perp_dir * diff[:, :, 1]
    perp_prev[:, :, 1] = perp_dir * diff[:, :, 0]

    diff = next_all - coords
    norm = np.linalg.norm(diff, axis=2)
    zeros = np.uint8(norm == np.zeros((2*d+1, 2*d+1)))  # all places where the norm is zero are one here
    norm += zeros  # prevent division by zero
    diff[:, :, 0] = diff[:, :, 0] / norm
    diff[:, :, 1] = diff[:, :, 1] / norm
    perp_next = np.zeros((2*d+1, 2*d+1, 2))
    perp_next[:, :, 0] = -perp_dir * diff[:, :, 1]
    perp_next[:, :, 1] = perp_dir * diff[:, :, 0]

    perp = (perp_prev + perp_next) / 2

    norm = np.linalg.norm(vects, axis=2)
    zeros = np.uint8(norm == np.zeros((2 * d + 1, 2 * d + 1)))  # all places where the norm is zero are one here
    norm += zeros  # prevent division by zero
    vects[:, :, 0] = vects[:, :, 0] / norm
    vects[:, :, 1] = vects[:, :, 1] / norm

    result = np.multiply(perp[:, :, 0], vects[:, :, 0]) + np.multiply(perp[:, :, 1], vects[:, :, 1])
    return result


def fit(sobel, points, pca_data, stds, delta, vects, nbh):
    d = 10
    alpha = 0.2  # tension 0.2
    beta = 0.3  # stiffness 0.3
    gamma = 5  # sensitivity
    epsilon = 400  # pick right line
    # delta = 0.3 # shape
    new_points = np.zeros(points.shape)
    cost_map = []
    last_point = points[-1]
    avg_dist = average_dist(points)
    cost = np.zeros(4)
    for index, point in enumerate(points):
        x, y = point
        if index == len(points) - 1:
            next_point = points[0]
        else:
            next_point = points[index + 1]
        # start = time.clock()
        direction_cost = calculate_direction(vects[y-d:y+d+1, x-d:x+d+1], last_point, next_point, point, d, nbh[y-d:y+d+1, x-d:x+d+1], index)
        # print("dir: " + str(time.clock() - start))
        # start = time.clock()
        # ext_cost = - gamma * sobel[y-d:y+d+1, x-d:x+d+1]
        ext_cost = - gamma * np.multiply(sobel[y-d:y+d+1, x-d:x+d+1], direction_cost)
        # print("ext: " + str(time.clock() - start))
        # start = time.clock()
        elastic_cost = alpha * np.square(matrix_point_dist(last_point, point, d) - avg_dist)
        # print("ela: " + str(time.clock() - start))
        # start = time.clock()
        curvature_cost = beta * curvature(last_point, next_point, point, d)
        # print("cur: " + str(time.clock() - start))
        # start = time.clock()

        cost_map.append(curvature_cost + elastic_cost + ext_cost)
        cost[0] += curvature_cost[d, d]
        cost[1] += elastic_cost[d, d]
        cost[2] += ext_cost[d, d]
        cost[3] += direction_cost[d, d]
        # select point
        tl = point - [d, d]
        min_val = None
        min_point = None
        for y, row in enumerate(cost_map[index]):
            for x, value in enumerate(row):
                if min_val is None or value < min_val:
                    min_val = value
                    min_point = tl + [x, y]
        new_points[index] = min_point
        last_point = min_point
        # print("oth: " + str(time.clock() - start))
    print("curv: " + str(cost[0]) + "")
    print("elas: " + str(cost[1]) + "")
    print("exte: " + str(cost[2]) + "")
    print("dirc: " + str(cost[3]) + "\n")
    new_points = (1 - delta)*new_points + delta*fix_shape(new_points, pca_data, stds)

    return new_points


def main():
    global tn
    # import raw landmarks from file
    raw_landmarks = input.import_all_landmarks()

    # show landmarks on input files
    # display_input("input")

    # normalize landmarks
    lms = landmarks.process_landmarks(raw_landmarks)

    # show normalized landmarks
    # display_all_overlaid_landmarks(landmarks, "landmarks")


    # generate PCA and standard deviations
    pca_data = pca.pca_all(lms)
    stds = pca.calculate_all_std(lms, pca_data)

    # show modes of variation
    # for tn in range(0, TEETH_AMOUNT):
    #     display_side_by_side_landmarks(pca.vary_pca_parameter(0, stds[tn], pca_data[tn]), "modelvar-"+str(tn) + "-0")
    #     display_side_by_side_landmarks(pca.vary_pca_parameter(1, stds[tn], pca_data[tn]), "modelvar-"+str(tn) + "-1")

    crop_offsets, sobel_vectors = radiograph.process_radiographs()
    tn = 1
    imn = 1
    while tn < 8:
        homo_image = cv2.imread(HOMOMORPHIC_DIR + str(imn) + ".png")
        gradients = cv2.imread(GRADIENT_DIR + str(imn) + ".png")
        sobel = cv2.imread(SOBEL_DENOISED_DIR + str(imn) + ".png", 0)
        cv2.namedWindow("test")
        cv2.imshow("test", homo_image)
        cv2.setMouseCallback("test", onclick)
        cv2.waitKey()
        points = pca.vary_pca_parameter(0, stds[tn], pca_data[tn])[1]
        # eigval, eigvect, mu = pca_data[0]
        # default = np.zeros(len(eigvect[0]))
        # points = pca.reconstruct(eigvect, default, mu)
        # cx, cy = (189, 436) # 1
        # cx, cy = (295, 385) # 2
        # cx, cy = (211, 510) # 4
        points[:] *= 90
        if tn < 4:
            y_start = max(points[:, 1])
        else:
            y_start = min(points[:, 1])
        points[:] += (cx, cy - y_start)
        print(cx, cy)
        update(homo_image, gradients, points, "test")
        cv2.waitKey()
        delta = 0
        ind = 0
        while True:
        # while ind < 20:
            ind += 1
            points = fit(sobel, points, pca_data[tn], stds[tn], delta, sobel_vectors[imn], gradients)
            update(homo_image, gradients, points, "test")
            cv2.waitKey()
            # if delta < 0.3:
            #     delta += 0.03
            # if ind%15 == 0:
            #     delta = 0.8
            # else:
            #     delta = 0

        update(homo_image, gradients, points, "test")
        key = cv2.waitKey()
        if key == ord('r'):
            continue
        tn += 1

main()
