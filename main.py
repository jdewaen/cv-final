import cv2
import math
import numpy as np
import input
import landmarks
import radiograph
from output import *
import pca
from config import *
cx = 0
cy = 0


def onclick(event, x, y, flags, param):
    global cx, cy
    if event == cv2.EVENT_LBUTTONUP:
        cx = x
        cy = y


def update(img, sobelc, lm, window):
    for point in lm:
        cv2.circle(img, (int(point[0]), int(point[1])), 1, (0, 0, 255), 1)
        cv2.circle(sobelc, (int(point[0]), int(point[1])), 1, (0, 0, 255), 1)
    cv2.imshow(window, np.hstack([img, sobelc]))

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
        scale = 45
    if tn == 1 or tn == 2:
        scale = 50
    if tn == 4 or tn == 7:
        scale = 42.5
    if tn == 5 or tn == 6:
        scale = 40
    result = landmarks.restore_landmarks(result, translation, scale, angle) # TODO: dont hardcode scale
    print("fixed: " + str(fixed) + "\n")
    return result


def calculate_direction(vects, prev, matrix_center, d):
    perp_dir = -1  # INPUT IS CCW, points inwards if == 1
    result = np.zeros((2*d+1, 2*d+1))
    x_coords = np.ones(2*d+1) * matrix_center[0]
    x_coords += np.array(range(-d, d+1))
    y_coords = np.ones(2*d+1) * matrix_center[1]
    y_coords += np.array(range(-d, d+1))
    # if index == 20:
    #     bla = np.zeros((vects.shape[0], vects.shape[1], 3))
    #     test = np.arctan2(vects[:, :, 1], vects[:, :, 0])
    #     test[:, :] /= np.pi * 2
    #     test[:, :] += 0.5
    #     for y, row in enumerate(test):
    #         for x, val in enumerate(row):
    #             res = float(sobel[y, x]) / 255
    #             bla[y, x] = hsv_to_bgr(val, 1, res)
    #     bla /= 255
    #     display_single_image(bla)
    for y_i, y in enumerate(y_coords):
        for x_i, x in enumerate(x_coords):
            cur = np.array([x, y])
            diff = cur - prev
            diff /= np.linalg.norm(diff)
            perp = np.array([-perp_dir * diff[1], perp_dir * diff[0]])
            cur_vect = vects[y_i, x_i]
            vect_norm = np.linalg.norm(cur_vect)
            if vect_norm != 0:
                cur_vect /= vect_norm
            dot_prod = perp.dot(cur_vect)
            result[y_i, x_i] = dot_prod


    return result


def fit(sobel, points, pca_data, stds, delta, vects):
    d = 5
    alpha = 0.2  # tension 0.2
    beta = 0.3  # stiffness 0.3
    gamma = 1.5  # sensitivity
    # delta = 0.3 # shape
    new_points = np.zeros(points.shape)
    cost_map = []
    last_point = points[-1]
    avg_dist = average_dist(points)
    cost = np.zeros(3)
    for index, point in enumerate(points):
        x, y = point
        direction_cost = calculate_direction(vects[y-d:y+d+1, x-d:x+d+1], last_point, point, d)
        ext_cost = - gamma * np.multiply(sobel[y-d:y+d+1, x-d:x+d+1], direction_cost)
        elastic_cost = alpha * np.square(matrix_point_dist(last_point, point, d) - avg_dist)
        if index == len(points) - 1:
            next_point = points[0]
        else:
            next_point = points[index + 1]
        curvature_cost = beta * curvature(last_point, next_point, point, d)

        cost_map.append(curvature_cost + elastic_cost + ext_cost)
        cost[0] += curvature_cost[d, d]
        cost[1] += elastic_cost[d, d]
        cost[2] += ext_cost[d, d]
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
    print("curv: " + str(cost[0]) + "")
    print("elas: " + str(cost[1]) + "")
    print("exte: " + str(cost[2]) + "")
    new_points = (1 - delta)*new_points + delta*fix_shape(new_points, pca_data, stds)

    return new_points

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

# radiograph.process_radiographs()
tn = 0
while tn < 8:
    imn = 1
    full = cv2.imread("processed/homo_only/proc-" + str(imn) + ".png")
    topo = cv2.imread("output/topology-" + str(imn) + ".png")
    sobel, vects = radiograph.sobel_only(full[:, :, 0])
    sobelcopy = np.copy(topo)
    show = np.copy(full)
    cv2.namedWindow("test")
    cv2.imshow("test", show)
    cv2.setMouseCallback("test", onclick)
    cv2.waitKey()
    points = pca.vary_pca_parameter(0, stds[tn], pca_data[tn])[1]
    # eigval, eigvect, mu = pca_data[0]
    # default = np.zeros(len(eigvect[0]))
    # points = pca.reconstruct(eigvect, default, mu)
    # cx, cy = (189, 436) # 1
    # cx, cy = (295, 385) # 2
    # cx, cy = (211, 510) # 4
    points[:] *= 45
    if tn < 4:
        y_start = max(points[:, 1])
    else:
        y_start = min(points[:, 1])
    points[:] += (cx, cy - y_start)
    print(cx, cy)
    update(show, sobelcopy, points, "test")
    cv2.waitKey()
    delta = 0
    ind = 0
    # while True:
    while ind < 25:
        ind += 1
        points = fit(sobel, points, pca_data[tn], stds[tn], delta, vects)
        # show = np.copy(full)
        # sobelcopy = np.copy(topo)
        # update(show, sobelcopy, points, "test")
        # cv2.waitKey()
        if delta < 0.6:
            delta += 0.03
        # if ind%15 == 0:
        #     delta = 0.8
        # else:
        #     delta = 0

    show = np.copy(full)
    sobelcopy = np.copy(topo)
    update(show, sobelcopy, points, "test")
    cv2.waitKey()
    tn += 1
