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
from radiograph import sobel

cx = 0
cy = 0
tn = 0

def onclick(event, x, y, flags, param):
    global cx, cy
    if event == cv2.EVENT_LBUTTONUP:
        cx = x
        cy = y


def update(homo_image, gradients, all_points, window):
    homo_copy = np.copy(homo_image)
    gradients_copy = np.copy(gradients)
    hue = 0
    for index, point_set in enumerate(all_points):
        for point in point_set:
            cv2.circle(homo_copy, (int(point[0]), int(point[1])), 1, hsv_to_bgr(hue, 1, 1), 2)
            cv2.circle(gradients_copy, (int(point[0]), int(point[1])), 1, hsv_to_bgr(hue, 1, 1), 2)
        hue += 1.0 / (len(all_points) + 1)
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

def get_approx_scale(tn):
    scale = 90
    if tn == 0 or tn == 3:
        scale = 90
    if tn == 1 or tn == 2:
        scale = 100
    if tn == 4 or tn == 7:
        scale = 85
    if tn == 5 or tn == 6:
        scale = 80
    return scale

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
    scale = get_approx_scale(tn)
    result = landmarks.restore_landmarks(result, translation, scale, angle) # TODO: dont hardcode scale
    # print("fixed: " + str(fixed) + "\n")
    return result


def calculate_direction(vects, prev, next, matrix_center, d, nbh, point, bla):
    perp_dir = -1  # INPUT IS CCW, points inwards if == 1
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


    # sobel vectors
    norm = np.linalg.norm(vects, axis=2)
    zeros = np.uint8(norm == np.zeros((2 * d + 1, 2 * d + 1)))  # all places where the norm is zero are one here
    norm += zeros  # prevent division by zero
    vects[:, :, 0] = vects[:, :, 0] / norm
    vects[:, :, 1] = vects[:, :, 1] / norm

    result = (np.multiply(perp_prev[:, :, 0], vects[:, :, 0]) + np.multiply(perp_prev[:, :, 1], vects[:, :, 1]))
    # result = np.multiply(result)
    # test1 = radiograph.generate_gradient_from_vectors(nbh, perp_prev)
    # test2 = radiograph.generate_gradient_from_vectors(nbh, vects)
    # sobel = cv2.cvtColor(nbh,cv2.COLOR_GRAY2BGR)
    # resulttest = cv2.cvtColor(np.uint8(np.multiply(nbh, result + np.ones(result.shape))), cv2.COLOR_GRAY2BGR)
    # display_single_image(cv2.resize(np.hstack([resulttest, test1, test2]), (0, 0),fx=10,fy=10))
    return result


def fit(sobel, points, pca_data, stds, delta, vects, nbh):
    d = 10
    alpha = 0.3  # tension 0.2
    beta = 0.5  # stiffness 0.3
    gamma = 3  # sensitivity
    epsilon = 400  # pick right line
    # delta = 1 # shape
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
        direction_cost = calculate_direction(vects[y-d:y+d+1, x-d:x+d+1], last_point, next_point, point, d, sobel[y-d:y+d+1, x-d:x+d+1], index, nbh[y-d:y+d+1, x-d:x+d+1])
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
        # cost[0] += curvature_cost[d, d]
        # cost[1] += elastic_cost[d, d]
        # cost[2] += ext_cost[d, d]
        # cost[3] += direction_cost[d, d]
        # select point
        tl = point - [d, d]
        min_val = None
        min_point = None
        cost = 0
        for y, row in enumerate(cost_map[index]):
            for x, value in enumerate(row):
                if min_val is None or value < min_val:
                    min_val = value
                    min_point = tl + [x, y]
        new_points[index] = min_point
        last_point = min_point
        cost += min_val
        # print("oth: " + str(time.clock() - start))
    # print("curv: " + str(cost[0]) + "")
    # print("elas: " + str(cost[1]) + "")
    # print("exte: " + str(cost[2]) + "")
    # print("dirc: " + str(cost[3]) + "\n")
    print("cost: " + str(cost) + "\n")
    new_points = (1 - delta)*new_points + delta*fix_shape(new_points, pca_data, stds)

    return new_points, cost

# calculate_cost()


def main():
    global tn
    # import raw landmarks from file
    raw_landmarks = input.import_all_landmarks()

    # show landmarks on input files
    # display_input("input")

    # normalize landmarks
    lms, scale_stats, angle_stats = landmarks.process_landmarks(raw_landmarks)

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
    tn = 0
    imn = 1
    while tn < 8:
        homo_image = cv2.imread(HOMOMORPHIC_DIR + str(imn) + ".png")
        gradients = cv2.imread(GRADIENT_DIR + str(imn) + ".png")
        sobel = cv2.imread(SOBEL_DENOISED_DIR + str(imn) + ".png", 0)
        cv2.namedWindow("test")
        cv2.imshow("test", homo_image)
        cv2.setMouseCallback("test", onclick)
        cv2.waitKey()
        points_raw = pca.vary_pca_parameter(0, stds[tn], pca_data[tn])
        points = np.zeros((9, len(points_raw[0]), len(points_raw[0][0])))
        points[0] = landmarks.rotate_landmarks(points_raw[0], angle_stats[tn][0] - 3*angle_stats[tn][1])
        points[1] = landmarks.rotate_landmarks(points_raw[0], angle_stats[tn][0])
        points[2] = landmarks.rotate_landmarks(points_raw[0], angle_stats[tn][0] + 3*angle_stats[tn][1])
        points[3] = landmarks.rotate_landmarks(points_raw[1], angle_stats[tn][0] - 3*angle_stats[tn][1])
        points[4] = landmarks.rotate_landmarks(points_raw[1], angle_stats[tn][0])
        points[5] = landmarks.rotate_landmarks(points_raw[1], angle_stats[tn][0] + 3*angle_stats[tn][1])
        points[6] = landmarks.rotate_landmarks(points_raw[2], angle_stats[tn][0] - 3*angle_stats[tn][1])
        points[7] = landmarks.rotate_landmarks(points_raw[2], angle_stats[tn][0])
        points[8] = landmarks.rotate_landmarks(points_raw[2], angle_stats[tn][0] + 3*angle_stats[tn][1])
        # eigval, eigvect, mu = pca_data[0]
        # default = np.zeros(len(eigvect[0]))
        # points = pca.reconstruct(eigvect, default, mu)
        # cx, cy = (162, 530) # 1
        # cx, cy = (295, 385) # 2
        # cx, cy = (211, 510) # 4
        points = np.multiply(points, get_approx_scale(tn))
        # y_start = np.zeros(3)
        for point_set in points:
            if tn < 4:
                point_set += (cx, cy - max(point_set[:, 1]))
            else:
                point_set += (cx, cy - min(point_set[:, 1]))
        #
        # if tn < 4:
        #     y_start[0] = max(points[0, :, 1])
        #     y_start[1] = max(points[1, :, 1])
        #     y_start[2] = max(points[2, :, 1])
        # else:
        #     y_start[0] = min(points[0, :, 1])
        #     y_start[1] = min(points[1, :, 1])
        #     y_start[2] = min(points[2, :, 1])
        # points[0] += (cx, cy - y_start[0])
        # points[1] += (cx, cy - y_start[1])
        # points[2] += (cx, cy - y_start[2])
        print(cx, cy)
        update(homo_image, gradients, points, "test")
        cv2.waitKey()
        delta = 0
        ind = 0
        # while True:
        costs = np.zeros(9)
        while ind < 10:
            ind += 1
            for index, point_set in enumerate(points):
                points[index], costs[index] = fit(sobel, point_set, pca_data[tn], stds[tn], delta, sobel_vectors[imn-1], gradients)
            # points[1], costs[1] = fit(sobel, points[1], pca_data[tn], stds[tn], delta, sobel_vectors[imn-1], gradients)
            # points[2], costs[2] = fit(sobel, points[2], pca_data[tn], stds[tn], delta, sobel_vectors[imn-1], gradients)
            print ("\n")
            update(homo_image, gradients, points, "test")
            cv2.waitKey()
            if delta < 0.6:
                delta += 0.1
            # if ind%15 == 0:
            #     delta = 1
            # else:
            #     delta = 0
        min_cost = None
        min_cost_i = 0
        for index, cost in enumerate(costs):
            if min_cost is None or cost < min_cost:
                min_cost = cost
                min_cost_i = index

        print(min_cost_i)
        print (min_cost)

        for index in range(0, len(costs)):
            if index != min_cost_i:
                points[index,:,:] = 0
        update(homo_image, gradients, points, "test")
        key = cv2.waitKey()
        if key == ord('r'):
            continue
        tn += 1

main()
