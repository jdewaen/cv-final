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


def update(homo_image, gradients, all_points, window, scores=None, details=None):
    homo_copy = np.copy(homo_image)
    gradients_copy = np.copy(gradients)
    hue = 0
    for index, point_set in enumerate(all_points):
        for point in point_set:
            cv2.circle(homo_copy, (int(point[0]), int(point[1])), 1, hsv_to_bgr(hue, 1, 1), 2)
            cv2.circle(gradients_copy, (int(point[0]), int(point[1])), 1, hsv_to_bgr(hue, 1, 1), 2)
        if scores is not None:
            cv2.putText(gradients_copy, str(int(0.1*scores[index])), (0, 25 + index*20), cv2.FONT_HERSHEY_PLAIN, 1, hsv_to_bgr(hue, 1, 1))
            cv2.putText(gradients_copy, str(int(0.1*details[index, 0])), (100, 25 + index*20), cv2.FONT_HERSHEY_PLAIN, 1, hsv_to_bgr(hue, 1, 1))
            cv2.putText(gradients_copy, str(int(0.1*details[index, 1])), (150, 25 + index*20), cv2.FONT_HERSHEY_PLAIN, 1, hsv_to_bgr(hue, 1, 1))
            cv2.putText(gradients_copy, str(int(0.1*details[index, 2])), (200, 25 + index*20), cv2.FONT_HERSHEY_PLAIN, 1, hsv_to_bgr(hue, 1, 1))
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


def calculate_direction(vects, prev, next, matrix_center, d, nbh):
    perp_dir = -1  # INPUT IS CCW, points inwards if == 1

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

    # sobel vectors
    norm = np.linalg.norm(vects, axis=2)
    zeros = np.uint8(norm == np.zeros((2 * d + 1, 2 * d + 1)))  # all places where the norm is zero are one here
    norm += zeros  # prevent division by zero
    vects[:, :, 0] = vects[:, :, 0] / norm
    vects[:, :, 1] = vects[:, :, 1] / norm

    result = (np.multiply(perp_prev[:, :, 0], vects[:, :, 0]) + np.multiply(perp_prev[:, :, 1], vects[:, :, 1]))
    # result += np.ones(result.shape)
    result = np.multiply(nbh, result)
    # result = np.multiply(nbh, result)
    # test1 = radiograph.generate_gradient_from_vectors(nbh, perp_prev)
    # test2 = radiograph.generate_gradient_from_vectors(nbh, vects)
    # resulttest = cv2.cvtColor(np.uint8(result), cv2.COLOR_GRAY2BGR)
    # display_single_image(cv2.resize(np.hstack([resulttest, test1, test2]), (0, 0),fx=10,fy=10))
    return result


def normalize_map(raw):
    norm = np.linalg.norm(raw)
    zeros = np.uint8(norm == np.zeros(raw.shape))  # all places where the norm is zero are one here
    norm += zeros  # prevent division by zero
    raw[:, :] = raw[:, :] / norm
    return raw

def fit(sobel, points, pca_data, stds, delta, vects, nbh):
    d = 7
    alpha = .2  # tension 0.2
    beta = .3  # stiffness 0.3
    gamma = 1.0  # sensitivity

    new_points = np.zeros(points.shape)
    cost_map = []
    last_point = points[-1]
    avg_dist = average_dist(points)
    cost = np.zeros(3)
    total_cost = 0
    for index, point in enumerate(points):
        x, y = point
        if index == len(points) - 1:
            next_point = points[0]
        else:
            next_point = points[index + 1]
        # start = time.clock()
        ext_cost = calculate_direction(vects[y-d:y+d+1, x-d:x+d+1], last_point, next_point, point, d, sobel[y-d:y+d+1, x-d:x+d+1])
        # print("dir: " + str(time.clock() - start))
        # start = time.clock()
        # ext_cost = - gamma * sobel[y-d:y+d+1, x-d:x+d+1]
        # ext_cost = - gamma * np.multiply(sobel[y-d:y+d+1, x-d:x+d+1], direction_cost)
        # print("ext: " + str(time.clock() - start))
        # start = time.clock()
        elastic_cost = np.square(matrix_point_dist(last_point, point, d) - avg_dist)
        # print("ela: " + str(time.clock() - start))
        # start = time.clock()
        curvature_cost = curvature(last_point, next_point, point, d)
        # print("cur: " + str(time.clock() - start))
        # start = time.clock()
        cost_map.append(beta * curvature_cost + alpha * elastic_cost - gamma * ext_cost)
        # display_single_image(cv2.resize(np.hstack([ext_cost, elastic_cost, curvature_cost, cost_map[index]]), (0, 0), fx=10, fy=10))

        # cost[3] += direction_cost[d, d]
        # select point
        tl = point - [d, d]
        min_val = None
        min_point = None

        for y, row in enumerate(cost_map[index]):
            for x, value in enumerate(row):
                if min_val is None or value < min_val:
                    min_val = value
                    min_point = [x, y]
        new_points[index] = np.ones(2)*tl + min_point
        last_point = np.ones(2)*tl + min_point
        # total_cost += ext_cost[min_point[1], min_point[0]]
        cost[0] += beta * curvature_cost[y, x]
        cost[1] += alpha * elastic_cost[y, x]
        cost[2] -= gamma * ext_cost[y, x]
        total_cost += min_val
        # print("oth: " + str(time.clock() - start))
    # print("curv: " + str(cost[0]) + "")
    # print("elas: " + str(cost[1]) + "")
    # print("exte: " + str(cost[2]) + "")
    # print("cost: " + str(total_cost) + "")
    new_points = (1 - delta)*new_points + delta*fix_shape(new_points, pca_data, stds)

    return new_points, total_cost, cost

# calculate_cost()


def main():
    global tn

    print("Building Active Shape Models...")
    # import raw landmarks from file
    raw_landmarks = input.import_all_landmarks()

    # show landmarks on input files
    # display_input("input")

    # normalize landmarks
    lms, scale_stats, angle_stats, _ = landmarks.process_landmarks(raw_landmarks)
    # show normalized landmarks
    # display_all_overlaid_landmarks(landmarks, "landmarks")


    # generate PCA and standard deviations
    pca_data = pca.pca_all(lms)
    stds = pca.calculate_all_std(lms, pca_data)

    # show modes of variation
    # for tn in range(0, TEETH_AMOUNT):
    #     display_side_by_side_landmarks(pca.vary_pca_parameter(0, stds[tn], pca_data[tn]), "modelvar-"+str(tn) + "-0")
    #     display_side_by_side_landmarks(pca.vary_pca_parameter(1, stds[tn], pca_data[tn]), "modelvar-"+str(tn) + "-1")



    print("Processing radiographs...")
    crop_offsets, sobel_vectors = radiograph.process_radiographs()
    print("\n")
    for imn in range(1, TOTAL_RADIO_AMOUNT):
        print("Fitting for image " + str(imn) + ":")
        print("Importing data...")
        homo_image = cv2.imread(HOMOMORPHIC_DIR + str(imn) + ".png")
        gradients = cv2.imread(GRADIENT_DIR + str(imn) + ".png")
        sobel = cv2.imread(SOBEL_DENOISED_DIR + str(imn) + ".png", 0)
        normalized = cv2.imread(NORMALIZED_DIR + str(imn) + ".png", 0)

        print("Estimating starting positions...")
        starting_positions = radiograph.detect_mouth(imn, homo_image, gradients, normalized)

        single = False
        pca_variations = 3
        rotation_variations = 3
        position_variations = 3
        total_variations = pca_variations * rotation_variations * position_variations
        points_result = np.zeros((TEETH_AMOUNT, POINTS_AMOUNT, 2))

        update(homo_image, gradients, [starting_positions], "test")
        cv2.waitKey()
        cv2.destroyAllWindows()

        print("Initialized!")
        # Tooth loop
        tn = 0
        while tn < 8:
            print("Fitting tooth " + str(tn+1) + "...")
            cx = starting_positions[tn][0]
            cy = starting_positions[tn][1]


            # cv2.namedWindow("test")
            # cv2.imshow("test", homo_image)
            # cv2.setMouseCallback("test", onclick)
            # cv2.waitKey()
            points_raw = pca.vary_pca_parameter(0, stds[tn], pca_data[tn])
            points = np.zeros((total_variations, len(points_raw[0]), len(points_raw[0][0])))

            if not single:
                for index in range(0, total_variations/pca_variations):
                    points[pca_variations * index] = points_raw[index % pca_variations]
                    points[pca_variations * index + 1] = points_raw[index % pca_variations]
                    points[pca_variations * index + 2] = points_raw[index % pca_variations]
            else:
                points[0] = points_raw[1]

            points = np.multiply(points, get_approx_scale(tn))
            for point_set in points:
                if tn < 4:
                    offset_y_index = np.argmax(point_set[:, 1])
                else:
                    offset_y_index = np.argmin(point_set[:, 1])
                point_set -= point_set[offset_y_index]

            if not single:
                for index, point_set in enumerate(points):
                    mult = (index % 3) - 1
                    points[index] = landmarks.rotate_landmarks(point_set, mult*1.5*angle_stats[tn][1])
            for index in range(0, total_variations):
                mult = ((index / (total_variations / position_variations)) % position_variations) - 1
                points[index] += (10*mult + cx, cy)
            # points[:] += (cx, cy)
            # update(homo_image, gradients, points, "test")
            # cv2.waitKey()
            delta = 0
            ind = 0
            # while True:
            costs = np.zeros(total_variations)
            costs_details = np.zeros((total_variations, 3))
            while ind < 8:
                ind += 1
                if not single:
                    for index, point_set in enumerate(points):
                        points[index], costs[index], costs_details[index] = fit(sobel, point_set, pca_data[tn], stds[tn], delta, sobel_vectors[imn-1], gradients)
                else:
                    points[0], costs[0], costs_details[0] = fit(sobel, points[0], pca_data[tn], stds[tn], delta,
                                                      sobel_vectors[imn - 1], gradients)
                # print ("\n")
                # update(homo_image, gradients, points, "test", costs, costs_details)
                # cv2.waitKey()
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
            points_result[tn] = points[min_cost_i]
            # print(min_cost_i)
            # print (min_cost)


            if not single:
                for index in range(0, len(costs)):
                    if index != min_cost_i:
                        points[index,:,:] = 0
            # update(homo_image, gradients, points, "test", costs, costs_details)
            # key = cv2.waitKey()
            # if key == ord('r'):
            #     continue
            tn += 1
        print("Done!\n\n")
        update(homo_image, gradients, points_result, "test")
        cv2.waitKey()
main()
