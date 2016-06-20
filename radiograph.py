import input
import output
import cv2
import scipy.signal
import scipy.fftpack
from config import *
import numpy as np
from matplotlib import pyplot as plt


def normalize_histogram(image, as_int=False):
    height = len(image)
    width = len(image[0])
    num_pixels = width * height
    hist = cv2.calcHist([image], [0], None, [IMAGE_BIT_DEPTH], [0, IMAGE_BIT_DEPTH])
    # pf = np.zeros(IMAGE_BIT_DEPTH, np.float_)
    cdf = np.zeros(IMAGE_BIT_DEPTH, np.float_)
    for index, count in enumerate(hist):
        prob = float(count) / num_pixels
        if index == 0:
            cdf[index] = prob
        else:
            cdf[index] = cdf[index - 1] + prob
    norm_image = cdf[image]
    if as_int:
        norm_image[:, :] *= (IMAGE_BIT_DEPTH - 1)
        norm_image = np.uint8(norm_image)
    return norm_image


def calculate_crop_data(image, x_ratio, y_ratio, y_offset):
    height = len(image)
    y_offset *= height
    width = len(image[0])
    x_crop = width * (1 - x_ratio) / 2
    y_crop = height * (1 - y_ratio) / 2
    top_left = (y_crop + y_offset, x_crop)
    bottom_right = (height + y_offset - y_crop, width - x_crop)
    return top_left, bottom_right


def crop(image, top_left, bottom_right):
    top_left = np.int32(top_left)
    bottom_right = np.int32(bottom_right)
    return image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

def sobel(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    result = np.sqrt(np.square(sobelx) + np.square(sobely))
    vects = np.zeros((img.shape[0], img.shape[1], 2))
    vects[:, :, 0] = sobelx
    vects[:, :, 1] = sobely
    result = np.uint8(result)
    return result, vects


def generate_gradient_from_vectors(sobel_image, sobel_vectors):
    hues = np.arctan2(sobel_vectors[:, :, 1], sobel_vectors[:, :, 0])
    hues[:, :] /= np.pi * 2
    hues[:, :] += 0.5
    # hues[:, :] *= IMAGE_BIT_DEPTH - 1
    result = np.zeros((sobel_vectors.shape[0], sobel_vectors.shape[1], 3))
    for y, row in enumerate(hues):
        for x, val in enumerate(row):
            res = float(sobel_image[y, x]) / (IMAGE_BIT_DEPTH-1)
            result[y, x] = output.hsv_to_bgr(val, 1, res)
    return np.uint8(result)

def homo_3(img):
    return

def denoise_image(img):
    result = cv2.fastNlMeansDenoising(img, None, h=6, templateWindowSize=7, searchWindowSize=21)
    return result

def homomorphic_filter(img):

    rows = img.shape[0]
    cols = img.shape[1]
    # Convert image to 0 to 1, then do log(1 + I)
    imgLog = np.log1p(np.array(img, dtype="float") / (IMAGE_BIT_DEPTH-1))

    # Create Gaussian mask of sigma = 10
    M = 2 * rows + 1
    N = 2 * cols + 1
    sigma = HOMOMORPHIC_GAUSS_SIGMA
    (X, Y) = np.meshgrid(np.linspace(0, N - 1, N), np.linspace(0, M - 1, M))
    centerX = np.ceil(N / 2)
    centerY = np.ceil(M / 2)
    gaussianNumerator = (X - centerX) ** 2 + (Y - centerY) ** 2

    # Low pass and high pass filters
    Hlow = np.exp(-gaussianNumerator / (2 * sigma * sigma))
    Hhigh = 1 - Hlow

    # Move origin of filters so that it's at the top left corner to
    # match with the input image
    HlowShift = scipy.fftpack.ifftshift(Hlow)
    HhighShift = scipy.fftpack.ifftshift(Hhigh)

    # Filter the image and crop
    If = scipy.fftpack.fft2(imgLog.copy(), (M, N))
    Ioutlow = scipy.real(scipy.fftpack.ifft2(If.copy() * HlowShift, (M, N)))
    Iouthigh = scipy.real(scipy.fftpack.ifft2(If.copy() * HhighShift, (M, N)))

    # Set scaling factors and add
    alpha = HOMOMORPHIC_ALPHA
    beta = HOMOMORPHIC_BETA
    Iout = alpha * Ioutlow[0:rows, 0:cols] + beta * Iouthigh[0:rows, 0:cols]

    # Anti-log then rescale to [0,1]
    Ihmf = np.expm1(Iout)
    Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))
    Ihmf2 = np.array((IMAGE_BIT_DEPTH-1) * Ihmf, dtype="uint8")
    return Ihmf2


def calculate_mouth_center(normalized):
    min_mouth_y = 250
    norm_invert = np.uint8(np.ones(normalized.shape) * 255 - normalized)
    _, thresh = cv2.threshold(norm_invert, 200, 255, cv2.THRESH_BINARY)
    per_line = np.mean(thresh, axis=1)
    per_line = cv2.GaussianBlur(per_line, (5, 5), 10)
    mouth = min_mouth_y + np.argmax(per_line[min_mouth_y:])

    means = np.uint8(per_line * np.ones((1, 100)))
    normalized = cv2.cvtColor(np.uint8(normalized), cv2.COLOR_GRAY2BGR)
    means = cv2.cvtColor(np.uint8(means), cv2.COLOR_GRAY2BGR)
    return mouth


def calculate_bottom_start(sobely, mouth_y, middle):
    y_min_offset = 20
    y_max_offset = 150
    x_offset = 50
    result_bottom = np.copy(sobely)
    result_bottom = result_bottom[:, :, 0]
    result_bottom = np.uint8(result_bottom)
    _, mask = cv2.threshold(result_bottom, 128, 255, cv2.THRESH_BINARY_INV)
    result_bottom = cv2.bitwise_and(result_bottom, mask)
    per_line_bottom = np.mean(result_bottom[mouth_y - y_min_offset:mouth_y + y_max_offset, middle - x_offset:middle + x_offset], axis=1)
    per_line_bottom = cv2.GaussianBlur(per_line_bottom, (5, 5), 10)
    _, per_line_bottom = cv2.threshold(np.uint8(per_line_bottom), 8, 255, cv2.THRESH_BINARY)
    bottom_start = 0
    for index, value in enumerate(per_line_bottom):
        if value == 255:
            bottom_start = index
            break
    cv2.line(result_bottom, (1, bottom_start), (len(result_bottom[0]), bottom_start), 255)
    bottom_start -= y_min_offset - mouth_y

    return bottom_start, result_bottom

def calculate_top_start(sobely, mouth_y, middle):
    y_min_offset = 70
    y_max_offset = 50
    x_offset = 50
    result_top = np.copy(sobely)
    result_top = result_top[:, :, 0]
    result_top = np.uint8(result_top * -1)
    _, maskinv = cv2.threshold(result_top, 128, 255, cv2.THRESH_BINARY_INV)
    result_top = cv2.bitwise_and(result_top, maskinv)
    per_line_top = np.mean(result_top[mouth_y - y_min_offset:mouth_y + y_max_offset, middle - x_offset:middle + x_offset], axis=1)
    per_line_top = cv2.GaussianBlur(per_line_top, (1, 1), 10)
    _, per_line_top = cv2.threshold(np.uint8(per_line_top), 9, 255, cv2.THRESH_BINARY)
    top_start = 0
    for index, value in reversed(list(enumerate(per_line_top))):
        if value == 255:
            top_start = index
            break
    # cv2.line(result_top, (1, top_start), (len(result_top[0]), top_start), 255)
    top_start -= y_min_offset - mouth_y
    return top_start, result_top


def detect_mouth(imn, homo_image, gradients, normalized):
    result = []

    mouth = calculate_mouth_center(normalized)
    # cv2.line(gradients, (1, mouth), (len(gradients[0]), mouth), output.hsv_to_bgr(0, 1, 1))

    middle = len(normalized[0]) / 2
    sobely = cv2.Sobel(homo_image, cv2.CV_64F, 0, 1, ksize=3)
    sobelx = cv2.Sobel(homo_image, cv2.CV_64F, 1, 0, ksize=3)

    # BOTTOM TEETH VERTICAL
    bottom_start, gradient_bottom = calculate_bottom_start(sobely, mouth, middle)
    # cv2.line(gradients, (1, bottom_start), (len(gradients[0]), bottom_start), output.hsv_to_bgr(0.33, 1, 1))

    # TOP TEETH VERTICAL
    top_start, gradient_top = calculate_top_start(sobely, mouth, middle)
    # cv2.line(gradients, (1, top_start), (len(gradients[0]), top_start), output.hsv_to_bgr(0.66, 1, 1))

    # TOP TEETH HORIZONTAL
    top_gradients = gradients[top_start-150:top_start - 50, middle - 300: middle + 300]
    top_hori = sobelx[top_start-150:top_start - 50, middle - 300: middle + 300]
    top_hori = top_hori[:, :, 0]



    top_hori_1 = np.uint8(top_hori)
    _, hori_mask_1 = cv2.threshold(top_hori_1, 80, 255, cv2.THRESH_BINARY_INV)
    top_hori_1 = cv2.bitwise_and(top_hori_1, hori_mask_1)
    per_line_hori_top_1 = np.mean(top_hori_1, axis=0)
    per_line_hori_top_1 = cv2.GaussianBlur(per_line_hori_top_1, (5, 5), 10)
    _, per_line_hori_top_1a = cv2.threshold(np.uint8(per_line_hori_top_1), np.max(per_line_hori_top_1)*0.35, 255, cv2.THRESH_BINARY)

    top_hori_2 = np.uint8(top_hori * -1)
    _, hori_mask_2 = cv2.threshold(top_hori_2, 80, 255, cv2.THRESH_BINARY_INV) # 110
    top_hori_2 = cv2.bitwise_and(top_hori_2, hori_mask_2)
    per_line_hori_top_2 = np.mean(top_hori_2, axis=0)
    per_line_hori_top_2 = cv2.GaussianBlur(per_line_hori_top_2, (5, 5), 10)
    _, per_line_hori_top_2a = cv2.threshold(np.uint8(per_line_hori_top_2), np.max(per_line_hori_top_2)*0.35, 255, cv2.THRESH_BINARY) # 0.4


    test_hori_top_1 = np.transpose(np.uint8(per_line_hori_top_1 * np.ones((1, 40))))
    test_hori_top_1a = np.transpose(np.uint8(per_line_hori_top_1a * np.ones((1, 40))))
    test_hori_top_2 = np.transpose(np.uint8(per_line_hori_top_2 * np.ones((1, 40))))
    test_hori_top_2a = np.transpose(np.uint8(per_line_hori_top_2a * np.ones((1, 40))))
    # output.display_single_image(np.vstack((top_hori_1, test_hori_top_1, test_hori_top_1a,
    #                                       top_hori_2, test_hori_top_2, test_hori_top_2a
    # )))

    top_middle_edge = 0
    top_first_left_edge = 0
    top_first_right_edge = 0
    top_sec_left_edge = 0
    top_sec_right_edge = 0
    top_start_middle = len(per_line_hori_top_1a)/2
    shifted = False
    while True:
        top_middle_edge = find_teeth_borders(per_line_hori_top_1a, per_line_hori_top_2a, top_start_middle, top_gradients)
        middle_left = (top_middle_edge < top_start_middle)
        top_first_left_edge = find_teeth_borders(per_line_hori_top_1a, per_line_hori_top_2a, top_middle_edge - 60, top_gradients, -1)
        top_first_right_edge = find_teeth_borders(per_line_hori_top_1a, per_line_hori_top_2a, top_middle_edge + 60, top_gradients, 1)
        top_first_left_width = abs(top_middle_edge - top_first_left_edge)
        top_first_right_width = abs(top_first_right_edge - top_middle_edge)
        top_avg_width = (top_first_left_width + top_first_right_width) / 2
        top_sec_left_edge = find_teeth_borders(per_line_hori_top_1a, per_line_hori_top_2a, top_first_left_edge - top_avg_width/2, top_gradients, -1)
        top_sec_right_edge = find_teeth_borders(per_line_hori_top_1a, per_line_hori_top_2a, top_first_right_edge + top_avg_width/2, top_gradients, 1)
        top_sec_left_width = top_first_left_edge - top_sec_left_edge
        top_sec_right_width = top_sec_right_edge - top_first_right_edge
        middle_ratio = (1.0 * top_first_left_width / top_first_right_width)
        if middle_ratio < 1:
            middle_ratio = 1.0/middle_ratio
        middle_ratio -= 1
        if top_sec_left_width == 0:
            left_ratio = 20
        else:
            left_ratio = (1.0 * top_first_left_width / top_sec_left_width)
        if left_ratio < 1:
            left_ratio = 1.0/left_ratio
        left_ratio -= 1

        if top_sec_right_width == 0:
            right_ratio = 20
        else:
            right_ratio = (1.0 * top_first_right_width / top_sec_right_width)
        if right_ratio < 1:
            right_ratio = 1.0/right_ratio
        right_ratio -= 1
        if not shifted and 1.0 * abs(top_middle_edge - top_start_middle) / top_avg_width > 0.4:
            if not middle_left and left_ratio * 2 < middle_ratio:
                shifted = True
                top_start_middle = top_first_left_edge
                continue
            if middle_left and right_ratio * 2 < middle_ratio:
                shifted = True
                top_start_middle = top_first_right_edge
                continue
        break

    cv2.line(top_gradients, (top_middle_edge, 1), (top_middle_edge, len(top_gradients)), output.hsv_to_bgr(0, 1, 1))
    cv2.line(top_gradients, (top_first_left_edge, 1), (top_first_left_edge, len(top_gradients)), output.hsv_to_bgr(0.33, 1, 1))
    cv2.line(top_gradients, (top_first_right_edge, 1), (top_first_right_edge, len(top_gradients)), output.hsv_to_bgr(0.33, 1, 1))
    cv2.line(top_gradients, (top_sec_left_edge, 1), (top_sec_left_edge, len(top_gradients)), output.hsv_to_bgr(0.66, 1, 1))
    cv2.line(top_gradients, (top_sec_right_edge, 1), (top_sec_right_edge, len(top_gradients)), output.hsv_to_bgr(0.66, 1, 1))
    # cv2.line(top_gradients, (top_first_left_edge + top_avg_width, 1), (top_first_left_edge + top_avg_width, len(top_gradients)), output.hsv_to_bgr(0.66, 1, 1))


    # output.display_single_image(top_gradients)

    top_centers = []
    top_centers.append((top_sec_left_edge + top_first_left_edge)/2)
    top_centers.append((top_middle_edge + top_first_left_edge)/2)
    top_centers.append((top_middle_edge + top_first_right_edge)/2)
    top_centers.append((top_sec_right_edge + top_first_right_edge)/2)
    top_stds = []

    for tooth_center in top_centers:
        abs_center = (len(gradient_top[0]) / 2) - 300 + tooth_center
        slice = gradient_top[top_start - 80:top_start + 30, abs_center - 20:abs_center + 20]
        per_line = np.mean(slice, axis=1)
        top_stds.append(np.std(per_line))

    median_std = np.median(top_stds)
    for index, std in enumerate(top_stds):
        if median_std > 10:
            top_stds[index] = median_std / 2
        if std > 2 * median_std:
            top_stds[index] = median_std

    for index, tooth_center in enumerate(top_centers):
        abs_center = (len(gradient_top[0]) / 2) - 300 + tooth_center
        slice = gradient_top[top_start - 80:top_start + 30, abs_center - 20:abs_center + 20]
        per_line = np.mean(slice, axis=1)
        mn = np.mean(per_line)
        std = top_stds[index]
        per_line = cv2.GaussianBlur(per_line, (5, 5), 10)
        _, per_line = cv2.threshold(np.uint8(per_line), mn + 1.5 * std, 255, cv2.THRESH_BINARY)
        start = 0
        for index, value in reversed(list(enumerate(per_line))):
            if value == 255:
                start = index
                break
        result.append([abs_center, top_start - 80 + start])
        # output.display_single_image(np.hstack((slice, np.uint8(per_line * np.ones((1, 20))))), scale=3)







    # BOTTOM TEETH HORIZONTAL
    bottom_gradients = gradients[bottom_start+30:bottom_start + 150, middle - 250: middle + 250]
    bottom_hori = sobelx[bottom_start+30:bottom_start + 150, middle - 250: middle + 250]
    bottom_hori = bottom_hori[:, :, 0]

    bot_middle_edge = 0
    bot_first_left_edge = 0
    bot_first_right_edge = 0
    bot_sec_left_edge = 0
    bot_sec_right_edge = 0

    bot_sensitivity = 0.6
    first_iter = True
    last_iter = False
    lowest_std = 0
    best_sens = 0.6

    while True:
        bottom_hori_1 = np.uint8(bottom_hori)
        _, hori_mask_1 = cv2.threshold(bottom_hori_1, 110, 255, cv2.THRESH_BINARY_INV)
        bottom_hori_1 = cv2.bitwise_and(bottom_hori_1, hori_mask_1)
        per_line_hori_bottom_1 = np.mean(bottom_hori_1, axis=0)
        per_line_hori_bottom_1 = cv2.GaussianBlur(per_line_hori_bottom_1, (5, 5), 10)
        _, per_line_hori_bottom_1a = cv2.threshold(np.uint8(per_line_hori_bottom_1), np.max(per_line_hori_bottom_1)*bot_sensitivity, 255, cv2.THRESH_BINARY)


        bottom_hori_2 = np.uint8(bottom_hori * -1)
        _, hori_mask_2 = cv2.threshold(bottom_hori_2, 110, 255, cv2.THRESH_BINARY_INV)
        bottom_hori_2 = cv2.bitwise_and(bottom_hori_2, hori_mask_2)
        per_line_hori_bottom_2 = np.mean(bottom_hori_2, axis=0)
        per_line_hori_bottom_2 = cv2.GaussianBlur(per_line_hori_bottom_2, (5, 5), 10)
        _, per_line_hori_bottom_2a = cv2.threshold(np.uint8(per_line_hori_bottom_2), np.max(per_line_hori_bottom_2)*bot_sensitivity, 255, cv2.THRESH_BINARY)




        test_hori_bottom_1 = np.transpose(np.uint8(per_line_hori_bottom_1 * np.ones((1, 40))))
        test_hori_bottom_1a = np.transpose(np.uint8(per_line_hori_bottom_1a * np.ones((1, 40))))
        test_hori_bottom_2 = np.transpose(np.uint8(per_line_hori_bottom_2 * np.ones((1, 40))))
        test_hori_bottom_2a = np.transpose(np.uint8(per_line_hori_bottom_2a * np.ones((1, 40))))


        # output.display_single_image(np.vstack((bottom_hori_1, test_hori_bottom_1, test_hori_bottom_1a,
        #                                        bottom_hori_2, test_hori_bottom_2, test_hori_bottom_2a
        # )))

        if count_bars(per_line_hori_bottom_1a) < 3 or count_bars(per_line_hori_bottom_2a) < 3:
            bot_sensitivity -= 0.05
            continue

        bot_start_middle = len(per_line_hori_bottom_1a)/2
        bot_middle_edge = find_teeth_borders(per_line_hori_bottom_1a, per_line_hori_bottom_2a, bot_start_middle, bottom_gradients)
        bot_first_left_edge = find_teeth_borders(per_line_hori_bottom_1a, per_line_hori_bottom_2a, bot_middle_edge - 50, bottom_gradients, -1)
        bot_first_right_edge = find_teeth_borders(per_line_hori_bottom_1a, per_line_hori_bottom_2a, bot_middle_edge + 50, bottom_gradients, 1)
        bot_first_left_width = abs(bot_middle_edge - bot_first_left_edge)
        bot_first_right_width = abs(bot_first_right_edge - bot_middle_edge)
        if bot_first_left_width > 80:
            bot_first_left_width /= 2
            bot_first_left_edge += bot_first_left_width
        if bot_first_right_width > 80:
            bot_first_right_width /= 2
            bot_first_right_edge -= bot_first_right_width
        bot_avg_width = (bot_first_left_width + bot_first_right_width) / 2
        bot_sec_left_edge = find_teeth_borders(per_line_hori_bottom_1a, per_line_hori_bottom_2a, int(bot_first_left_edge - bot_avg_width*0.9), bottom_gradients, -1)
        bot_sec_right_edge = find_teeth_borders(per_line_hori_bottom_1a, per_line_hori_bottom_2a, int(bot_first_right_edge + bot_avg_width*0.9), bottom_gradients, 1)
        # _, per_line_hori_bottom = cv2.threshold(np.uint8(per_line_hori_bottom), 8, 255, cv2.THRESH_BINARY)
        bot_sec_left_width = bot_first_left_edge - bot_sec_left_edge
        bot_sec_right_width = bot_sec_right_edge - bot_first_right_edge
        std = np.std([bot_sec_left_width, bot_first_left_width, bot_first_right_width, bot_sec_right_width])
        if std > 10:
            if last_iter:
                break
            if bot_sensitivity < 0.2:
                last_iter = True
                bot_sensitivity = best_sens
                continue
            if first_iter:
                lowest_std = std
                best_sens = bot_sensitivity
                bot_sensitivity -= 0.05
                continue
            elif std < lowest_std:
                lowest_std = std
                best_sens = bot_sensitivity
                bot_sensitivity -= 0.05
                continue
        else:
            break
    cv2.line(bottom_gradients, (bot_middle_edge, 1), (bot_middle_edge, len(bottom_gradients)), output.hsv_to_bgr(0, 1, 1))
    cv2.line(bottom_gradients, (bot_first_left_edge, 1), (bot_first_left_edge, len(bottom_gradients)), output.hsv_to_bgr(0.33, 1, 1))
    cv2.line(bottom_gradients, (bot_first_right_edge, 1), (bot_first_right_edge, len(bottom_gradients)), output.hsv_to_bgr(0.33, 1, 1))
    cv2.line(bottom_gradients, (bot_sec_left_edge, 1), (bot_sec_left_edge, len(bottom_gradients)), output.hsv_to_bgr(0.66, 1, 1))
    cv2.line(bottom_gradients, (bot_sec_right_edge, 1), (bot_sec_right_edge, len(bottom_gradients)), output.hsv_to_bgr(0.66, 1, 1))

    # output.display_single_image(bottom_gradients)

    bottom_centers = []
    bottom_centers.append((bot_sec_left_edge + bot_first_left_edge)/2)
    bottom_centers.append((bot_middle_edge + bot_first_left_edge)/2)
    bottom_centers.append((bot_middle_edge + bot_first_right_edge)/2)
    bottom_centers.append((bot_sec_right_edge + bot_first_right_edge)/2)

    bot_stds = []
    for tooth_center in bottom_centers:
        abs_center = (len(gradient_bottom[0]) / 2) - 250 + tooth_center
        slice = gradient_bottom[bottom_start - 30:bottom_start + 30, abs_center - 10:abs_center + 10]
        per_line = np.mean(slice, axis=1)
        bot_stds.append(np.std(per_line))

    median_std = np.median(bot_stds)
    for index, std in enumerate(bot_stds):
        # print(std)
        if std > 2 * median_std:
            bot_stds[index] = median_std

    for index, tooth_center in enumerate(bottom_centers):
        abs_center = (len(gradient_bottom[0]) / 2) - 250 + tooth_center
        slice = gradient_bottom[bottom_start - 30:bottom_start + 30, abs_center - 10:abs_center + 10]
        per_line = np.mean(slice, axis=1)
        mn = np.mean(per_line)
        std = bot_stds[index]
        # per_line = cv2.GaussianBlur(per_line, (5, 5), 10)
        _, per_line = cv2.threshold(np.uint8(per_line), mn + 1.5*std, 255, cv2.THRESH_BINARY)
        start = 0
        for index, value in enumerate(per_line):
            if value == 255:
                start = index
                break
        result.append([abs_center, bottom_start - 30 + start])
        # output.display_single_image(np.hstack((slice, np.uint8(per_line * np.ones((1, 20))))), scale=3)


    # output.display_single_image(top_gradients)
    # output.display_single_image(np.hstack((result_top, test_top)))
    return result


def count_bars(block):
    in_bar = False
    count = 0
    for i in range(0, len(block)-1):
        if block[i] == 0:
            if in_bar:
                in_bar = False
            continue
        else:
            if not in_bar:
                count += 1
                in_bar = True
            continue
    return count

def find_teeth_borders(left, right, start, gradients, force_direction=0):
    gradients = np.copy(gradients)
    # cv2.line(gradients, (start, 1), (start, len(gradients)), output.hsv_to_bgr(0, 1, 1))
    lefta = cv2.cvtColor(np.uint8(np.transpose(np.uint8(left * np.ones((1, 40))))), cv2.COLOR_GRAY2BGR)
    righta = cv2.cvtColor(np.uint8(np.transpose(np.uint8(right * np.ones((1, 40))))), cv2.COLOR_GRAY2BGR)
    #

    # find middle
    search_direction = -1
    i = 0
    left_edge_found = False
    skip_current = False
    first_border = 0
    while i in range(0, start):
        if force_direction == 1 and start+i >= len(left):
            return start + i - 10
        if force_direction == -1 and start-i <= 0:
            return 10
        if left[start] == 255:
            search_direction = -1
            left_edge_found = True
            skip_current = True
            break
        if right[start] == 255:
            search_direction = 1
            left_edge_found = False
            skip_current = True
            break
        if left[start + i] == 255 and force_direction != -1:
            search_direction = -1
            left_edge_found = True
            skip_current = False
            first_border = start + i
            break
        elif right[start + i] == 255 and force_direction != -1:
            search_direction = 1
            left_edge_found = False
            skip_current = True
            break
        elif left[start - i] == 255 and force_direction != 1:
            search_direction = -1
            left_edge_found = True
            skip_current = True
            break
        elif right[start - i] == 255 and force_direction != 1:
            search_direction = 1
            left_edge_found = False
            skip_current = False
            first_border = start - i

        i += 1
    if left_edge_found:
        first_edge_found = left
        other_edge = right
    else:
        first_edge_found = right
        other_edge = left

    if skip_current:
        while start + search_direction*i < len(first_edge_found)-1 and first_edge_found[start + search_direction*i] == 255:
            i += 1
        first_border = start + search_direction*i

    while start + search_direction*i < len(other_edge)-1 and other_edge[start + search_direction*i] != 255:
        i += 1
    other_border = start + search_direction*i

    e1 = find_block_middle(first_edge_found, first_border) #red
    e2 = find_block_middle(other_edge, first_border) #blue
    middle = ( e1 + e2)/2

    # cv2.line(gradients, (middle, 1), (middle, len(gradients)), output.hsv_to_bgr(0.33, 1, 1))
    # cv2.line(gradients, (e1, 1), (e1, len(gradients)), output.hsv_to_bgr(0, 1, 1))
    # cv2.line(gradients, (e2, 1), (e2, len(gradients)), output.hsv_to_bgr(0.66, 1, 1))
    # output.display_single_image(np.vstack((gradients, lefta, righta)))



    # output.display_single_image(np.vstack((gradients, lefta, righta)))
    return middle


def find_block_middle(block, start):
    i = start
    while i > 0 and block[i] == 255:
        i -= 1

    left = i
    i = start
    while i < len(block) and block[i] == 255:
        i += 1
    right = i
    middle = (left + right) / 2
    return middle

def process_radiographs():
    crop_offsets = []
    gradient_set = []
    for image_nb in range(1, TOTAL_RADIO_AMOUNT + 1):
        raw_image = input.import_radiograph(image_nb)
        crop_top_left, crop_bottom_right = calculate_crop_data(raw_image, X_CROP_RATIO, Y_CROP_RATIO, Y_CROP_OFFSET)
        crop_offsets.append(crop_top_left)

        if REPROCESS_IMAGES:
            mono_image = raw_image[:, :, 0]

            cropped_image = crop(mono_image, crop_top_left, crop_bottom_right)
            output.save_image(cropped_image, str(image_nb), CROPPED_DIR)

            homomorphic_image = homomorphic_filter(cropped_image)
            output.save_image(homomorphic_image, str(image_nb), HOMOMORPHIC_DIR)
            normalized_image = normalize_histogram(homomorphic_image, True)
            output.save_image(normalized_image, str(image_nb), NORMALIZED_DIR)
        else:
            homomorphic_image = cv2.imread(HOMOMORPHIC_DIR + str(image_nb) + ".png", 0)

        sobel_image, gradient_vectors = sobel(homomorphic_image)
        gradient_set.append(gradient_vectors)

        if REPROCESS_IMAGES:
            output.save_image(sobel_image, str(image_nb), SOBEL_DIR)

            sobel_image = denoise_image(sobel_image)
            output.save_image(sobel_image, str(image_nb), SOBEL_DENOISED_DIR)

            gradient_image = generate_gradient_from_vectors(sobel_image, gradient_vectors)
            output.save_image(gradient_image, str(image_nb), GRADIENT_DIR)

    return crop_offsets, gradient_set

