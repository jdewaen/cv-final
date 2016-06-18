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


def detect_mouth(imn, vectors):
    # thres = cv2.threshold(img,)
    homo_image = cv2.imread(HOMOMORPHIC_DIR + str(imn) + ".png")
    gradients = cv2.imread(GRADIENT_DIR + str(imn) + ".png")
    sobel = cv2.imread(SOBEL_DENOISED_DIR + str(imn) + ".png", 0)
    normalized = cv2.imread(NORMALIZED_DIR + str(imn) + ".png", 0)
    norm_invert = np.uint8(np.ones(normalized.shape) * 255 - normalized)
    _, thresh = cv2.threshold(norm_invert, 200, 255, cv2.THRESH_BINARY)
    per_line = np.mean(thresh, axis=1)
    per_line = cv2.GaussianBlur(per_line, (5, 5), 10)
    means = np.uint8(per_line * np.ones((1, 100)))
    mouth = 250 + np.argmax(per_line[250:])
    normalized = cv2.cvtColor(np.uint8(normalized), cv2.COLOR_GRAY2BGR)
    means = cv2.cvtColor(np.uint8(means), cv2.COLOR_GRAY2BGR)
    cv2.line(gradients, (1, mouth), (len(gradients[0]), mouth), output.hsv_to_bgr(0, 1, 1))
    # result = np.hstack((normalized, means))
    middle = len(normalized[0]) / 2
    sobely = cv2.Sobel(homo_image, cv2.CV_64F, 0, 1, ksize=3)
    result_bottom = sobely[mouth - 20:mouth + 150, middle - 100:middle + 100]

    # lowest = np.min(result)
    # result -= lowest
    # highest = np.max(result)
    # result = np.float64(result)
    # result /= highest
    result_bottom = result_bottom[:, :, 0]
    result_bottom = np.uint8(result_bottom)

    _, mask = cv2.threshold(result_bottom, 128, 255, cv2.THRESH_BINARY_INV)
    result_bottom = cv2.bitwise_and(result_bottom, mask)
    per_line_bottom = np.mean(result_bottom, axis=1)
    per_line_bottom = cv2.GaussianBlur(per_line_bottom, (5, 5), 10)
    _, per_line_bottom = cv2.threshold(np.uint8(per_line_bottom), 8, 255, cv2.THRESH_BINARY)
    bottom_start = 0
    for index, value in enumerate(per_line_bottom):
        if value == 255:
            bottom_start = index
            break
    cv2.line(result_bottom, (1, bottom_start), (len(result_bottom[0]), bottom_start), 255)
    bottom_start -= 20 - mouth
    cv2.line(gradients, (1, bottom_start), (len(gradients[0]), bottom_start), output.hsv_to_bgr(0.33, 1, 1))

    result_top = sobely[mouth - 70:mouth + 50, middle - 100:middle + 100]
    # result_top = normalized[mouth - 70:mouth + 60, middle - 100:middle + 100]
    result_top = result_top[:, :, 0]
    result_top = np.uint8(result_top * -1)
    _, maskinv = cv2.threshold(result_top, 128, 255, cv2.THRESH_BINARY_INV)
    result_top = cv2.bitwise_and(result_top, maskinv)
    per_line_top = np.mean(result_top, axis=1)
    per_line_top = cv2.GaussianBlur(per_line_top, (1, 1), 10)
    _, per_line_top = cv2.threshold(np.uint8(per_line_top), 9, 255, cv2.THRESH_BINARY)
    top_start = 0
    for index, value in reversed(list(enumerate(per_line_top))):
        if value == 255:
            top_start = index
            break
    cv2.line(result_top, (1, top_start), (len(result_top[0]), top_start), 255)
    top_start -= 70 - mouth
    cv2.line(gradients, (1, top_start), (len(gradients[0]), top_start), output.hsv_to_bgr(0.66, 1, 1))
    test_top = np.uint8(per_line_top * np.ones((1, 100)))

    # output.display_single_image(gradients)
    # output.display_single_image(np.hstack((result_top, test_top)))
    return top_start, bottom_start

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

