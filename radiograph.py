import input
import output
import cv2
import scipy.signal
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


def crop(image, x_ratio, y_ratio):
    height = len(image)
    width = len(image[0])
    x_offset = width * (1 - x_ratio) / 2
    y_offset = height * (1 - y_ratio) / 2
    new_image = image[y_offset:height-y_offset, x_offset:width-x_offset]
    return new_image, (y_offset, x_offset)


def sobel(img):

    # img = cv2.blur(img, (5, 5))
    img = homomorphic_filter(img)
    # img = cv2.blur(img, (3, 3))

    # img = normalize_histogram(img, True)
    # img = np.uint8(img)


    #

    # sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    # sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    # sobelx = cv2.filter2D(img, cv2.CV_64F, np.array([[-1,0,+1],[-1,0,+1],[-1,0,+1]]))
    # sobely = cv2.filter2D(img, cv2.CV_64F, np.array([[-1,-1,-1],[0,0,0],[+1,+1,+1]]))
    # sobelx = cv2.filter2D(img, cv2.CV_64F, np.array([[+1,0],[0,-1]]))
    # sobely = cv2.filter2D(img, cv2.CV_64F, np.array([[0,+1],[-1,0]]))
    #
    # result = np.sqrt(np.square(sobelx) + np.square(sobely))
    # result[:, :] = result[:, :]*(IMAGE_BIT_DEPTH-1)
    # result[:, :] = result[:, :] + (IMAGE_BIT_DEPTH/2)
    # result = np.uint8(result)

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    result = np.sqrt(np.square(sobelx) + np.square(sobely))
    result[:, :] = abs(result[:, :])
    result = np.uint8(result)

    # result = cv2.Canny(img, 25, 45)
    # img = normalize_histogram(img, True)
    return result

def sobel_only(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    result = np.sqrt(np.square(sobelx) + np.square(sobely))
    vects = np.zeros((img.shape[0], img.shape[1], 2))
    vects[:, :, 0] = sobelx
    vects[:, :, 1] = sobely
    # test[:, :] /= np.pi * 2
    # test[:, :] += 0.5
    # test[:, :] *= IMAGE_BIT_DEPTH - 1
    # bla = np.zeros((test.shape[0], test.shape[1], 3))
    # for y, row in enumerate(test):
    #     for x, val in enumerate(row):
    #         res = result[y, x] / 255
    #         bla[y, x] = output.hsv_to_bgr(val, 1, res)
    # result[:, :] = abs(result[:, :])
    # result = normalize_histogram(np.uint8(result), True)
    result = np.uint8(result)
    return result, vects


def homo_3(img):
    return

def homomorphic_filter(img):

    rows = img.shape[0]
    cols = img.shape[1]
    # Convert image to 0 to 1, then do log(1 + I)
    imgLog = np.log1p(np.array(img, dtype="float") / 255)

    # Create Gaussian mask of sigma = 10
    M = 2 * rows + 1
    N = 2 * cols + 1
    sigma = 10
    (X, Y) = np.meshgrid(np.linspace(0, N - 1, N), np.linspace(0, M - 1, M))
    centerX = np.ceil(N / 2)
    centerY = np.ceil(M / 2)
    gaussianNumerator = (X - centerX) ** 2 + (Y - centerY) ** 2

    # Low pass and high pass filters
    Hlow = np.exp(-gaussianNumerator / (2 * sigma * sigma))
    Hhigh = 1 - Hlow

    # Move origin of filters so that it's at the top left corner to
    # match with the input image
    HlowShift = scipy.fftpack.ifftshift(Hlow.copy())
    HhighShift = scipy.fftpack.ifftshift(Hhigh.copy())

    # Filter the image and crop
    If = scipy.fftpack.fft2(imgLog.copy(), (M, N))
    Ioutlow = scipy.real(scipy.fftpack.ifft2(If.copy() * HlowShift, (M, N)))
    Iouthigh = scipy.real(scipy.fftpack.ifft2(If.copy() * HhighShift, (M, N)))

    # Set scaling factors and add
    gamma1 = 0.3
    gamma2 = 1.5
    Iout = gamma1 * Ioutlow[0:rows, 0:cols] + gamma2 * Iouthigh[0:rows, 0:cols]

    # Anti-log then rescale to [0,1]
    Ihmf = np.expm1(Iout)
    Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))
    Ihmf2 = np.array(255 * Ihmf, dtype="uint8")
    return Ihmf2




def process_radiographs():
    for image_nb in range(1, PROCESSED_RADIO_AMOUNT + 1):
        # raw_image = input.import_radiograph(image_nb)
        # mono_image = raw_image[:, :, 0]
        # cropped_image, shift = crop(mono_image, X_CROP_RATIO, Y_CROP_RATIO)
        # downsampled = cv2.resize(cropped_image, (0, 0), fx=0.5, fy=0.5)
        # # output.display_single_image(cropped_image, scale=0.5)
        # result = sobel_only(downsampled)
        # canny = cv2.imread("processed/5-5 blur 15-20 canny/proc.15.20-" + str(image_nb) + ".png",-1)
        pass
        # canny = sobel_only(full[:, :, 0])
        # output.save_image(result, "proc-"+str(image_nb), PROCESSED_DIR + "cont/")
        # homomorphic_image = homomorphic_filter(cropped_image)
        # processed_image = normalize_histogram(homomorphic_image)
        # output.display_single_image(processed_image, scale=0.5)
        # processed_image = normalize_histogram(cropped_image)
        # output.display_single_image(processed_image, scale=0.5)

        # output.save_image(processed_image,"norm-"+str(image_nb), PROCESSED_DIR)

