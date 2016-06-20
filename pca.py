import cv2
import numpy as np


def project(w, x, mu, raw=False):
    '''
    Project X on the space spanned by the vectors in W.
    mu is the average image.
    '''
    if not raw:
        x = np.hstack(x)
    return np.dot(cv2.transpose(w), (x - mu))


def reconstruct(w, y, mu, raw=False):
    '''
    Reconstruct an image based on its PCA-coefficients Y, the eigenvectors W and the average mu.
    '''
    raw_result = mu + np.dot(w, y)
    if raw:
        return raw_result
    result = np.zeros((len(raw_result)/2, 2))
    for index in range(0, len(raw_result)/2):
            result[index] = [raw_result[2*index], raw_result[2*index + 1]]
    return result


def pca(landmarks, nb_components=0):
    '''
    Do a PCA analysis on X
    @param landmarks:                np.array containing the samples
                             shape = (nb samples, nb dimensions of each sample)
    @param nb_components:    the nb components we're interested in
    @return: return the nb_components largest eigenvalues and eigenvectors of the covariance matrix and return the average sample
    '''
    n = len(landmarks)
    d = len(landmarks[0]) * 2
    data = np.zeros((n, d))
    for index, sample in enumerate(landmarks):
        data[index, :] = np.hstack(sample)

    if (nb_components <= 0) or (nb_components > n):
        nb_components = n

    # average image
    mu = np.zeros(d)
    for img in data:
        mu += img
    mu /= n

    # covariance matrix
    data[:] -= mu
    xt = cv2.transpose(data)
    cov = np.dot(data, xt)

    _, eigenvalues, eigenvectors = cv2.eigen(cov, True)

    eigenvectors = np.dot(xt, cv2.transpose(eigenvectors))

    return eigenvalues[0:nb_components], cv2.normalize(eigenvectors[:, 0:nb_components]), mu


def vary_pca_parameter(parameter, std, pca_data):
    _, eigenvectors, mu = pca_data
    result = []
    num_dim = len(eigenvectors[0])
    params = np.zeros(num_dim)

    params[parameter] = -3 * std[parameter]
    result.append(reconstruct(eigenvectors, params, mu))

    params[parameter] = 0
    result.append(reconstruct(eigenvectors, params, mu))

    params[parameter] = 3 * std[parameter]
    result.append(reconstruct(eigenvectors, params, mu))

    return result


def calculate_std(landmarks_set, eigenvectors, mu):
    num_dim = len(eigenvectors[0])
    num_val = len(landmarks_set)
    data = np.zeros((num_val, num_dim))
    for index, landmarks in enumerate(landmarks_set):
        data[index, :] = project(eigenvectors, landmarks, mu)
    result = []
    for index in range(0, num_dim):
        result.append(np.std(data[:, index]))
    return result


def calculate_all_std(all_teeth, pca_data):
    result = []
    for index, landmarks_set in enumerate(all_teeth):
        _, eigenvectors, mu = pca_data[index]
        result.append(calculate_std(landmarks_set, eigenvectors, mu))
    return result


def pca_all(all_teeth, nb_components=0):
    result = []
    for landmarks in all_teeth:
        result.append(pca(landmarks, nb_components))
    return result
