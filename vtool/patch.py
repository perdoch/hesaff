# LICENCE
from __future__ import print_function, division
# Python
# Science
import cv2
import numpy as np
# HotSpotter
from hsviz import extract_patch
# VTool
from vtool.hist import *  # NOQA


def patch_gradient(image, ksize=1):
    image_ = np.array(image, dtype=np.float64)
    gradx = cv2.Sobel(image_, cv2.CV_64F, 1, 0, ksize=ksize)
    grady = cv2.Sobel(image_, cv2.CV_64F, 0, 1, ksize=ksize)
    return gradx, grady


def patch_mag(gradx, grady):
    return np.sqrt((gradx ** 2) + (grady ** 2))


def patch_ori(gradx, grady):
    np.tau = 2 * np.pi
    gori = np.arctan2(grady, gradx)  # outputs from -pi to pi
    gori[gori < 0] = gori[gori < 0] + np.tau  # map to 0 to tau (keep coords)
    return gori


def get_patch(imgBGR, kp):
    wpatch, wkp = extract_patch.get_warped_patch(imgBGR, kp)
    wpatchLAB = cv2.cvtColor(wpatch, cv2.COLOR_BGR2LAB)
    wpatchL = wpatchLAB[:, :, 0]
    return wpatchL


def get_patch_grad_ori(imgBGR, kp):
    patch = get_patch(imgBGR, kp)
    gradx, grady = patch_gradient(patch)
    gmag = patch_mag(gradx, grady)
    gori = patch_ori(gradx, grady)
    return patch, gradx, grady, gmag, gori


def rotation_matrix(radians):
    sin_ = np.sin(radians)
    cos_ = np.cos(radians)
    rot = np.array(((cos_, -sin_),
                    (sin_,  cos_)))
    return rot


def get_orientation_histogram(gori):
    # Get wrapped histogram (because we are finding a direction)
    hist_, edges_ = np.histogram(gori.flatten(), bins=8)
    hist, edges = wrap_histogram(hist_, edges_)
    centers = hist_edges_to_centers(edges)
    return hist, centers


def find_dominant_gradient_direction(gori):
    hist, centers = get_orientation_histogram(gori)
    # Find submaxima
    maxima_x, maxima_y, argmaxima = hist_argmaxima(hist, centers)
    submaxima_x, submaxima_y = interpolate_submaxima(argmaxima, hist, centers)
    rad = submaxima_x[submaxima_y.argmax()]
    return rad


def find_kpts_direction(imgBGR, kpts):
    rad_list = []
    for kp in kpts:
        patch = get_patch(imgBGR, kp)
        gradx, grady = patch_gradient(patch)
        gori = patch_ori(gradx, grady)
        hist, centers = get_orientation_histogram(gori)
        # Find submaxima
        maxima_x, maxima_y, argmaxima = hist_argmaxima(hist, centers)
        submaxima_x, submaxima_y = interpolate_submaxima(argmaxima, hist, centers)
        rad = submaxima_x[submaxima_y.argmax()]
        rad_list.append(rad)
    print(kpts.shape)
    print(len(rad_list))
    kpts2 = np.vstack([kpts.T, rad_list]).T
    return kpts2
