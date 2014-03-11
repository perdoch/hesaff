# LICENCE
from __future__ import print_function, division
# Python
# Science
import cv2
import numpy as np
from numpy import array, sqrt
# VTool
import vtool.histogram as htool
import vtool.keypoint as ktool
import vtool.linalg as ltool


def patch_gradient(image, ksize=1):
    image_ = array(image, dtype=np.float64)
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


def get_warped_patch(rchip, kp):
    'Returns warped patch around a keypoint'
    if len(kp) == 5:
        (x, y, a, c, d), r = kp, 0
    else:
        (x, y, a, c, d, r) = kp
    sfx, sfy = ktool.orthogonal_scales(kp.reshape((1, kp.size)))[0, :]
    s = 41  # sf
    ss = sqrt(s) * 3
    (h, w) = rchip.shape[0:2]
    # Translate to origin(0,0) = (x,y)
    T = ltool.translation_mat(-x, -y)
    A = ktool.warp_to_circle_mat(a, c, d)
    R = ltool.rotation_mat(r)
    S = ltool.scale_mat(ss)
    X = ltool.translation_mat(s / 2, s / 2)
    rchip_h, rchip_w = rchip.shape[0:2]
    dsize = np.array(np.ceil(np.array([s, s])), dtype=int)
    M = X.dot(S).dot(R).dot(A).dot(T)
    cv2_flags = cv2.INTER_LANCZOS4
    cv2_borderMode = cv2.BORDER_CONSTANT
    cv2_warp_kwargs = {'flags': cv2_flags, 'borderMode': cv2_borderMode}
    warped_patch = cv2.warpAffine(rchip, M[0:2], tuple(dsize), **cv2_warp_kwargs)
    #warped_patch = cv2.warpPerspective(rchip, M, dsize, **__cv2_warp_kwargs())
    wkp = np.array([(s / 2, s / 2, ss, 0., ss)])
    return warped_patch, wkp


def get_patch(imgBGR, kp):
    wpatch, wkp = get_warped_patch(imgBGR, kp)
    wpatchLAB = cv2.cvtColor(wpatch, cv2.COLOR_BGR2LAB)
    wpatchL = wpatchLAB[:, :, 0]
    return wpatchL


def get_orientation_histogram(gori):
    # Get wrapped histogram (because we are finding a direction)
    hist_, edges_ = np.histogram(gori.flatten(), bins=8)
    hist, edges = htool.wrap_histogram(hist_, edges_)
    centers = htool.hist_edges_to_centers(edges)
    return hist, centers


def find_kpts_direction(imgBGR, kpts):
    theta_list = []
    for kp in kpts:
        patch = get_patch(imgBGR, kp)
        gradx, grady = patch_gradient(patch)
        gori = patch_ori(gradx, grady)
        hist, centers = get_orientation_histogram(gori)
        # Find submaxima
        maxima_x, maxima_y, argmaxima = htool.hist_argmaxima(hist, centers)
        submaxima_x, submaxima_y = htool.interpolate_submaxima(argmaxima, hist, centers)
        theta = submaxima_x[submaxima_y.argmax()]
        theta_list.append(theta)
    print(kpts.shape)
    print(len(theta_list))
    kpts2 = np.vstack([kpts.T, theta_list]).T
    return kpts2
