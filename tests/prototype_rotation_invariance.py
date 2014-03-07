#!/usr/bin/env python
from __future__ import print_function, division
# Standard
import sys
from os.path import join, exists, realpath, expanduser
import multiprocessing
# Scientific
import numpy as np
import cv2
# Hotspotter
from hscom import helpers  # NOQA
from hsviz import draw_func2 as df2
from hsviz import extract_patch
from hsviz import viz  # NOQA
from hsviz import interact  # NOQA
from hscom import fileio as io
from hscom import __common__
# TPL
import pyhesaff
print, print_, print_on, print_off, rrr, profile, printDBG =\
    __common__.init(__name__, module_prefix='[testhesaff]', DEBUG=False, initmpl=False)


def ensure_hotspotter():
    import matplotlib
    matplotlib.use('Qt4Agg', warn=True, force=True)
    # Look for hotspotter in ~/code
    hotspotter_dir = join(expanduser('~'), 'code', 'hotspotter')
    if not exists(hotspotter_dir):
        print('[jon] hotspotter_dir=%r DOES NOT EXIST!' % (hotspotter_dir,))
    # Append hotspotter to PYTHON_PATH (i.e. sys.path)
    if not hotspotter_dir in sys.path:
        sys.path.append(hotspotter_dir)


def load_test_data(short=False, n=0):
    if not 'short' in vars():
        short = False
    # Read Image
    #ellipse.rrr()
    nScales = 4
    nSamples = 16
    img_fname = 'zebra.png'
    #img_fname = 'lena.png'
    img_fpath = realpath(img_fname)
    imgBGR = io.imread(img_fpath)
    imgLAB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2LAB)
    imgL = imgLAB[:, :, 0]
    kpts, desc = pyhesaff.detect_kpts(img_fpath, scale_min=20, scale_max=100)
    if short:
        extra_fxs = []
        if img_fname == 'zebra.png':
            extra_fxs = [374, 520, 880][0:1]
        fxs = np.array(spaced_elements2(kpts, n).tolist() + extra_fxs)
        kpts = kpts[fxs]
        desc = desc[fxs]
    test_data = locals()
    return test_data


def spaced_elements2(list_, n):
    if n is None:
        return np.arange(len(list_))
    if n == 0:
        return np.empty(0)
    indexes = np.arange(len(list_))
    stride = len(indexes) // n
    return indexes[0:-1:stride]


def hist_edges_to_centers(edges):
    from itertools import izip
    return np.array([(e1 + e2) / 2 for (e1, e2) in izip(edges[:-1], edges[1:])])


def wrap_histogram(hist, edges):
    low, high = np.diff(edges)[[0, -1]]
    hist_wrap = np.hstack((hist[-1:], hist, hist[0:1]))
    edge_wrap = np.hstack((edges[0:1] - low, edges, edges[-1:] + high))
    return hist_wrap, edge_wrap


def hist_argmaxima(hist, centers=None):
    from scipy.signal import argrelextrema
    argmaxima = argrelextrema(hist, np.greater)[0]
    maxima_x = argmaxima if centers is None else centers[argmaxima]
    maxima_y = hist[argmaxima]
    return maxima_x, maxima_y, argmaxima


def maxima_neighbors(argmaxima, hist, centers=None):
    neighbs = np.vstack((argmaxima - 1, argmaxima, argmaxima + 1))
    y123 = hist[neighbs]
    x123 = neighbs if centers is None else centers[neighbs]
    return x123, y123


def interpolate_submaxima(argmaxima, hist, centers=None):
    x123, y123 = maxima_neighbors(argmaxima, hist, centers)
    (y1, y2, y3) = y123
    (x1, x2, x3) = x123
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    A     = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    B     = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
    C     = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom
    xv = -B / (2 * A)
    yv = C - B * B / (4 * A)
    submaxima_x, submaxima_y = np.vstack((xv.T, yv.T))
    return submaxima_x, submaxima_y


def draw_hist_subbin_maxima(hist, centers=None, fnum=None, pnum=None):
    # Find maxima
    maxima_x, maxima_y, argmaxima = hist_argmaxima(hist, centers)
    # Expand parabola points around submaxima
    x123, y123 = maxima_neighbors(argmaxima, hist, centers)
    # Find submaxima
    submaxima_x, submaxima_y = interpolate_submaxima(argmaxima, hist, centers)
    xpoints = []
    ypoints = []
    for (x1, x2, x3), (y1, y2, y3) in zip(x123.T, y123.T):
        coeff = np.polyfit((x1, x2, x3), (y1, y2, y3), 2)
        x_pts = np.linspace(x1, x3, 50)
        y_pts = np.polyval(coeff, x_pts)
        xpoints.append(x_pts)
        ypoints.append(y_pts)

    df2.figure(fnum=fnum, pnum=pnum, doclf=True, docla=True)
    df2.plot(centers, hist, 'bo-')
    # Draw maxbin
    df2.plot(maxima_x, maxima_y, 'ro')
    # Draw maxsubbin
    df2.plot(submaxima_x, submaxima_y, 'rx')
    # Draw parabola
    from itertools import izip
    for x_pts, y_pts in izip(xpoints, ypoints):
        df2.plot(x_pts, y_pts, 'g--')
    df2.update()


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


if __name__ == '__main__':
    multiprocessing.freeze_support()
    ensure_hotspotter()
    np.set_printoptions(threshold=5000, linewidth=5000, precision=3)

    # Read data
    test_data = load_test_data(short=True, n=3)
    kpts = test_data['kpts']
    desc = test_data['desc']
    imgBGR = test_data['imgBGR']
    kp = kpts[1]

    # Extract things to viz
    patch, gradx, grady, gmag, gori = get_patch_grad_ori(imgBGR, kp)

    df2.update()
    nRow, nCol = (2, 3)
    df2.figure(fnum=1, pnum=(nRow, nCol, 1))
    # Show patch, gradients, magintude, and orientation
    df2.imshow(patch, pnum=(nRow, nCol, 1), fnum=1)
    df2.imshow(gradx, pnum=(nRow, nCol, 2), fnum=1)
    df2.imshow(grady, pnum=(nRow, nCol, 3), fnum=1)
    df2.imshow(gmag, pnum=(nRow, nCol, 4), fnum=1)
    df2.draw_vector_field(gradx, grady, pnum=(nRow, nCol, 5), fnum=1)

    def color_orimag(gori, gmag, fnum=None, pnum=None):
        # Turn a 0 to 1 orienation map into hsv colors
        gori_01 = (gori - gori.min()) / (gori.max() - gori.min())
        cmap_ = df2.plt.get_cmap('hsv')
        flat_rgb = np.array(cmap_(gori_01.flatten()), dtype=np.float32)
        rgb_ori_alpha = flat_rgb.reshape(np.hstack((gori.shape, [4])))
        rgb_ori = cv2.cvtColor(rgb_ori_alpha, cv2.COLOR_RGBA2RGB)
        hsv_ori = cv2.cvtColor(rgb_ori, cv2.COLOR_RGB2HSV)
        # Desaturate colors based on magnitude
        hsv_ori[:, :, 1] = (gmag / 255.0)
        hsv_ori[:, :, 2] = (gmag / 255.0)
        # Convert back to bgr
        bgr_ori = cv2.cvtColor(hsv_ori, cv2.COLOR_HSV2RGB)
        return bgr_ori
    gorimag = color_orimag(gori, gmag)
    df2.imshow(gorimag, pnum=(nRow, nCol, 6), fnum=1)

    # Get orientation histogram
    hist, centers = get_orientation_histogram(gori)
    # Find orientation submaxima
    maxima_x, maxima_y, argmaxima = hist_argmaxima(hist, centers)
    submaxima_x, submaxima_y = interpolate_submaxima(argmaxima, hist, centers)

    # Draw histogram with interpolation annotations
    draw_hist_subbin_maxima(hist, centers, fnum=3, pnum=(1, 1, 1))

    # Get dominant direction in radians
    rad = find_dominant_gradient_direction(gori)

    # Augment keypoint and plot rotated patch
    kp2 = np.hstack([kp, [rad]])
    wpatch, wkp = extract_patch.get_warped_patch(imgBGR, kp2)
    df2.imshow(wpatch, fnum=4)

    #maxima, argmaxima = hist_argmaxima(hist, centers)
    #submaxima = interpolate_submaxima(argmaxima, hist, centers)

    df2.update()

    kpts2 = find_kpts_direction(imgBGR, kpts)

    #viz.show_keypoints(rchip, kpts)
    interact.interact_keypoints(imgBGR, kpts2, desc, arrow=True, rect=True)
    exec(df2.present(wh=800))
