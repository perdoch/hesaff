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
from hscom import util  # NOQA
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
import vtool


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


def draw_hist_subbin_maxima(hist, centers=None, fnum=None, pnum=None):
    # Find maxima
    maxima_x, maxima_y, argmaxima = vtool.hist_argmaxima(hist, centers)
    # Expand parabola points around submaxima
    x123, y123 = vtool.maxima_neighbors(argmaxima, hist, centers)
    # Find submaxima
    submaxima_x, submaxima_y = vtool.interpolate_submaxima(argmaxima, hist, centers)
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
    patch, gradx, grady, gmag, gori = vtool.get_patch_grad_ori(imgBGR, kp)

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
    hist, centers = vtool.get_orientation_histogram(gori)
    # Find orientation submaxima
    maxima_x, maxima_y, argmaxima = vtool.hist_argmaxima(hist, centers)
    submaxima_x, submaxima_y = vtool.interpolate_submaxima(argmaxima, hist, centers)

    # Draw histogram with interpolation annotations
    draw_hist_subbin_maxima(hist, centers, fnum=3, pnum=(1, 1, 1))

    # Get dominant direction in radians
    rad = vtool.find_dominant_gradient_direction(gori)

    # Augment keypoint and plot rotated patch
    kp2 = np.hstack([kp, [rad]])
    wpatch, wkp = extract_patch.get_warped_patch(imgBGR, kp2)
    df2.imshow(wpatch, fnum=4)

    #maxima, argmaxima = hist_argmaxima(hist, centers)
    #submaxima = interpolate_submaxima(argmaxima, hist, centers)

    df2.update()

    kpts2 = vtool.find_kpts_direction(imgBGR, kpts)

    #viz.show_keypoints(rchip, kpts)
    interact.interact_keypoints(imgBGR, kpts2, desc, arrow=True, rect=True)
    exec(df2.present(wh=800))
