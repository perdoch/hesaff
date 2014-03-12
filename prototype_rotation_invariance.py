#!/usr/bin/env python
from __future__ import print_function, division
# Standard
from itertools import izip
import multiprocessing
# Scientific
import numpy as np
import cv2
# Hotspotter
from hscom import util  # NOQA
from hsviz import draw_func2 as df2
from hsviz import viz
# TPL
import pyhestest
# VTool
import vtool.patch as ptool
import vtool.histogram as htool


def draw_hist_subbin_maxima(hist, centers=None, fnum=None, pnum=None):
    # Find maxima
    maxima_x, maxima_y, argmaxima = htool.hist_argmaxima(hist, centers)
    # Expand parabola points around submaxima
    x123, y123 = htool.maxima_neighbors(argmaxima, hist, centers)
    # Find submaxima
    submaxima_x, submaxima_y = htool.interpolate_submaxima(argmaxima, hist, centers)
    xpoints = []
    ypoints = []
    for (x1, x2, x3), (y1, y2, y3) in zip(x123.T, y123.T):
        coeff = np.polyfit((x1, x2, x3), (y1, y2, y3), 2)
        x_pts = np.linspace(x1, x3, 50)
        y_pts = np.polyval(coeff, x_pts)
        xpoints.append(x_pts)
        ypoints.append(y_pts)

    df2.figure(fnum=fnum, pnum=pnum, doclf=True, docla=True)
    df2.plot(centers, hist, 'bo-')            # Draw hist
    df2.plot(maxima_x, maxima_y, 'ro')        # Draw maxbin
    df2.plot(submaxima_x, submaxima_y, 'rx')  # Draw maxsubbin
    for x_pts, y_pts in izip(xpoints, ypoints):
        df2.plot(x_pts, y_pts, 'g--')         # Draw parabola


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


if __name__ == '__main__':
    multiprocessing.freeze_support()
    np.set_printoptions(threshold=5000, linewidth=5000, precision=3)
    # Read data
    print('[rot-invar] loading test data')
    test_data = pyhestest.load_test_data(short=True, n=3)
    kpts = test_data['kpts']
    desc = test_data['desc']
    imgBGR = test_data['imgBGR']
    sel = 3
    kp = kpts[sel]

    # Extract things to viz
    print('Extract patch, gradients, and orientations')
    patch = ptool.get_patch(imgBGR, kp)
    gradx, grady = ptool.patch_gradient(patch)
    gmag = ptool.patch_mag(gradx, grady)
    gori = ptool.patch_ori(gradx, grady)

    # Get orientation histogram
    print('Get orientation histogram')
    hist, centers = ptool.get_orientation_histogram(gori)

    # Get dominant direction in radians
    print('Find dominant orientation in histogram')
    kpts2 = ptool.find_kpts_direction(imgBGR, kpts)

    # Augment keypoint and plot rotated patch
    kp2 = kpts2[sel]
    wpatch, wkp = ptool.get_warped_patch(imgBGR, kp2)

    # --- Draw Results --- #
    print('Show patch, gradients, magintude, and orientation')
    gorimag = color_orimag(gori, gmag)
    nRow, nCol = (2, 3)
    df2.figure(fnum=1, pnum=(nRow, nCol, 1))
    df2.imshow(patch, pnum=(nRow, nCol, 1), fnum=1)
    df2.imshow(gradx, pnum=(nRow, nCol, 2), fnum=1)
    df2.imshow(grady, pnum=(nRow, nCol, 3), fnum=1)
    df2.imshow(gmag, pnum=(nRow, nCol, 4), fnum=1)
    df2.draw_vector_field(gradx, grady, pnum=(nRow, nCol, 5), fnum=1)
    df2.imshow(gorimag, pnum=(nRow, nCol, 6), fnum=1)

    print('Draw histogram with interpolation annotations')
    draw_hist_subbin_maxima(hist, centers, fnum=3, pnum=(1, 1, 1))

    df2.imshow(wpatch, fnum=4)

    viz.show_keypoints(imgBGR, kpts, sifts=desc, ell=True, eig=True, ori=True,
                       rect=True, ell_alpha=1)
    #pinteract.interact_keypoints(imgBGR, kpts2, desc, arrow=True, rect=True)
    exec(df2.present(wh=800))
