#!/usr/bin/env python
from __future__ import print_function, division
from hsviz import draw_func2 as df2
from vtool.drawtool import mpl_keypoint
from vtool.drawtool import mpl_sift  # NOQA
import vtool.keypoint as ktool  # NOQA
import numpy as np
import matplotlib as mpl  # NOQA
from itertools import product as iprod

np.tau = 2 * np.pi

RIGHT = (0 * np.tau / 4)
DOWN  = (1 * np.tau / 4)
LEFT  = (2 * np.tau / 4)
UP    = (3 * np.tau / 4)


def test_keypoint(ori=DOWN, skew=0):
    # Test Keypoint
    x, y = 0, 0
    a, c, d = 2, skew, 3
    kp = np.array([x, y, a, c, d, ori])

    # Test SIFT descriptor
    sift = np.zeros(128)
    sift[ 0: 8]   = 1
    sift[ 8:16]   = .5
    sift[16:24]   = .0
    sift[24:32]   = .5
    sift[32:40]   = .8
    sift[40:48]   = .8
    sift[48:56]   = .1
    sift[56:64]   = .2
    sift[64:72]   = .3
    sift[72:80]   = .4
    sift[80:88]   = .5
    sift[88:96]   = .6
    sift[96:104]  = .7
    sift[104:112] = .8
    sift[112:120] = .9
    sift[120:128] = 1
    sift = sift / np.sqrt((sift ** 2).sum())
    sift = np.round(sift * 255)

    kpts = np.array([kp])
    sifts = np.array([sift])
    return kpts, sifts


def square_axis(ax, s=4):
    ax.set_xlim(-s, s)
    ax.set_ylim(-s, s)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    df2.set_xticks([])
    df2.set_yticks([])


def test_shape(ori=0, skew=0, pnum=(1, 1, 1), fnum=1):
    df2.figure(fnum=fnum, pnum=pnum)
    kpts, sifts = test_keypoint(ori=ori, skew=skew)
    ax = df2.gca()
    square_axis(ax)
    mpl_keypoint.draw_keypoints(ax, kpts, sifts=sifts, ell_color=df2.ORANGE, ori=True,
                                rect_color=df2.DARK_RED,
                                ori_color=df2.DEEP_PINK, eig_color=df2.PINK,
                                rect=True, eig=True, bin_color=df2.RED,
                                arm1_color=df2.YELLOW, arm2_color=df2.BLACK)
    title = 'ori = %.2fpi' % (ori / np.pi)
    df2.set_title(title)
    df2.dark_background()
    return kpts, sifts


np.set_printoptions(precision=3)
px_ = 0

THETA1 = DOWN
THETA2 = (DOWN + DOWN + RIGHT) / 3
THETA3 = (DOWN + RIGHT) / 2
THETA4 = (DOWN + RIGHT + RIGHT) / 3
THETA5 = RIGHT

nRows = 3
nCols = 4


def pnum_(px=None):
    global px_
    if px is None:
        px_ += 1
        px = px_
    return (nRows, nCols, px)

MAX_ORI = np.tau
MIN_ORI = np.tau / 4
MAX_SKEW = 1.5
MIN_SWEW = 0


for row, col in iprod(xrange(nRows), xrange(nCols)):
    print((row, col))
    alpha = col / (nCols)
    beta  = row / (nRows)
    ori  = (MIN_ORI  * (1 - alpha)) + (MAX_ORI  * (alpha))
    skew = (MIN_SWEW * (1 - beta))  + (MAX_SKEW * (beta))
    kpts, sifts = test_shape(pnum=pnum_(), ori=ori, skew=skew)


#scale_factor = 1
#offset = (0, 0)
#(_xs, _ys, _as, _bs, _cs, _ds, _oris) = ktool.scaled_kpts(kpts, scale_factor, offset)

#aff_list = mpl_keypoint.get_aff_list(_xs, _ys, _as, _bs, _cs, _ds)
#aff = aff_list[0]

#ori = _oris[0]
#aff2 = mpl.transforms.Affine2D().rotate(-ori)

#print((aff + aff2).frozen())
#print((aff2 + aff).frozen())

#mpl_sift.draw_sift(ax, sift)
df2.update()

exec(df2.present(wh=(700, 700)))
