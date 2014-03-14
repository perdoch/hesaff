#!/usr/bin/env python
from __future__ import print_function, division
from hsviz import draw_func2 as df2
from vtool.drawtool import mpl_keypoint
from vtool.drawtool import mpl_sift  # NOQA
import vtool.keypoint as ktool  # NOQA
import numpy as np
import matplotlib as mpl  # NOQA
from itertools import product as iprod
import vtool  # NOQA

np.tau = 2 * np.pi

# Hack these directions to be relative to gravity
RIGHT = ((0 * np.tau / 4) - ktool.GRAVITY_THETA) % np.tau
DOWN  = ((1 * np.tau / 4) - ktool.GRAVITY_THETA) % np.tau
LEFT  = ((2 * np.tau / 4) - ktool.GRAVITY_THETA) % np.tau
UP    = ((3 * np.tau / 4) - ktool.GRAVITY_THETA) % np.tau


def test_keypoint(xscale=1, yscale=1, ori=DOWN, skew=0):
    # Test Keypoint
    x, y = 0, 0
    iv11, iv21, iv22 = xscale, skew, yscale
    kp = np.array([x, y, iv11, iv21, iv22, ori])

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


def square_axis(ax, s=3):
    ax.set_xlim(-s, s)
    ax.set_ylim(-s, s)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    df2.set_xticks([])
    df2.set_yticks([])


def test_shape(ori=0, skew=0, xscale=1, yscale=1, pnum=(1, 1, 1), fnum=1):
    df2.figure(fnum=fnum, pnum=pnum)
    kpts, sifts = test_keypoint(ori=ori, skew=skew, xscale=xscale, yscale=yscale)
    ax = df2.gca()
    square_axis(ax)
    mpl_keypoint.draw_keypoints(ax, kpts, sifts=sifts, ell_color=df2.ORANGE, ori=True,
                                rect_color=df2.DARK_RED,
                                ori_color=df2.DEEP_PINK, eig_color=df2.PINK,
                                rect=True, eig=True, bin_color=df2.RED,
                                arm1_color=df2.YELLOW, arm2_color=df2.BLACK)
    title = 'xyscale=(%.1f, %.1f),\n skew=%.1f, ori=%.2ftau' % (xscale, yscale, skew, ori / np.tau)
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

nRows = 5
nCols = 3


def pnum_(px=None):
    global px_
    if px is None:
        px_ += 1
        px = px_
    return (nRows, nCols, px)

MAX_ORI = np.tau
MIN_ORI = np.tau / 4 - .01

MIN_X = .5
MAX_X = 2

MAX_SKEW = 0
MIN_SWEW = 1

MIN_Y = .5
MAX_Y = 2

kp_list = []


for row, col in iprod(xrange(nRows), xrange(nCols)):
    #print((row, col))
    alpha = col / (nCols)
    beta  = row / (nRows)
    xsca = (MIN_X    * (1 - alpha)) + (MAX_X    * (alpha))
    ori  = (MIN_ORI  * (1 - alpha)) + (MAX_ORI  * (alpha))
    skew = (MIN_SWEW * (1 - beta))  + (MAX_SKEW * (beta))
    ysca = (MIN_Y    * (1 - beta))  + (MAX_Y    * (beta))

    kpts, sifts = test_shape(pnum=pnum_(),
                             ori=ori,
                             skew=skew,
                             xscale=xsca,
                             yscale=ysca)
    print('+----')
    kp_list.append(kpts[0])
    S_list = ktool.orthogonal_scales(kpts=kpts)
    #print('xscale=%r yscale=%r, skew=%r' % (xsca, ysca, skew))
    #print(S_list)

    import vtool.linalg as ltool
    #kpts = np.vstack(kp_list)
    invV_mats = ktool.get_invV_mats(kpts, ashomog=False)
    USV_list = [ltool.svd(invV) for invV in invV_mats]

    from hscom import util
    str_list = []
    for USV in USV_list:
        U, s, V = USV
        S = np.diag(s)

        S2 = S[::-1, ::-1]
        U2 = U[::-1, ::-1].T
        V2 = V[::-1, ::-1].T

        A = U.dot(S).dot(V)
        A2 = U2.dot(S2).dot(V2)
        A3 = S2.dot(V2)

        (np.diag(s).dot(V) ** 2).sum(0)

        str_ = util.horiz_string([U, ' * ', S, ' * ', V, ' = ', A])
        str2_ = util.horiz_string([V2, ' * ', S2, ' * ', U2, ' = ', A2])
        str_list.append(str_)
        print(util.horiz_string(('Input: ', invV)))
        print('')
        print(str_)
        print('')
        print(A3)
        #print(str2_)
    print('---')

#scale_factor = 1
#offset = (0, 0)
#(_xs, _ys, _iv11s, _iv12s, _iv21s, _iv22s, _oris) = ktool.scaled_kpts(kpts, scale_factor, offset)

#invVR_aff2Ds = mpl_keypoint.get_invV_aff2Ds(_xs, _ys, _iv11s, _iv12s, _iv21s, _iv22s)
#aff = invVR_aff2Ds[0]

#ori = _oris[0]
#aff2 = mpl.transforms.Affine2D().rotate(-ori)

#print((aff + aff2).frozen())
#print((aff2 + aff).frozen())

#mpl_sift.draw_sift(ax, sift)
#df2.update()

exec(df2.present(wh=(700, 700)))
