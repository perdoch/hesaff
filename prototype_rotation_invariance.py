#!/usr/bin/env python
from __future__ import print_function, division
#from hscom import __common__
#(print, print_, print_on, print_off,
 #rrr, profile, printDBG) = __common__.init(__name__, '[util]', DEBUG=False)
# Standard
import multiprocessing
# Scientific
import numpy as np
# Hotspotter
from hscom import util  # NOQA
from hsviz import draw_func2 as df2
from hsviz import viz
# TPL
import pyhestest
import pyhesaff
# VTool
import vtool.patch as ptool
import vtool.drawtool as dtool


def TEST_ptool_find_kpts_direction(imgBGR, kpts):
    hrint = util.horiz_print
    print('[rotinvar] +---')
    print('[rotinvar] | 3) Find dominant orientation in histogram')
    hrint('[rotinvar] |  * kpts.shape = ', (kpts.shape,))
    hrint('[rotinvar] |  * kpts = ', kpts)
    kpts2 = ptool.find_kpts_direction(imgBGR, kpts)
    hrint('[rotinvar] |  * kpts2.shape = ', (kpts.shape,))
    hrint('[rotinvar] |  * kpts2 = ', kpts2)
    print('[rotinvar] L___')
    return kpts2


def TEST_figure1(wpatch, gradx, grady, gmag, gori, hist, centers):
    print('[rotinvar] 4) Draw histogram with interpolation annotations')
    fnum = 1
    gorimag = dtool.color_orimag(gori, gmag)

    df2.figure(fnum=1, pnum=(2, 1, 2), doclf=True, docla=True)
    dtool.draw_hist_subbin_maxima(hist, centers)
    df2.set_xlabel('grad orientation (radians)')
    df2.set_ylabel('grad magnitude')
    df2.set_title('dominant orientations')

    print('[rotinvar] 5) Show patch, gradients, magintude, and orientation')
    nRow, nCol = (2, 6)
    df2.imshow(wpatch,    pnum=(nRow, nCol, 1), fnum=fnum, title='patch')
    df2.draw_vector_field(gradx, grady, pnum=(nRow, nCol, 2), fnum=fnum)
    df2.set_title('gori (vec)')
    df2.imshow(gorimag, pnum=(nRow, nCol, 3), fnum=fnum, title='gori (col)')
    df2.imshow(gradx,   pnum=(nRow, nCol, 4), fnum=fnum, title='gradx')
    df2.imshow(grady,   pnum=(nRow, nCol, 5), fnum=fnum, title='grady')
    df2.imshow(gmag,    pnum=(nRow, nCol, 6), fnum=fnum, title='gmag')


def TEST_figure2(imgBGR, kpts, desc, sel, fnum=2):
    #df2.imshow(wpatch, fnum=2)
    sift = desc[sel]
    viz_kwargs = dict(ell=True, eig=False,
                      rect=True, ori_color=df2.DEEP_PINK, ell_alpha=1, fnum=fnum, pnum=(2, 1, 1))
    viz.show_keypoints(imgBGR, kpts, sifts=None, sel_fx=sel, ori=False, **viz_kwargs)
    viz._annotate_kpts(kpts, sel, True, False, ori=True, **viz_kwargs)
    viz.draw_feat_row(imgBGR, sel, kpts[sel], sift, fnum=fnum, nRows=2, nCols=3, px=3)


def TEST_keypoint(imgBGR, img_fpath, kpts, desc, sel):
    #----------------------#
    # --- Extract Data --- #
    #----------------------#
    kp = kpts[sel]
    # Extract patches, gradients, and orientations
    print('[rotinvar] 1) Extract patch, gradients, and orientations')
    wpatch, wkp  = ptool.get_warped_patch(imgBGR, kp, gray=True)
    gradx, grady = ptool.patch_gradient(wpatch)
    gmag         = ptool.patch_mag(gradx, grady)
    gori         = ptool.patch_ori(gradx, grady)

    # Get orientation histogram
    print('[rotinvar] 2) Get orientation histogram')
    hist, centers = ptool.get_orientation_histogram(gori)

    # Get dominant direction in radians
    kpts2 = TEST_ptool_find_kpts_direction(imgBGR, kpts)
    kpts2, desc2 = pyhesaff.adapt_rotation(img_fpath, kpts)

    #----------------------#
    # --- Draw Results --- #
    #----------------------#
    TEST_figure1(wpatch, gradx, grady, gmag, gori, hist, centers)
    df2.set_figtitle('Dominant Orienation Extraction')

    TEST_figure2(imgBGR, kpts, desc, sel, fnum=2)
    df2.set_figtitle('Gravity Vector')
    TEST_figure2(imgBGR, kpts2, desc2, sel, fnum=3)
    df2.set_figtitle('Rotation Invariant')

    #df2.draw_vector_field(gradx, grady, pnum=(1, 1, 1), fnum=4)
    #df2.draw_kpts2(np.array([wkp]), sifts=desc[sel:sel + 1], ori=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    np.set_printoptions(threshold=5000, linewidth=5000, precision=3)
    # Read data
    print('[rotinvar] loading test data')
    test_data = pyhestest.load_test_data(short=True, n=3)
    img_fpath = test_data['img_fpath']
    kpts = test_data['kpts']
    desc = test_data['desc']
    imgBGR = test_data['imgBGR']
    sel = 3

    TEST_keypoint(imgBGR, img_fpath, kpts, desc, sel)

    #pinteract.interact_keypoints(imgBGR, kpts2, desc, arrow=True, rect=True)
    exec(df2.present(wh=800))
