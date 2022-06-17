#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import numpy as np
import ubelt as ub


def TEST_ptool_find_kpts_direction(imgBGR, kpts):
    import vtool.patch as ptool
    def hrint(*x):
        print(ub.hzcat(x))
    print('[rotinvar] +---')
    print('[rotinvar] | 3) Find dominant orientation in histogram')
    hrint('[rotinvar] |  * kpts.shape = ', (kpts.shape,))
    hrint('[rotinvar] |  * kpts = ', kpts)
    kpts2 = ptool.find_kpts_direction(imgBGR, kpts)
    hrint('[rotinvar] |  * kpts2.shape = ', (kpts.shape,))
    hrint('[rotinvar] |  * kpts2 = ', kpts2)
    print('[rotinvar] L___')
    return kpts2


def TEST_figure1(wpatch, gradx, grady, gmag, gori, hist, centers, fnum=1):
    print('[rotinvar] 4) Draw histogram with interpolation annotations')
    import vtool.patch as ptool
    import plottool
    from plottool import draw_func2 as df2
    gorimag = plottool.color_orimag(gori, gmag)
    nRow, nCol = (2, 7)

    df2.figure(fnum=fnum, pnum=(nRow, 1, nRow), doclf=True, docla=True)
    plottool.draw_hist_subbin_maxima(hist, centers)
    df2.set_xlabel('grad orientation (radians)')
    df2.set_ylabel('grad magnitude')
    df2.set_title('dominant orientations')

    print('[rotinvar] 5) Show patch, gradients, magintude, and orientation')
    df2.imshow(wpatch,    pnum=(nRow, nCol, 1), fnum=fnum, title='patch')
    df2.draw_vector_field(gradx, grady, pnum=(nRow, nCol, 2), fnum=fnum, title='gori (vec)')
    df2.imshow(gorimag, pnum=(nRow, nCol, 3), fnum=fnum, title='gori (col)')
    df2.imshow(np.abs(gradx),   pnum=(nRow, nCol, 4), fnum=fnum, title='gradx')
    df2.imshow(np.abs(grady),   pnum=(nRow, nCol, 5), fnum=fnum, title='grady')
    df2.imshow(gmag,    pnum=(nRow, nCol, 6), fnum=fnum, title='gmag')

    gpatch = ptool.gaussian_patch(shape=gori.shape)
    df2.imshow(gpatch * 255, pnum=(nRow, nCol, 7), fnum=fnum, title='gauss weights', cmap_='hot')
    #gpatch3 = np.dstack((gpatch, gpatch, gpatch))
    #df2.draw_vector_field(gradx * gpatch, grady * gpatch, pnum=(nRow, nCol, 8), fnum=fnum, title='gori (vec)')
    #df2.imshow(gorimag * gpatch3, pnum=(nRow, nCol, 9), fnum=fnum, title='gori (col)')
    #df2.imshow(gradx * gpatch,   pnum=(nRow, nCol, 10), fnum=fnum, title='gradx')
    #df2.imshow(grady * gpatch,   pnum=(nRow, nCol, 11), fnum=fnum, title='grady')
    #df2.imshow(gmag * gpatch,    pnum=(nRow, nCol, 12), fnum=fnum, title='gmag')
    return locals()


def TEST_figure2(imgBGR, kpts, desc, sel, fnum=2):
    from plottool import draw_func2 as df2
    from plottool.viz_keypoints import _annotate_kpts, show_keypoints
    from plottool.viz_featrow import draw_feat_row
    #df2.imshow(wpatch, fnum=2)
    sift = desc[sel]
    viz_kwargs = dict(ell=True, eig=False,
                      rect=True, ori_color=df2.DEEP_PINK, ell_alpha=1, fnum=fnum, pnum=(2, 1, 1))
    show_keypoints(imgBGR, kpts, sifts=None, sel_fx=sel, ori=False, **viz_kwargs)
    _annotate_kpts(kpts, sel, ori=True, **viz_kwargs)
    draw_feat_row(imgBGR, sel, kpts[sel], sift, fnum=fnum, nRows=2, nCols=3, px=3)


def TEST_keypoint(imgBGR, img_fpath, kpts, desc, sel, fnum=1, figtitle=''):
    from plottool import draw_func2 as df2
    import vtool.patch as ptool
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
    gori_weights = ptool.gaussian_weight_patch(gmag)

    # Get orientation histogram
    print('[rotinvar] 2) Get orientation histogram')
    hist, centers = ptool.get_orientation_histogram(gori, gori_weights)

    #----------------------#
    # --- Draw Results --- #
    #----------------------#
    f1_loc = TEST_figure1(wpatch, gradx, grady, gmag, gori, hist, centers, fnum=fnum)
    df2.set_figtitle(figtitle + 'Dominant Orienation Extraction')

    TEST_figure2(imgBGR, kpts, desc, sel, fnum=fnum + 1)
    df2.set_figtitle(figtitle)
    # TEST_figure2(imgBGR, kpts2, Desc2, sel, fnum=fnum + 2)
    # df2.set_figtitle('Rotation Invariant')

    #df2.draw_keypoint_gradient_orientations(imgBGR, kp=kpts2[sel],
    #                                        sift=desc[sel], mode='vec',
    #                                        fnum=4)

    #df2.draw_vector_field(gradx, grady, pnum=(1, 1, 1), fnum=4)
    #df2.draw_kpts2(np.array([wkp]), sifts=desc[sel:sel + 1], ori=True)
    return locals()


def test_cpp_rotinvar_main():
    r"""
    CommandLine:
        python -m pyhesaff.tests.test_cpp_rotation_invariance --test-test_cpp_rotinvar_main
        python -m pyhesaff.tests.test_cpp_rotation_invariance --test-test_cpp_rotinvar_main --show

    # Example:
    #     >>> # DISABLE_DOCTEST
    #     >>> from pyhesaff.tests.test_cpp_rotation_invariance import *  # NOQA
    #     >>> # build test data
    #     >>> # execute function
    #     >>> result = test_cpp_rotinvar_main()
    #     >>> # verify results
    #     >>> print(result)
    """
    import pytest
    pytest.skip('Broken in CI')
    # TODO; take visualization out of this test by default
    from pyhesaff.tests import pyhestest
    import pyhesaff
    # Read data
    print('[rotinvar] loading test data')
    img_fpath = pyhestest.get_test_image()
    [kpts1], [desc1] = pyhesaff.detect_feats_list([img_fpath], rotation_invariance=False)
    [kpts2], [desc2] = pyhesaff.detect_feats_list([img_fpath], rotation_invariance=True)
    np.set_printoptions(threshold=5000, linewidth=5000, precision=8, suppress=True)

    print('kpts1.shape = %r' % (kpts1.shape,))
    print('kpts2.shape = %r' % (kpts2.shape,))

    print('desc1.shape = %r' % (desc1.shape,))
    print('desc2.shape = %r' % (desc2.shape,))

    print('\n----\n'.join([str(k1) + '\n' + str(k2) for k1, k2 in zip(kpts1[0:10], kpts2[0:10])]))

    n = 4
    #clip = min(len(kpts1), n)

    # HACK FIXME
    fxs = np.array(pyhestest.spaced_elements2(kpts2, n).tolist()[0:3])

    print('fxs=%r' % fxs)
    kpts1 = kpts1[fxs]
    kpts2 = kpts2[fxs]
    desc1 = desc1[fxs]
    desc2 = desc2[fxs]

    print('\n----\n'.join([str(k1) + '\n' + str(k2) for k1, k2 in zip(kpts1, kpts2)]))

    imgBGR = pyhestest.cv2.imread(img_fpath)
    sel = min(len(kpts1) - 1, 3)

    TEST_keypoint(imgBGR, img_fpath, kpts1, desc1, sel, fnum=1, figtitle='Downward Rotation')
    TEST_keypoint(imgBGR, img_fpath, kpts2, desc2, sel, fnum=9001, figtitle='Adapted Rotation')

    #locals_ = TEST_keypoint(imgBGR, img_fpath, kpts1, desc1, sel)

    #pinteract.interact_keypoints(imgBGR, kpts2, desc, arrow=True, rect=True)
    if ub.argflag('--show'):
        from plottool import draw_func2 as df2
        exec(df2.present())


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/pyhesaff/tests/test_cpp_rotation_invariance.py
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
