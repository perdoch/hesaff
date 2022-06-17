#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import numpy as np
import utool as ut


def double_detect(img_fpath, **kw):
    import pyhesaff
    # Checks to make sure computation is determinsitc
    _kpts, _vecs = pyhesaff.detect_feats(img_fpath, **kw)
    kpts_, vecs_ = pyhesaff.detect_feats(img_fpath, **kw)
    assert np.all(kpts_ == _kpts)
    assert np.all(vecs_ == _vecs)
    print('double detect ok')
    return kpts_, vecs_


def test_ori_extract_main():
    """
    CommandLine:
        python -m pyhesaff.tests.test_exhaustive_ori_extract --test-test_ori_extract_main
        python -m pyhesaff.tests.test_exhaustive_ori_extract --test-test_ori_extract_main --show

    # Example:
    #     >>> # GUI_DOCTEST
    #     >>> from pyhesaff.tests.test_exhaustive_ori_extract import *  # NOQA
    #     >>> test_ori_extract_main()
    #     >>> ut.show_if_requested()
    """
    import pytest
    pytest.skip('Broken CI')

    import pyhesaff
    from plottool import draw_func2 as df2
    from plottool.viz_keypoints import show_keypoints
    import vtool  # NOQA
    import vtool.image as gtool
    import vtool.keypoint as ktool
    np.set_printoptions(threshold=5000, linewidth=5000, precision=3)
    # Read data
    print('[rotinvar] loading test data')

    img_fpath = ut.grab_test_imgpath('jeff.png')
    imgL = gtool.cvt_BGR2L(gtool.imread(img_fpath))
    detect_kw0 = {
    }
    detect_kw1 = {
        'scale_min': 20,
        'scale_max': 100
    }
    detect_kw2 = {
        'scale_min': 40,
        'scale_max': 60
    }
    detect_kw3 = {
        'scale_min': 45,
        'scale_max': 49
    }
    # Remove skew and anisotropic scaling
    def force_isotropic(kpts):
        kpts_ = kpts.copy()
        kpts_[:, ktool.SKEW_DIM] = 0
        kpts_[:, ktool.SCAX_DIM] = kpts_[:, ktool.SCAY_DIM]
        vecs_ = pyhesaff.extract_vecs(img_fpath, kpts_)
        return kpts_, vecs_

    def force_ori(kpts, ori):
        kpts_ = kpts.copy()
        kpts_[:, ktool.ORI_DIM] = ori
        vecs_ = pyhesaff.extract_vecs(img_fpath, kpts_)
        return kpts_, vecs_

    def shift_kpts(kpts, x, y):
        kpts_ = kpts.copy()
        kpts_[:, ktool.XDIM] += x
        kpts_[:, ktool.YDIM] += y
        vecs_ = pyhesaff.extract_vecs(img_fpath, kpts_)
        return kpts_, vecs_

    # --- Experiment ---
    kpts0, vecs0 = double_detect(img_fpath, **detect_kw0)
    kpts1, vecs1 = double_detect(img_fpath, **detect_kw1)
    kpts2, vecs2 = double_detect(img_fpath, **detect_kw2)
    #
    kpts3, vecs3 = double_detect(img_fpath, **detect_kw3)
    kpts4, vecs4 = force_isotropic(kpts3)
    kpts5, vecs5 = force_ori(kpts3, 1.45)
    kpts6, vecs6 = shift_kpts(kpts5, -60, -50)
    kpts7, vecs7 = force_ori(kpts6, 0)
    kpts8, vecs8 = force_ori(kpts7, 2.40)
    kpts9, vecs9 = force_ori(kpts8, 5.40)
    kpts10, vecs10 = force_ori(kpts9, 10.40)

    # --- Print ---

    # ---- Draw ----
    nRow, nCol = 1, 2
    df2.figure(fnum=2, doclf=True, docla=True)
    df2.figure(fnum=1, doclf=True, docla=True)

    def show_kpts_(fnum, pnum, kpts, vecs, title):
        print('--------')
        print('show_kpts: %r.%r' % (fnum, pnum))
        print('kpts  = %r' % (kpts,))
        print('scales = %r' % ktool.get_scales(kpts))
        # FIXME: this exists in ibeis. move to vtool
        #dev_consistency.check_vecs(vecs3)

        show_keypoints(imgL, kpts, sifts=vecs, pnum=pnum, rect=True,
                       ori=True, fnum=fnum, title=title, ell_alpha=1)

    show_kpts_(1, (nRow, nCol, 1), kpts3, vecs3, 'kpts3: original')
    show_kpts_(1, (nRow, nCol, 2), kpts4, vecs4, 'kpts4: isotropic + redetect')
    show_kpts_(2, (2, 3, 1), kpts5, vecs5, 'kpts5: force_ori + redetect')
    show_kpts_(2, (2, 3, 2), kpts6, vecs6, 'kpts6: shift')
    show_kpts_(2, (2, 3, 3), kpts7, vecs7, 'kpts7: shift + reorient')
    show_kpts_(2, (2, 3, 4), kpts8, vecs8, 'kpts8: shift + reorient')
    show_kpts_(2, (2, 3, 5), kpts9, vecs9, 'kpts9: reorient')
    show_kpts_(2, (2, 3, 6), kpts10, vecs10, 'kpts10: reorient')
    #df2.iup()

    #pinteract.interact_keypoints(imgBGR, kpts2, vecs, arrow=True, rect=True)
    #exec(df2.present(wh=800))


if __name__ == '__main__':
    """
    CommandLine:
        python -m pyhesaff.tests.test_exhaustive_ori_extract
        python -m pyhesaff.tests.test_exhaustive_ori_extract --allexamples
        python -m pyhesaff.tests.test_exhaustive_ori_extract --allexamples --noface --nosrc
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
