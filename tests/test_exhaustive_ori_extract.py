#!/usr/bin/env python
# ALREADY PORTED FIXME DELETE
from __future__ import absolute_import, division, print_function
import sys
import __sysreq__  # NOQA
sys.argv.append('--nologgging')
#from hscom import __common__
from os.path import realpath
#(print, print_, print_on, print_off,
 #rrr, profile, printDBG) = __common__.init(__name__, '[util]', DEBUG=False)
# Standard
# Scientific
import numpy as np
# Hotspotter
from plottool import draw_func2 as df2
from plottool.viz_keypoints import show_keypoints
#from hsdev import dev_consistency
# TPL
import pyhesaff
# VTool
import vtool  # NOQA
import vtool.image as gtool
import vtool.keypoint as ktool
from vtool.tests import grabdata

if __name__ == '__main__':
    np.set_printoptions(threshold=5000, linewidth=5000, precision=3)
    # Read data
    print('[rotinvar] loading test data')

    img_fpath = grabdata.get_testimg_path('jeff.png')
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
    def double_detect(img_fpath, **kw):
        # Checks to make sure computation is determinsitc
        _kpts, _desc = pyhesaff.detect_kpts(img_fpath, **kw)
        kpts_, desc_ = pyhesaff.detect_kpts(img_fpath, **kw)
        assert np.all(kpts_ == _kpts)
        assert np.all(desc_ == _desc)
        print('double detect ok')
        return kpts_, desc_

    # Remove skew and anisotropic scaling
    def force_isotropic(kpts):
        kpts_ = kpts.copy()
        kpts_[:, ktool.SKEW_DIM] = 0
        kpts_[:, ktool.SCAX_DIM] = kpts_[:, ktool.SCAY_DIM]
        desc_ = pyhesaff.extract_desc(img_fpath, kpts_)
        return kpts_, desc_

    def force_ori(kpts, ori):
        kpts_ = kpts.copy()
        kpts_[:, ktool.ORI_DIM] = ori
        desc_ = pyhesaff.extract_desc(img_fpath, kpts_)
        return kpts_, desc_

    def shift_kpts(kpts, x, y):
        kpts_ = kpts.copy()
        kpts_[:, ktool.XDIM] += x
        kpts_[:, ktool.YDIM] += y
        desc_ = pyhesaff.extract_desc(img_fpath, kpts_)
        return kpts_, desc_

    # --- Experiment ---
    kpts0, desc0 = double_detect(img_fpath, **detect_kw0)
    kpts1, desc1 = double_detect(img_fpath, **detect_kw1)
    kpts2, desc2 = double_detect(img_fpath, **detect_kw2)
    #
    kpts3, desc3 = double_detect(img_fpath, **detect_kw3)
    kpts4, desc4 = force_isotropic(kpts3)
    kpts5, desc5 = force_ori(kpts3, 1.45)
    kpts6, desc6 = shift_kpts(kpts5, -60, -50)
    kpts7, desc7 = force_ori(kpts6, 0)
    kpts8, desc8 = force_ori(kpts7, 2.40)
    kpts9, desc9 = force_ori(kpts8, 5.40)
    kpts10, desc10 = force_ori(kpts9, 10.40)

    # --- Print ---

    # ---- Draw ----
    nRow, nCol = 1, 2
    df2.figure(fnum=2, doclf=True, docla=True)
    df2.figure(fnum=1, doclf=True, docla=True)

    def show_kpts_(fnum, pnum, kpts, desc, title):
        print('--------')
        print('show_kpts: %r.%r' % (fnum, pnum))
        print('kpts  = %r' % (kpts,))
        print('scales = %r' % ktool.get_scales(kpts))
        # FIXME: this exists in ibeis. move to vtool
        #dev_consistency.check_desc(desc3)

        show_keypoints(imgL, kpts, sifts=desc, pnum=pnum, rect=True,
                       ori=True, fnum=fnum, title=title, ell_alpha=1)

    show_kpts_(1, (nRow, nCol, 1), kpts3, desc3, 'kpts3: original')
    show_kpts_(1, (nRow, nCol, 2), kpts4, desc4, 'kpts4: isotropic + redetect')
    show_kpts_(2, (2, 3, 1), kpts5, desc5, 'kpts5: force_ori + redetect')
    show_kpts_(2, (2, 3, 2), kpts6, desc6, 'kpts6: shift')
    show_kpts_(2, (2, 3, 3), kpts7, desc7, 'kpts7: shift + reorient')
    show_kpts_(2, (2, 3, 4), kpts8, desc8, 'kpts8: shift + reorient')
    show_kpts_(2, (2, 3, 5), kpts9, desc9, 'kpts9: reorient')
    show_kpts_(2, (2, 3, 6), kpts10, desc10, 'kpts10: reorient')
    df2.iup()

    #pinteract.interact_keypoints(imgBGR, kpts2, desc, arrow=True, rect=True)
    exec(df2.present(wh=800))
