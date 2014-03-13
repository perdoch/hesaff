#!/usr/bin/env python
from __future__ import print_function, division
# Standard
import multiprocessing
# Scientific
import numpy as np
# Hotspotter
from hscom import util  # NOQA
from hsviz import draw_func2 as df2
from hsviz import interact
from hscom import fileio as io
# TPL
import pyhesaff
import pyhestest


def test_hesaff_kpts():
    test_data = pyhestest.load_test_data(short=True)
    img_fpath = test_data['img_fpath']
    # Make detector and read image
    hesaff_ptr = pyhesaff.new_hesaff(img_fpath)
    # Return the number of keypoints detected
    nKpts = pyhesaff.hesaff_lib.detect(hesaff_ptr)
    #print('[pyhesaff] detected: %r keypoints' % nKpts)
    # Allocate arrays
    kpts, desc = pyhesaff.allocate_kpts(nKpts)
    # Populate arrays
    pyhesaff.hesaff_lib.exportArrays(hesaff_ptr, nKpts, kpts, desc)
    # TODO: Incorporate parameters
    # TODO: Scale Factor
    #hesaff_lib.extractDesc(hesaff_ptr, nKpts, kpts, desc2)
    #hesaff_lib.extractDesc(hesaff_ptr, nKpts, kpts, desc3)
    #print('[hesafflib] returned')
    return kpts, desc, img_fpath


if __name__ == '__main__':
    multiprocessing.freeze_support()
    np.set_printoptions(threshold=5000, linewidth=5000, precision=3)
    kpts, desc, img_fpath = test_hesaff_kpts()
    rchip = io.imread(img_fpath)
    #viz.show_keypoints(rchip, kpts)
    interact.interact_keypoints(rchip, kpts, desc)
    exec(df2.present(override1=True))
