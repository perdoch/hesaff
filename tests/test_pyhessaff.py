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


@profile
def test_hesaff_kpts():
    test_data = load_test_data(short=True)
    img_fpath = test_data['img_fpath']
    # Make detector and read image
    hesaff_ptr = pyhesaff.new_hesaff(img_fpath)
    # Return the number of keypoints detected
    nKpts = pyhesaff.hesaff_lib.detect(hesaff_ptr)
    #print('[pyhesaff] detected: %r keypoints' % nKpts)
    # Allocate arrays
    kpts = np.empty((nKpts, 5), pyhesaff.kpts_dtype)
    desc = np.empty((nKpts, 128), pyhesaff.desc_dtype)
    # Populate arrays
    pyhesaff.hesaff_lib.exportArrays(hesaff_ptr, nKpts, kpts, desc)
    # TODO: Incorporate parameters
    # TODO: Scale Factor
    #hesaff_lib.extractDesc(hesaff_ptr, nKpts, kpts, desc2)
    #hesaff_lib.extractDesc(hesaff_ptr, nKpts, kpts, desc3)
    #print('[hesafflib] returned')
    return kpts, desc, img_fpath


def spaced_elements2(list_, n):
    if n is None:
        return np.arange(len(list_))
    if n == 0:
        return np.empty(0)
    indexes = np.arange(len(list_))
    stride = len(indexes) // n
    return indexes[0:-1:stride]


def spaced_elements(list_, n):
    if n is None:
        return 'list'
    indexes = np.arange(len(list_))
    stride = len(indexes) // n
    return list_[indexes[0:-1:stride]]


if __name__ == '__main__':
    multiprocessing.freeze_support()
    ensure_hotspotter()
    np.set_printoptions(threshold=5000, linewidth=5000, precision=3)
    kpts, desc, img_fpath = test_hesaff_kpts()
    rchip = io.imread(img_fpath)
    #viz.show_keypoints(rchip, kpts)
    interact.interact_keypoints(rchip, kpts, desc)
    exec(df2.present(override1=True))
