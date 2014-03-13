from __future__ import print_function, division
# Standard
import sys
from os.path import join, exists, realpath, expanduser
# Scientific
import numpy as np
import cv2
# Hotspotter
from hscom import fileio as io
# TPL
import pyhesaff


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


def load_test_data(short=False, n=0, **kwargs):
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
    detect_kwargs = {
        'scale_min': 20,
        'scale_max': 100
    }
    detect_kwargs.update(kwargs)
    kpts, desc = pyhesaff.detect_kpts(img_fpath, **detect_kwargs)
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


def spaced_elements(list_, n):
    if n is None:
        return 'list'
    indexes = np.arange(len(list_))
    stride = len(indexes) // n
    return list_[indexes[0:-1:stride]]

ensure_hotspotter()
