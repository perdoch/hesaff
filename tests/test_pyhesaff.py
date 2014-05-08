#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import pyhestest  # NOQA
# Scientific
import cv2
# Tools
from plottool import draw_func2 as df2
from plottool.interact_keypoints import ishow_keypoints
from vtool.tests import grabdata
# Pyhesaff
import pyhesaff


def test_pyheaff(img_fpath):
    kpts, desc = pyhesaff.detect_kpts(img_fpath)
    rchip = cv2.imread(img_fpath)
    ishow_keypoints(rchip, kpts, desc)
    return locals()


if __name__ == '__main__':
    img_fpath = grabdata.get_testimg_path('jeff.png')
    test_pyheaff(img_fpath)
    exec(df2.present())
