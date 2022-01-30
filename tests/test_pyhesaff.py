#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import cv2
import utool as ut


def test_pyheaff(img_fpath):
    r"""
    This show is interactive in this test!

    Args:
        img_fpath (str):

    Example:
        >>> # ENABLE_DOCTEST
        >>> from pyhesaff.tests.test_pyhesaff import *  # NOQA
        >>> img_fpath = ut.grab_test_imgpath('jeff.png')
        >>> test_pyheaff(img_fpath)
    """
    import pyhesaff
    kpts, desc = pyhesaff.detect_feats(img_fpath)
    rchip = cv2.imread(img_fpath)
    if ut.show_was_requested():
        from plottool.interact_keypoints import ishow_keypoints
        ishow_keypoints(rchip, kpts, desc)
    return locals()


if __name__ == '__main__':
    import xdoctest
    xdoctest.doctest_module(__file__)
