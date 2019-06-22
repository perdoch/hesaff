#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import cv2
import utool as ut


def test_pyheaff(img_fpath):
    r"""
    This show is interactive in this test!

    Args:
        img_fpath (str):

    CommandLine:
        python -m pyhesaff.tests.test_pyhesaff --test-test_pyheaff
        python -m pyhesaff.tests.test_pyhesaff --test-test_pyheaff --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from pyhesaff.tests.test_pyhesaff import *  # NOQA
        >>> img_fpath = ut.grab_test_imgpath('jeff.png')
        >>> test_pyheaff(img_fpath)
        >>> ut.show_if_requested()
    """
    import pyhesaff
    kpts, desc = pyhesaff.detect_feats(img_fpath)
    rchip = cv2.imread(img_fpath)
    if ut.show_was_requested():
        from plottool.interact_keypoints import ishow_keypoints
        ishow_keypoints(rchip, kpts, desc)
    return locals()


if __name__ == '__main__':
    """
    CommandLine:
        python -m pyhesaff.tests.test_pyhesaff
        python -m pyhesaff.tests.test_pyhesaff --allexamples
        python -m pyhesaff.tests.test_pyhesaff --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
