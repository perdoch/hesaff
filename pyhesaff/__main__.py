#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals


def detect_feats_main():
    import pyhesaff
    from pyhesaff import grab_test_imgpath
    from pyhesaff import argparse_hesaff_params
    import cv2
    import ubelt as ub
    img_fpath = grab_test_imgpath(ub.argval('--fname', default='astro.png'))
    kwargs = argparse_hesaff_params()
    print('kwargs = %r' % (kwargs,))
    (kpts, vecs) = pyhesaff.detect_feats(img_fpath, **kwargs)
    # Show keypoints
    # xdoctest: +REQUIRES(--show)
    imgBGR = cv2.imread(img_fpath)
    # take a random stample
    default_showkw = dict(ori=False, ell=True, ell_linewidth=2,
                          ell_alpha=.4, ell_color='distinct')
    print('default_showkw = %r' % (default_showkw,))
    import utool as ut
    showkw = ut.argparse_dict(default_showkw)
    import plottool as pt
    pt.interact_keypoints.ishow_keypoints(imgBGR, kpts, vecs, **showkw)
    pt.show_if_requested()


def main():
    import pyhesaff
    print('WELCOME TO THE MAIN FOR pyhesaff = {!r}'.format(pyhesaff))
    print('TODO: Write a program that takes and image and outputs keypoints / descriptors')

if __name__ == '__main__':
    main()
