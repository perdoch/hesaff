#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import utool as ut


def simple_iterative_test():
    r"""
    CommandLine:
        python -m pyhesaff.tests.test_pyhesaff_simple_iterative --test-simple_iterative_test
        python -m pyhesaff.tests.test_pyhesaff_simple_iterative --test-simple_iterative_test --show

    Example:
        >>> # GUI_DOCTEST
        >>> from pyhesaff.tests.test_pyhesaff_simple_iterative import *  # NOQA
        >>> result = simple_iterative_test()
        >>> print(result)
        >>> ut.show_if_requested()
    """
    import pyhesaff
    fpath_list = [
        ut.grab_test_imgpath('lena.png'),
        ut.grab_test_imgpath('carl.jpg'),
        ut.grab_test_imgpath('grace.jpg'),
        ut.grab_test_imgpath('ada.jpg'),
    ]
    kpts_list = []

    for img_fpath in fpath_list:
        kpts, vecs = pyhesaff.detect_feats(img_fpath)
        print('img_fpath=%r' % img_fpath)
        print('kpts=%s' % (ut.truncate_str(repr(kpts)),))
        print('vecs=%s' % (ut.truncate_str(repr(vecs)),))
        assert len(kpts) == len(vecs)
        assert len(kpts) > 0, 'no keypoints were detected!'
        kpts_list.append(kpts)

    if ut.show_was_requested():
        import matplotlib as mpl
        from matplotlib import pyplot as plt
        fig = plt.figure()
        for i, fpath, kpts in enumerate(zip(fpath_list, kpts_list), start=1):
            ax = fig.add_subplot(2, 2, i)
            img = mpl.image.imread(fpath)
            plt.imshow(img)
            _xs, _ys = kpts.T[0:2]
            ax.plot(_xs, _ys, 'ro', alpha=.5)


if __name__ == '__main__':
    """
    CommandLine:
        python -m pyhesaff.tests.test_pyhesaff_simple_iterative simple_iterative_test
        python -m pyhesaff.tests.test_pyhesaff_simple_iterative --allexamples
        python -m pyhesaff.tests.test_pyhesaff_simple_iterative --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
