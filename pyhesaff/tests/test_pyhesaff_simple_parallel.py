#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
import matplotlib as mpl
import numpy as np
from six.moves import zip
from matplotlib import pyplot as plt
import utool as ut
import time

#sample_data_fpath = join(dirname(mpl.__file__), 'mpl-data', 'sample_data')

#mpl.cbook.get_sample_data('grace_hopper.png', False)
#mpl.cbook.get_sample_data('lena.png', False)

#mpl.cbook.get_sample_data('Minduka_Present_Blue_Pack.png', False)
#mpl.cbook.get_sample_data('logo2.png', False)
#mpl.cbook.get_sample_data('grace_hopper.jpg', False)
#mpl.cbook.get_sample_data('ada.png', False)


def test_simple_parallel():
    r"""
    CommandLine:
        python -m pyhesaff.tests.test_pyhesaff_simple_parallel --test-test_simple_parallel

    Example:
        >>> # ENABLE_DOCTEST
        >>> from pyhesaff.tests.test_pyhesaff_simple_parallel import *  # NOQA
        >>> test_simple_parallel()
    """
    import pyhesaff
    lena_fpath  = mpl.cbook.get_sample_data('lena.jpg', False)
    logo_fpath  = mpl.cbook.get_sample_data('logo2.png', False)
    grace_fpath = mpl.cbook.get_sample_data('grace_hopper.jpg', False)
    ada_fpath   = mpl.cbook.get_sample_data('ada.png', False)

    img_fpaths = [lena_fpath, logo_fpath, grace_fpath, ada_fpath] * 2

    start = time.time()
    kpts_array, desc_array = pyhesaff.detect_kpts_list(img_fpaths)
    partime = time.time() - start
    print('Parallel ran in %r seconds' % partime)

    itertime = 0
    for (img_fpath, kpts, desc) in zip(img_fpaths,
                                        kpts_array,
                                        desc_array):
        start = time.time()
        kpts_, desc_ = pyhesaff.detect_kpts(img_fpath)
        print('shape(kpts, kpts_, desc, desc_) = %9r, %9r, %11r, %11r' %
              (kpts.shape, kpts_.shape, desc.shape, desc_.shape))
        try:
            assert len(kpts_) > 0, 'no kpts detected'
            assert np.all(kpts_ == kpts), 'parallel computation inconsistent'
            assert np.all(desc_ == desc), 'parallel computation inconsistent'
            #assert False, 'deliberate triggering to see printouts'
        except Exception as ex:
            #print(np.dtype(float).itemsize)
            #print(np.dtype(int).itemsize)
            #print(np.dtype(np.float32).itemsize)
            #print(np.dtype(np.float64).itemsize)
            print('==========')
            print('kpts_')
            print(kpts_[0:2])
            print('==========')
            print('kpts')
            print(kpts[0:2])
            print('---')
            print('==========')
            print('desc_')
            print(desc_[0:2])
            print('==========')
            print('desc')
            print(desc[0:2])
            print('---')
            #print(kpts_)
            #print(kpts)
            #for i, (k1, k2) in enumerate(zip(kpts, kpts_)):
            #    print(i)
            #    print(k1)
            #    print(k2)
            #    print('-----')
            #print('dashdashdash')
            #for i, (d1, d2) in enumerate(zip(desc, desc_)):
            #    print(i)
            #    print(d1)
            #    print(d2)
            #    print('-----')
            ut.printex(ex)
            raise
        itertime += time.time() - start
    print('Iterative ran in %r seconds' % itertime)
    print('Keypoints seem consistent')

    if ut.show_was_requested():
        # Do not plot by default
        fig = plt.figure()

        for count, (img_fpath, kpts, desc) in enumerate(zip(img_fpaths,
                                                            kpts_array,
                                                            desc_array)):
            ax = fig.add_subplot(2, 2, count + 1)
            img = mpl.image.imread(img_fpath)
            plt.imshow(img)
            _xs, _ys = kpts.T[0:2]
            ax.plot(_xs, _ys, 'ro', alpha=.5)

        plt.show()


if __name__ == '__main__':
    """
    CommandLine:
        python -m pyhesaff.tests.test_pyhesaff_simple_parallel
        python -m pyhesaff.tests.test_pyhesaff_simple_parallel --allexamples
        python -m pyhesaff.tests.test_pyhesaff_simple_parallel --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
