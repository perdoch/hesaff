#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.getcwd())
import pyhesaff
import matplotlib as mpl
import numpy as np
from itertools import izip
from matplotlib import pyplot as plt
import time

#sample_data_fpath = join(dirname(mpl.__file__), 'mpl-data', 'sample_data')

#mpl.cbook.get_sample_data('grace_hopper.png', False)
#mpl.cbook.get_sample_data('lena.png', False)

#mpl.cbook.get_sample_data('Minduka_Present_Blue_Pack.png', False)
#mpl.cbook.get_sample_data('logo2.png', False)
#mpl.cbook.get_sample_data('grace_hopper.jpg', False)
#mpl.cbook.get_sample_data('ada.png', False)


if __name__ == '__main__':
    lena_fpath  = mpl.cbook.get_sample_data('lena.jpg', False)
    logo_fpath  = mpl.cbook.get_sample_data('logo2.png', False)
    grace_fpath = mpl.cbook.get_sample_data('grace_hopper.jpg', False)
    ada_fpath   = mpl.cbook.get_sample_data('ada.png', False)

    fig = plt.figure()

    img_fpaths = [lena_fpath, logo_fpath, grace_fpath, ada_fpath] * 10

    start = time.time()
    kpts_array, desc_array = pyhesaff.detect_kpts_list(img_fpaths)
    partime = time.time() - start
    print('Parallel ran in %r seconds' % partime)

    itertime = 0
    for (img_fpath, kpts, desc) in izip(img_fpaths,
                                        kpts_array,
                                        desc_array):
        start = time.time()
        kpts_, desc_ = pyhesaff.detect_kpts(img_fpath)
        assert np.all(kpts_ == kpts), 'parallel computation inconsistent'
        assert np.all(desc_ == desc), 'parallel computation inconsistent'
        itertime += time.time() - start
    print('Iterative ran in %r seconds' % itertime)
    print('Keypoints seem consistent')

    if '--show' in sys.argv:
        # Do not plot by default
        for count, (img_fpath, kpts, desc) in enumerate(izip(img_fpaths,
                                                             kpts_array,
                                                             desc_array)):
            ax = fig.add_subplot(2, 2, count + 1)
            img = mpl.image.imread(img_fpath)
            plt.imshow(img)
            _xs, _ys = kpts.T[0:2]
            ax.plot(_xs, _ys, 'ro', alpha=.5)

        plt.show()
