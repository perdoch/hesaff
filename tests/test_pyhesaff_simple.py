#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import pyhesaff
import matplotlib as mpl
from matplotlib import pyplot as plt

#sample_data_fpath = join(dirname(mpl.__file__), 'mpl-data', 'sample_data')

#mpl.cbook.get_sample_data('grace_hopper.png', False)
#mpl.cbook.get_sample_data('lena.png', False)

#mpl.cbook.get_sample_data('Minduka_Present_Blue_Pack.png', False)
#mpl.cbook.get_sample_data('logo2.png', False)
#mpl.cbook.get_sample_data('grace_hopper.jpg', False)
#mpl.cbook.get_sample_data('ada.png', False)


def test_detect_then_show(ax, img_fpath):
    kpts, desc = pyhesaff.detect_kpts(img_fpath)
    img = mpl.image.imread(img_fpath)
    plt.imshow(img)
    _xs, _ys = kpts.T[0:2]
    ax.plot(_xs, _ys, 'ro', alpha=.5)


if __name__ == '__main__':

    lena_fpath  = mpl.cbook.get_sample_data('lena.jpg', False)
    logo_fpath  = mpl.cbook.get_sample_data('logo2.png', False)
    grace_fpath = mpl.cbook.get_sample_data('grace_hopper.jpg', False)
    ada_fpath   = mpl.cbook.get_sample_data('ada.png', False)

    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    test_detect_then_show(ax, lena_fpath)

    ax = fig.add_subplot(2, 2, 2)
    test_detect_then_show(ax, logo_fpath)

    ax = fig.add_subplot(2, 2, 3)
    test_detect_then_show(ax, grace_fpath)

    ax = fig.add_subplot(2, 2, 4)
    test_detect_then_show(ax, ada_fpath)

    plt.show()
