from __future__ import print_function, division
# Science
import numpy as np

KPTS_DTYPE = np.float32


def append_gravity(kpts):
    assert kpts.shape[1] == 5
    np.tau = np.pi * 2
    theta = np.zeros(len(kpts)) + np.tau / 4
    kpts2 = np.vstack((kpts.T, theta)).T
    return kpts2


def cast_split(kpts, dtype=KPTS_DTYPE):
    'breakup keypoints into position and shape'
    assert kpts.shape[1] == 6
    kptsT = kpts.T
    _xs   = np.array(kptsT[0], dtype=dtype)
    _ys   = np.array(kptsT[1], dtype=dtype)
    _acds = np.array(kptsT[2:5], dtype=dtype)
    _oris = np.array(kptsT[5:6], dtype=dtype)
    return _xs, _ys, _acds, _oris


def split(kpts, dtype=KPTS_DTYPE):
    'breakup keypoints into position and shape'
    assert kpts.shape[1] == 6
    kptsT = kpts.T
    _xs   = np.array(kptsT[0], dtype=dtype)
    _ys   = np.array(kptsT[1], dtype=dtype)
    _acds = np.array(kptsT[2:5], dtype=dtype)
    _oris = np.array(kptsT[5:6], dtype=dtype)
    return _xs, _ys, _acds, _oris


def xys(kpts):
    'breakup keypoints into position and shape'
    _xs, _ys   = kpts.T[0:2]
    return _xs, _ys


def acd(kpts):
    _acds = kpts.T[2:5]
    return _acds


def oris(kpts):
    _oris = kpts.T[5:6]
    return _oris


def diag_extent_sqrd(kpts):
    xs, ys = xys(kpts)
    x_extent_sqrd = (xs.max() - xs.min()) ** 2
    y_extent_sqrd = (ys.max() - ys.min()) ** 2
    extent_sqrd = x_extent_sqrd + y_extent_sqrd
    return extent_sqrd
