from __future__ import print_function, division
# Science
import numpy as np
from numpy import (array, rollaxis, sqrt, vstack, zeros,)
# VTool
from vtool.linalg import svd


#PYX START
"""
// These are cython style comments for maintaining python compatibility
cimport numpy as np
ctypedef np.float64_t FLOAT64
"""
#PYX MAP FLOAT_2D np.ndarray[FLOAT64, ndim=2]
#PYX MAP FLOAT_1D np.ndarray[FLOAT64, ndim=1]
#PYX END

tau = np.pi * 2  # tauday.com
KPTS_DTYPE = np.float32


def _append_gravity(kpts):
    assert kpts.shape[1] == 5
    theta = zeros(len(kpts)) + tau / 4
    kpts2 = vstack((kpts.T, theta)).T
    return kpts2


def cast_split(kpts, dtype=KPTS_DTYPE):
    'breakup keypoints into location, shape, and orientation'
    assert kpts.shape[1] == 6
    kptsT = kpts.T
    _xs   = array(kptsT[0], dtype=dtype)
    _ys   = array(kptsT[1], dtype=dtype)
    _acds = array(kptsT[2:5], dtype=dtype)
    _oris = array(kptsT[5:6], dtype=dtype)
    return _xs, _ys, _acds, _oris


def xys(kpts):
    'keypoint locations'
    _xs, _ys   = kpts.T[0:2]
    return _xs, _ys


def acd(kpts):
    'keypoint shapes'
    _acds = kpts.T[2:5]
    return _acds


def ori(kpts):
    'keypoint orientations'
    _oris = kpts.T[5:6]
    return _oris


def scale_sqrd(kpts):
    'gets average squared scale (does not take into account elliptical shape'
    _as, _cs, _ds = acd(kpts)
    _scales_sqrd = _as * _ds
    return _scales_sqrd


def scale(kpts):
    'gets average scale (does not take into account elliptical shape'
    _scales = sqrt(scale_sqrd(kpts))
    return _scales


def affine_matrix(kpts):
    'packs keypoint shapes into affine matrixes'
    _as, _cs, _ds = acd(kpts)
    _bs = zeros(len(_cs))
    aff_tups = ((_as, _bs), (_cs, _ds))
    aff_mats = rollaxis(array(aff_tups), 2)
    return aff_mats


def orthogonal_scales(kpts):
    'gets the scales of the major and minor elliptical axis'
    aff_mats = affine_matrix(kpts)
    USV_list = [svd(A) for A in aff_mats]
    S_list = array([S for U, S, V in USV_list])
    return S_list


def diag_extent_sqrd(kpts):
    xs, ys = xys(kpts)
    x_extent_sqrd = (xs.max() - xs.min()) ** 2
    y_extent_sqrd = (ys.max() - ys.min()) ** 2
    extent_sqrd = x_extent_sqrd + y_extent_sqrd
    return extent_sqrd


# Ensure that a feature doesn't have multiple assignments
# --------------------------------
# Linear algebra functions on lower triangular matrices

#PYX DEFINE
def det_acd(acd):
    #cdef det_acd(FLOAT_2D acd):
    'Lower triangular determinant'
    #PYX CDEF FLOAT_1D
    det = acd[0] * acd[2]
    return det


#PYX DEFINE
def inv_acd(acd, det):
    #cdef inv_acd(FLOAT_2D acd, FLOAT_1D det):
    'Lower triangular inverse'
    #PYX CDEF FLOAT_2D
    inv_acd = np.array((acd[2], -acd[1], acd[0])) / det
    return inv_acd


#PYX BEGIN
def dot_acd(acd1, acd2):
    #cdef dot_acd(FLOAT_2D acd1, FLOAT_2D acd2):
    'Lower triangular dot product'
    a = (acd1[0] * acd2[0])     # PYX FLOAT_1D
    c = ((acd1[1] * acd2[0]) +
         (acd1[2] * acd2[1]))   # PYX FLOAT_1D
    d = (acd1[2] * acd2[2])     # PYX FLOAT_1D
    acd3 = np.array((a, c, d))  # PYX FLOAT_2D
    return acd3
# PYX END CDEF


def warp_to_circle_mat(a, c, d):
    invA = np.array([[a, 0, 0],
                     [c, d, 0],
                     [0, 0, 1]])
    # kp is given in invA format. Convert to A
    A = np.linalg.inv(invA)
    return A
