from __future__ import print_function, division
# Science
import cv2
from numpy import (array, sin, cos,)


def svd(M):
    flags = cv2.SVD_FULL_UV
    S, U, V = cv2.SVDecomp(M, flags=flags)
    S = S.flatten()
    return U, S, V


def rotation_mat(radians):
    sin_ = sin(radians)
    cos_ = cos(radians)
    R = array(((cos_, -sin_, 0),
               (sin_,  cos_, 0),
               (   0,      0, 1)))
    return R


def translation_mat(x, y):
    T = array([[1, 0,  x],
               [0, 1,  y],
               [0, 0,  1]])
    return T


def scale_mat(ss):
    S = array([[ss, 0, 0],
               [0, ss, 0],
               [0,  0, 1]])
    return S
