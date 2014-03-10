# LICENCE
from __future__ import print_function, division
# Python
from itertools import izip
# Science
from scipy.signal import argrelextrema
import numpy as np


def hist_edges_to_centers(edges):
    return np.array([(e1 + e2) / 2 for (e1, e2) in izip(edges[:-1], edges[1:])])


def wrap_histogram(hist, edges):
    low, high = np.diff(edges)[[0, -1]]
    hist_wrap = np.hstack((hist[-1:], hist, hist[0:1]))
    edge_wrap = np.hstack((edges[0:1] - low, edges, edges[-1:] + high))
    return hist_wrap, edge_wrap


def hist_argmaxima(hist, centers=None):
    argmaxima = argrelextrema(hist, np.greater)[0]
    maxima_x = argmaxima if centers is None else centers[argmaxima]
    maxima_y = hist[argmaxima]
    return maxima_x, maxima_y, argmaxima


def maxima_neighbors(argmaxima, hist, centers=None):
    neighbs = np.vstack((argmaxima - 1, argmaxima, argmaxima + 1))
    y123 = hist[neighbs]
    x123 = neighbs if centers is None else centers[neighbs]
    return x123, y123


def interpolate_submaxima(argmaxima, hist, centers=None):
    x123, y123 = maxima_neighbors(argmaxima, hist, centers)
    (y1, y2, y3) = y123
    (x1, x2, x3) = x123
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    A     = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    B     = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
    C     = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom
    xv = -B / (2 * A)
    yv = C - B * B / (4 * A)
    submaxima_x, submaxima_y = np.vstack((xv.T, yv.T))
    return submaxima_x, submaxima_y
