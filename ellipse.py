'This module should handle all things elliptical'
from __future__ import print_function, division
# Python
from itertools import izip
# Scientific
from numpy import array, zeros, ones
from numpy.core.umath_tests import matrix_multiply
from scipy.signal import argrelextrema
import cv2
import numpy as np
import scipy as sp  # NOQA
# Hotspotter
from hscom import __common__
print, print_, print_on, print_off, rrr, profile, printDBG =\
    __common__.init(__name__, module_prefix='[ell]', DEBUG=False, initmpl=False)


def test_data():
    import test_pyhesaff
    test_data = test_pyhesaff.load_test_data()
    kpts = test_data['kpts']
    exec(open('ellipse.py').read())
    return locals()


@profile
def adaptive_scale(img_fpath, kpts, nScales=4, low=-.5, high=.5, nSamples=16):
    imgBGR = cv2.imread(img_fpath, flags=cv2.CV_LOAD_IMAGE_COLOR)

    nKp = len(kpts)
    dtype_ = kpts.dtype

    # Work with float65
    kpts_ = np.array(kpts, dtype=np.float64)

    # Expand each keypoint into a number of different scales
    expanded_kpts = expand_scales(kpts_, nScales, low, high)

    # Sample gradient magnitude around the border
    border_vals_sum = sample_ell_border_vals(imgBGR, expanded_kpts, nKp, nScales, nSamples)

    # interpolate maxima
    subscale_kpts = subscale_peaks(border_vals_sum, kpts_, nScales, low, high)

    # Make sure that the new shapes are in bounds
    height, width = imgBGR.shape[0:2]
    isvalid = check_kpts_in_bounds(subscale_kpts, width, height)

    # Convert to the original dtype
    adapted_kpts = np.array(subscale_kpts[isvalid], dtype=dtype_)
    return adapted_kpts


@profile
def check_kpts_in_bounds(kpts_, width, height):
    # Test to make sure the extents of the keypoints are in bounds
    unit_bbox = np.array([(-1, -1, 1),
                          (-1,  1, 1),
                          ( 1, -1, 1),
                          ( 1,  1, 1)]).T
    invV = kpts_to_invV(kpts_)
    bbox_pts = [v.dot(unit_bbox)[0:2] for v in invV]
    maxx = np.array([pts[0].max() for pts in bbox_pts]) < width
    minx = np.array([pts[0].min() for pts in bbox_pts]) > 0
    maxy = np.array([pts[1].max() for pts in bbox_pts]) < height
    miny = np.array([pts[1].min() for pts in bbox_pts]) > 0
    isvalid = np.array(maxx * minx * maxy * miny, dtype=np.bool)
    return isvalid


@profile
def expand_scales(kpts, nScales, low, high):
    scales = 2 ** np.linspace(low, high, nScales)
    expanded_kpts_list = expand_kpts(kpts, scales)
    expanded_kpts = np.vstack(expanded_kpts_list)
    #assert len(expanded_kpts_list) == nScales
    #assert expanded_kpts.shape == (nKp * nScales, 5)
    return expanded_kpts


@profile
def sample_ell_border_pts(expanded_kpts, nSamples):
    ell_border_pts_list = sample_uniform(expanded_kpts, nSamples)
    #assert len(ell_border_pts_list) == nKp * nScales
    #assert ell_border_pts_list[0].shape == (nSamples, 2)
    ell_border_pts = np.vstack(ell_border_pts_list)
    #assert ell_border_pts.shape == (nKp * nScales * nSamples, 2)
    #assert ell_border_pts.shape == (nKp * nScales * nSamples, 2)
    return ell_border_pts


@profile
def sample_ell_border_vals(imgBGR, expanded_kpts, nKp, nScales, nSamples):
    # Sample points uniformly across the boundary
    ell_border_pts = sample_ell_border_pts(expanded_kpts, nSamples)
    # Build gradient magnitude imaeg
    imgLAB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2LAB)
    imgL = imgLAB[:, :, 0]
    imgMag = gradient_magnitude(imgL)
    border_vals = subpixel_values(imgMag, ell_border_pts)
    #assert len(border_vals) == (nKp * nScales * nSamples)
    border_vals.shape = (nKp, nScales, nSamples, 1)
    border_vals_sum = border_vals.sum(3).sum(2)
    #assert border_vals_sum.shape == (nKp, nScales)
    return border_vals_sum


def interpolate_between(peak_list, nScales, high, low):
    def bin_to_subscale(bins):
        return 2 ** ((peaks[:, 0] / nScales) * (high - low) + low)
    subscale_list = [bin_to_subscale(peaks) if len(peaks) > 0 else [] for peaks in peak_list]
    return subscale_list


@profile
def subscale_peaks(border_vals_sum, kpts, nScales, low, high):
    peak_list = interpolate_maxima(border_vals_sum)
    subscale_list = interpolate_between(peak_list, nScales, high, low)
    subscale_kpts = expand_subscales(kpts, subscale_list)
    return subscale_kpts


def expand_kpts(kpts, scales):
    expanded_kpts_list = []
    for scale in scales:
        kpts_ = kpts.copy()
        kpts_.T[2] *= scale
        kpts_.T[3] *= scale
        kpts_.T[4] *= scale
        expanded_kpts_list.append(kpts_)
    return expanded_kpts_list


def expand_subscales(kpts, subscale_list):
    subscale_kpts_list = [kp * np.array((1, 1, scale, scale, scale))
                          for kp, subscales in izip(kpts, subscale_list)
                          for scale in subscales]
    subscale_kpts = np.vstack(subscale_kpts_list)
    return subscale_kpts


def find_maxima(y_list):
    maxima_list = [argrelextrema(y, np.greater)[0] for y in y_list]
    return maxima_list


def extrema_neighbors(extrema_list, nBins):
    extrema_left_list  = [np.clip(extrema - 1, 0, nBins) for extrema in extrema_list]
    extrema_right_list = [np.clip(extrema + 1, 0, nBins) for extrema in extrema_list]
    return extrema_left_list, extrema_right_list


def find_maxima_with_neighbors(scalar_list):
    y_list = [scalars for scalars in scalar_list]
    nBins = len(y_list[0])
    x = np.arange(nBins)
    maxima_list = find_maxima(y_list)
    maxima_left_list, maxima_right_list = extrema_neighbors(maxima_list, nBins)
    data_list = [np.vstack([exl, exm, exr]) for exl, exm, exr in izip(maxima_left_list, maxima_list, maxima_right_list)]
    x_data_list = [[] if data.size == 0 else x[data] for data in iter(data_list)]
    y_data_list = [[] if data.size == 0 else y[data] for y, data in izip(y_list, data_list)]
    return x_data_list, y_data_list


def interpolate_maxima(scalar_list):
    # scalar_list = border_vals_sum
    x_data_list, y_data_list = find_maxima_with_neighbors(scalar_list)
    peak_list = interpolate_peaks(x_data_list, y_data_list)
    return peak_list


def interpolate_peaks2(x_data_list, y_data_list):
    coeff_list = []
    for x_data, y_data in izip(x_data_list, y_data_list):
        for x, y in izip(x_data.T, y_data.T):
            coeff = np.polyfit(x, y, 2)
            coeff_list.append(coeff)


@profile
def interpolate_peaks(x_data_list, y_data_list):
    #http://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-point
    peak_list = []
    for x_data, y_data in izip(x_data_list, y_data_list):
        if len(y_data) == 0:
            peak_list.append([])
            continue
        y1, y2, y3 = y_data
        x1, x2, x3 = x_data
        denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
        A     = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
        B     = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
        C     = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom
        xv = -B / (2 * A)
        yv = C - B * B / (4 * A)
        peak_list.append(np.vstack((xv.T, yv.T)).T)
    return peak_list


@profile
def sample_uniform(kpts, nSamples=128):
    nKp = len(kpts)
    invV, V, Z = kpts_matrix(kpts)
    circle_pts = homogenous_circle_pts(nSamples + 1)[0:-1]
    assert circle_pts.shape == (nSamples, 3)
    polygon1_list = array([v.dot(circle_pts.T).T for v in invV])
    assert polygon1_list.shape == (nKp, nSamples, 3)
    # The transformed points are not sampled uniformly... Bummer
    # We will sample points evenly across the sampled polygon
    # then we will project them onto the ellipse
    dists = array([circular_distance(arr) for arr in polygon1_list])
    assert dists.shape == (nKp, nSamples)
    # perimeter of the polygon
    perimeter = dists.sum(1)
    assert perimeter.shape == (nKp,)
    # Take a perfect multiple of steps along the perimeter
    multiplier = 1
    step_size = perimeter / (nSamples * multiplier)
    assert step_size.shape == (nKp,)
    # Walk along edge
    num_steps_list = []
    offset_list = []
    total_dist = zeros(step_size.shape)  # step_size.copy()
    dist_walked = zeros(step_size.shape)
    assert dist_walked.shape == (nKp,)
    assert total_dist.shape == (nKp,)
    distsT = dists.T
    assert distsT.shape == (nSamples, nKp)

    # This loops over the pt samples and performs the operation for every keypoint
    for count in xrange(nSamples):
        segment_len = distsT[count]
        # Find where your starting location is
        offset_list.append(total_dist - dist_walked)
        # How far can you possibly go?
        total_dist += segment_len
        # How many steps can you take?
        num_steps = (total_dist - dist_walked) // step_size
        num_steps_list.append(num_steps)
        # Log how much further youve gotten
        dist_walked += (num_steps * step_size)
    # Check for floating point errors
    # take an extra step if you need to
    num_steps_list[-1] += np.round((perimeter - dist_walked) / step_size)
    assert np.all(array(num_steps_list).sum(0) == nSamples)

    #offset_iter1 = zip(num_steps_list, distsT, offset_list)
    #offset_list = [((step_size - offset) / dist, ((num * step_size) - offset) / dist, num)
                   #for num, dist, offset in zip(num_steps_list, distsT, offset_list)]

    #offset_iter2 = offset_list

    #cut_locs = [[
        #np.linspace(off1, off2, n, endpoint=True) for (off1, off2, n) in izip(offset1, offset2, num)]
        #for (offset1, offset2, num) in offset_iter2
    #]
    # store the percent location at each line segment where
    # the cut will be made
    # HERE IS NEXT
    cut_list = []
    # This loops over the pt samples and performs the operation for every keypoint
    for num, dist, offset in izip(num_steps_list, distsT, offset_list):
        #if num == 0
            #cut_list.append([])
            #continue
        # This was a bitch to keep track of
        offset1 = (step_size - offset) / dist
        offset2 = ((num * step_size) - offset) / dist
        cut_locs = [np.linspace(off1, off2, n, endpoint=True) for (off1, off2, n) in izip(offset1, offset2, num)]
        # post check for divide by 0
        cut_locs = [array([0 if np.isinf(c) else c for c in cut]) for cut in cut_locs]
        cut_list.append(cut_locs)
    cut_list = array(cut_list).T
    assert cut_list.shape == (nKp, nSamples)

    # =================
    # METHOD 1
    # =================
    # Linearly interpolate between points on the polygons at the places we cut
    def interpolate(pt1, pt2, percent):
        # interpolate between point1 and point2
        return ((1 - percent) * pt1) + ((percent) * pt2)

    def polygon_points(polygon_pts, dist_list):
        return array([interpolate(polygon_pts[count], polygon_pts[(count + 1) % nSamples], loc)
                      for count, locs in enumerate(dist_list)
                      for loc in iter(locs)])

    new_locations = array([polygon_points(polygon_pts, cuts) for polygon_pts,
                           cuts in izip(polygon1_list, cut_list)])
    # =================
    # =================
    # METHOD 2
    #from itertools import cycle as icycle
    #from itertools import islice

    #def icycle_shift1(iterable):
        #return islice(icycle(poly_pts), 1, len(poly_pts) + 1)

    #cutptsIter_list = [izip(iter(poly_pts), icycle_shift1(poly_pts), cuts)
                       #for poly_pts, cuts in izip(polygon1_list, cut_list)]
    #new_locations = [[[((1 - cut) * pt1) + ((cut) * pt2) for cut in cuts]
                      #for (pt1, pt2, cuts) in cutPtsIter]
                     #for cutPtsIter in cutptsIter_list]
    # =================

    # assert new_locations.shape == (nKp, nSamples, 3)
    # Warp new_locations to the unit circle
    #new_unit = V.dot(new_locations.T).T
    new_unit = array([v.dot(newloc.T).T for v, newloc in izip(V, new_locations)])
    # normalize new_unit
    new_mag = np.sqrt((new_unit ** 2).sum(-1))
    new_unorm_unit = new_unit / np.dstack([new_mag] * 3)
    new_norm_unit = new_unorm_unit / np.dstack([new_unorm_unit[:, :, 2]] * 3)
    # Get angle (might not be necessary)
    #x_axis = array([1, 0, 0])
    #arccos_list = x_axis.dot(new_norm_unit.T)
    #uniform_theta_list = np.arccos(arccos_list)
    # Maybe this?
    # Find the angle from the center of the circle
    theta_list2 = np.arctan2(new_norm_unit[:, :, 1], new_norm_unit[:, :, 0])
    # assert uniform_theta_list.shape = (nKp, nSample)
    # Use this angle to unevenly sample the perimeter of the circle
    uneven_cicrle_pts = np.dstack([np.cos(theta_list2), np.sin(theta_list2), ones(theta_list2.shape)])
    # The uneven circle points were sampled in such a way that when they are
    # transformeed they will be approximately uniform along the boundary of the
    # ellipse.
    uniform_ell_hpts = [v.dot(pts.T).T for (v, pts) in izip(invV, uneven_cicrle_pts)]
    # Remove the homogenous coordinate and we're done
    ell_border_pts_list = [pts[:, 0:2] for pts in uniform_ell_hpts]
    return ell_border_pts_list


#----------------
# Image Helpers
#----------------


def gradient_magnitude(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    imgMag = np.sqrt(sobelx ** 2 + sobely ** 2)
    return imgMag


def subpixel_values(img, pts):
    ''' adapted from
    stackoverflow.com/uestions/12729228/simple-efficient-binlinear-
    interpolation-of-images-in-numpy-and-python'''
    # Image info
    height, width = img.shape[0:2]
    nChannels = 1 if len(img.shape) == 2 else img.shape[2]

    # Subpixel locations to sample
    ptsT = pts.T
    x = ptsT[0]
    y = ptsT[1]

    # Get quantized pixel locations near subpixel pts
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    # Make sure the values do not go past the boundary
    x0 = np.clip(x0, 0, width - 1)
    x1 = np.clip(x1, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)
    y1 = np.clip(y1, 0, height - 1)

    # Find bilinear weights
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)
    if  nChannels != 1:
        wa = array([wa] *  nChannels).T
        wb = array([wb] *  nChannels).T
        wc = array([wc] *  nChannels).T
        wd = array([wd] *  nChannels).T

    # Sample values
    Ia = img[y0, x0]
    Ib = img[y1, x0]
    Ic = img[y0, x1]
    Id = img[y1, x1]

    # Perform the bilinear interpolation
    subpxl_vals = (wa * Ia) + (wb * Ib) + (wc * Ic) + (wd * Id)
    return subpxl_vals


def get_num_channels(img):
    return 1 if len(img.shape) == 2 else img.shape[2]


#----------------
# Numeric Helpers
#----------------

def kpts_to_invV(kpts):
    nKp = len(kpts)
    (ia13, ia23, ia11, ia21, ia22) = np.array(kpts).T
    ia12 = zeros(nKp)
    ia31 = zeros(nKp)
    ia32 = zeros(nKp)
    ia33 = ones(nKp)
    # np.dot operates over the -1 and -2 axis of arrays
    # Start with
    #     invV.shape = (3, 3, nKp)
    invV = array([[ia11, ia12, ia13],
                  [ia21, ia22, ia23],
                  [ia31, ia32, ia33]])
    # And roll into
    invV = np.rollaxis(invV, 2)
    invV = np.ascontiguousarray(invV)
    assert invV.shape == (nKp, 3, 3)
    return invV


@profile
def kpts_matrix(kpts):
    # We are given the keypoint in invA format
    # invV = perdoch.invA
    #    V = perdoch.A
    #    Z = perdoch.E
    # invert into V
    nKp = len(kpts)
    invV = kpts_to_invV(kpts)
    V = [np.linalg.inv(v) for v in invV]
    assert len(V) == (nKp)
    #V = faster_inverse(invV)
    # transform into conic matrix Z
    # Z = (V.T).dot(V)
    Vt = array(map(np.transpose, V))
    Z = matrix_multiply(Vt, V)
    assert Z.shape == (nKp, 3, 3)
    return invV, V, Z


@profile
def homogenous_circle_pts(nSamples):
    # Make a list of homogenous circle points
    tau = 2 * np.pi
    theta_list = np.linspace(0, tau, nSamples)
    cicrle_pts = array([(np.cos(t_), np.sin(t_), 1) for t_ in theta_list])
    return cicrle_pts


@profile
def circular_distance(arr=None):
    dist_head_ = ((arr[0:-1] - arr[1:]) ** 2).sum(1)
    dist_tail_  = ((arr[-1] - arr[0]) ** 2).sum(0)
    #dist_end_.shape = (1, len(dist_end_))
    #print(dist_most_.shape)
    #print(dist_end_.shape)
    dists = np.sqrt(np.hstack((dist_head_, dist_tail_)))
    return dists


def almost_eq(a, b, thresh=1E-11):
    return abs(a - b) < thresh


def rotation(theta):
    sin_ = np.sin(theta)
    cos_ = np.cos(theta)
    rot_ = array([[cos_, -sin_],
                  [sin_, cos_]])
    return rot_


def rotation_around(theta, x, y):
    sin_ = np.sin(theta)
    cos_ = np.cos(theta)
    tr1_ = array([[1, 0, -x],
                  [0, 1, -y],
                  [0, 0, 1]])
    rot_ = array([[cos_, -sin_, 0],
                  [sin_, cos_,  0],
                  [   0,    0,  1]])
    tr2_ = array([[1, 0, x],
                  [0, 1, y],
                  [0, 0, 1]])
    rot = tr2_.dot(rot_).dot(tr1_)
    return rot
