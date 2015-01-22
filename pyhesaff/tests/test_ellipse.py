#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
#----------------
# Test Functions
#----------------
import pyhestest
from six.moves import zip, range
import utool
from plottool import draw_func2 as df2
from plottool.viz_keypoints import show_keypoints
import matplotlib as mpl
import numpy as np
import vtool.linalg as ltool


TAU = 2 * np.pi  # References: tauday.com


def draw_expanded_scales(imgL, sel_kpts, exkpts, exdesc_):
    draw_keypoint_patch = df2.draw_keypoint_patch
    get_warped_patch = df2.get_warped_patch  # NOQA

    # Rows are for different scales
    # Cols are for different patches
    # There is a prefix row showing the original image
    nRows, nCols = len(exkpts), len(sel_kpts)
    exkpts_ = np.vstack(exkpts)

    fnum = 1
    df2.figure(fnum=fnum, docla=True, doclf=True)

    nPreRows = 1
    nPreCols = (nPreRows * nCols) + 1

    show_keypoints(imgL, exkpts_, fnum=fnum, pnum=(nRows + nPreRows, 1, 1),
                   color=df2.BLUE)

    px = 0
    for row, kpts_ in enumerate(exkpts):
        for col, kp in enumerate(kpts_):
            sift = exdesc_[px]
            pnum = (nRows + nPreRows, nCols, px + nPreCols)
            draw_keypoint_patch(imgL, kp, sift, warped=True, fnum=fnum, pnum=pnum)
            px += 1
    df2.draw()

    print('nRows = %r' % nRows)
    print('nCols = %r' % nCols)


def in_depth_ellipse(kp):
    kp = np.array(kp, dtype=np.float64)
    print('kp = %r' % kp)
    #-----------------------
    # SETUP
    #-----------------------
    np.set_printoptions(precision=3)
    df2.reset()
    df2.figure(9003, docla=True, doclf=True)
    ax = df2.gca()
    ax.invert_yaxis()

    def _plotpts(data, px, color=df2.BLUE, label='', marker='.'):
        #df2.figure(9003, docla=True, pnum=(1, 1, px))
        df2.plot2(data.T[0], data.T[1], marker, '', color=color, label=label)
        df2.update()

    def _plotarrow(x, y, dx, dy, color=df2.BLUE, label=''):
        ax = df2.gca()
        arrowargs = dict(head_width=.5, length_includes_head=True, label=label)
        arrow = mpl.patches.FancyArrow(x, y, dx, dy, **arrowargs)
        arrow.set_edgecolor(color)
        arrow.set_facecolor(color)
        ax.add_patch(arrow)
        df2.update()

    #-----------------------
    # INPUT
    #-----------------------
    # We will call perdoch's invA = invV
    print('--------------------------------')
    print('Let V = Perdoch.A')
    print('Let Z = Perdoch.E')
    print('--------------------------------')
    print('Input from Perdoch\'s detector: ')

    # We are given the keypoint in invA format
    if len(kp) == 5:
        (ix, iy, iv11, iv21, iv22), iv12 = kp, 0
    elif len(kp) == 6:
        (ix, iy, iv11, iv21, iv22, ori), iv12 = kp, 0
    invV = np.array([[iv11, iv12, ix],
                     [iv21, iv22, iy],
                     [   0,    0,  1]])
    V = np.linalg.inv(invV)
    Z = (V.T).dot(V)

    print('invV is a transform from points on a unit-circle to the ellipse')
    utool.horiz_print('invV = ', invV)
    print('--------------------------------')
    print('V is a transformation from points on the ellipse to a unit circle')
    utool.horiz_print('V = ', V)
    print('--------------------------------')
    print('An ellipse is a special case of a conic. For any ellipse:')
    print('Points on the ellipse satisfy (x_ - x_0).T.dot(Z).dot(x_ - x_0) = 1')
    print('where Z = (V.T).dot(V)')
    utool.horiz_print('Z = ', Z)

    # Define points on a unit circle
    nSamples = 12
    theta_list = np.linspace(0, TAU, nSamples)
    cicrle_pts = np.array([(np.cos(t_), np.sin(t_), 1) for t_ in theta_list])

    # Transform those points to the ellipse using invV
    ellipse_pts1 = invV.dot(cicrle_pts.T).T

    #Lets check our assertion: (x_ - x_0).T.dot(Z).dot(x_ - x_0) == 1
    x_0 = np.array([ix, iy, 1])
    checks = [(x_ - x_0).T.dot(Z).dot(x_ - x_0) for x_ in ellipse_pts1]
    try:
        # HELP: The phase is off here. in 3x3 version I'm not sure why
        #assert all([almost_eq(1, check) for check in checks1])
        is_almost_eq_pos1 = [utool.almost_eq(1, check) for check in checks]
        is_almost_eq_neg1 = [utool.almost_eq(-1, check) for check in checks]
        assert all(is_almost_eq_pos1)
    except AssertionError as ex:
        print('circle pts = %r ' % cicrle_pts)
        print(ex)
        print(checks)
        print([utool.almost_eq(-1, check, 1E-9) for check in checks])
        raise
    else:
        #assert all([abs(1 - check) < 1E-11 for check in checks2])
        print('... all of our plotted points satisfy this')

    #=======================
    # THE CONIC SECTION
    #=======================
    # All of this was from the Perdoch paper, now lets move into conic sections
    # We will use the notation from wikipedia
    # http://en.wikipedia.org/wiki/Conic_section
    # http://en.wikipedia.org/wiki/Matrix_representation_of_conic_sections

    #-----------------------
    # MATRIX REPRESENTATION
    #-----------------------
    # The matrix representation of a conic is:
    #(A,  B2, B2_, C) = Z.flatten()
    #(D, E, F) = (0, 0, 1)
    (A,  B2, D2, B2_, C, E2, D2_, E2_, F) = Z.flatten()
    B = B2 * 2
    D = D2 * 2
    E = E2 * 2
    assert B2 == B2_, 'matrix should by symmetric'
    assert D2 == D2_, 'matrix should by symmetric'
    assert E2 == E2_, 'matrix should by symmetric'
    print('--------------------------------')
    print('Now, using wikipedia\' matrix representation of a conic.')
    con = np.array((('    A', 'B / 2', 'D / 2'),
                    ('B / 2', '    C', 'E / 2'),
                    ('D / 2', 'E / 2', '    F')))
    utool.horiz_print('A matrix A_Q = ', con)

    # A_Q is our conic section (aka ellipse matrix)
    A_Q = np.array(((    A, B / 2, D / 2),
                    (B / 2,     C, E / 2),
                    (D / 2, E / 2,     F)))

    utool.horiz_print('A_Q = ', A_Q)

    #-----------------------
    # DEGENERATE CONICS
    #---http://individual.utoronto.ca/somody/quiz.html--------------------
    print('----------------------------------')
    print('As long as det(A_Q) != it is not degenerate.')
    print('If the conic is not degenerate, we can use the 2x2 minor: A_33')
    print('det(A_Q) = %s' % str(np.linalg.det(A_Q)))
    assert np.linalg.det(A_Q) != 0, 'degenerate conic'
    A_33 = np.array(((    A, B / 2),
                     (B / 2,     C)))
    utool.horiz_print('A_33 = ', A_33)

    #-----------------------
    # CONIC CLASSIFICATION
    #-----------------------
    print('----------------------------------')
    print('The determinant of the minor classifies the type of conic it is')
    print('(det == 0): parabola, (det < 0): hyperbola, (det > 0): ellipse')
    print('det(A_33) = %s' % str(np.linalg.det(A_33)))
    assert np.linalg.det(A_33) > 0, 'conic is not an ellipse'
    print('... this is indeed an ellipse')

    #-----------------------
    # CONIC CENTER
    #-----------------------
    print('----------------------------------')
    print('the centers of the ellipse are obtained by: ')
    print('x_center = (B * E - (2 * C * D)) / (4 * A * C - B ** 2)')
    print('y_center = (D * B - (2 * A * E)) / (4 * A * C - B ** 2)')
    # Centers are obtained by solving for where the gradient of the quadratic
    # becomes 0. Without going through the derivation the calculation is...
    # These should be 0, 0 if we are at the origin, or our original x, y
    # coordinate specified by the keypoints. I'm doing the calculation just for
    # shits and giggles
    x_center = (B * E - (2 * C * D)) / (4 * A * C - B ** 2)
    y_center = (D * B - (2 * A * E)) / (4 * A * C - B ** 2)
    utool.horiz_print('x_center = ', x_center)
    utool.horiz_print('y_center = ', y_center)

    #-----------------------
    # MAJOR AND MINOR AXES
    #-----------------------
    # Now we are going to determine the major and minor axis
    # of this beast. It just the center augmented by the eigenvecs
    print('----------------------------------')
    # Plot ellipse axis
    # !HELP! I DO NOT KNOW WHY I HAVE TO DIVIDE, SQUARE ROOT, AND NEGATE!!!
    (evals, evecs) = np.linalg.eig(A_33)
    l1, l2 = evals
    # The major and minor axis lengths
    b = 1 / np.sqrt(l1)
    a = 1 / np.sqrt(l2)
    v1, v2 = evecs
    # Find the transformation to align the axis
    nminor = v1
    nmajor = v2
    dx1, dy1 = (v1 * b)
    dx2, dy2 = (v2 * a)
    minor = np.array([dx1, -dy1])
    major = np.array([dx2, -dy2])
    x_axis = np.array([[1], [0]])
    cosang = (x_axis.T.dot(nmajor)).T
    # Rotation angle
    theta = np.arccos(cosang)
    print('a = ' + str(a))
    print('b = ' + str(b))
    print('theta = ' + str(theta[0] / TAU) + ' * 2pi')
    # The warped eigenvects should have the same magintude
    # As the axis lengths
    assert utool.almost_eq(a, major.dot(ltool.rotation_mat2x2(theta))[0])
    assert utool.almost_eq(b, minor.dot(ltool.rotation_mat2x2(theta))[1])
    try:
        # HACK
        if len(theta) == 1:
            theta = theta[0]
    except Exception:
        pass

    #-----------------------
    # ECCENTRICITY
    #-----------------------
    print('----------------------------------')
    print('The eccentricity is determined by:')
    print('')
    print('            (2 * np.sqrt((A - C) ** 2 + B ** 2))     ')
    print('ecc = -----------------------------------------------')
    print('      (nu * (A + C) + np.sqrt((A - C) ** 2 + B ** 2))')
    print('')
    print('(nu is always 1 for ellipses)')
    nu = 1
    ecc_numer = (2 * np.sqrt((A - C) ** 2 + B ** 2))
    ecc_denom = (nu * (A + C) + np.sqrt((A - C) ** 2 + B ** 2))
    ecc = np.sqrt(ecc_numer / ecc_denom)
    print('ecc = ' + str(ecc))

    # Eccentricity is a little easier in axis aligned coordinates
    # Make sure they aggree
    ecc2 = np.sqrt(1 - (b ** 2) / (a ** 2))
    assert utool.almost_eq(ecc, ecc2)

    #-----------------------
    # APPROXIMATE UNIFORM SAMPLING
    #-----------------------
    # We are given the keypoint in invA format
    print('----------------------------------')
    print('Approximate uniform points an inscribed polygon bondary')

    #def next_xy(x, y, d):
    #    # http://gamedev.stackexchange.com/questions/1692/what-is-a-simple-algorithm-for-calculating-evenly-distributed-points-on-an-ellip
    #    num = (b ** 2) * (x ** 2)
    #    den = ((a ** 2) * ((a ** 2) - (x ** 2)))
    #    dxdenom = np.sqrt(1 + (num / den))
    #    deltax = d / dxdenom
    #    x_ = x + deltax
    #    y_ = b * np.sqrt(1 - (x_ ** 2) / (a ** 2))
    #    return x_, y_

    def xy_fn(t):
        return np.array((a * np.cos(t), b * np.sin(t))).T

    #nSamples = 16
    #(ix, iy, iv11, iv21, iv22), iv12 = kp, 0
    #invV = np.array([[iv11, iv12, ix],
    #                 [iv21, iv22, iy],
    #                 [   0,    0,  1]])
    #theta_list = np.linspace(0, TAU, nSamples)
    #cicrle_pts = np.array([(np.cos(t_), np.sin(t_), 1) for t_ in theta_list])
    uneven_points = invV.dot(cicrle_pts.T).T[:, 0:2]
    #uneven_points2 = xy_fn(theta_list)

    def circular_distance(arr):
        dist_most_ = ((arr[0:-1] - arr[1:]) ** 2).sum(1)
        dist_end_  = ((arr[-1] - arr[0]) ** 2).sum()
        return np.sqrt(np.hstack((dist_most_, dist_end_)))

    # Calculate the distance from each point on the ellipse to the next
    dists = circular_distance(uneven_points)
    total_dist = dists.sum()
    # Get an even step size
    multiplier = 1
    step_size = total_dist / (nSamples * multiplier)
    # Walk along edge
    num_steps_list = []
    offset_list = []
    dist_walked = 0
    total_dist = step_size
    for count in range(len(dists)):
        segment_len = dists[count]
        # Find where your starting location is
        offset_list.append(total_dist - dist_walked)
        # How far can you possibly go?
        total_dist += segment_len
        # How many steps can you take?
        num_steps = int((total_dist - dist_walked) // step_size)
        num_steps_list.append(num_steps)
        # Log how much further youve gotten
        dist_walked += (num_steps * step_size)
    #print('step_size = %r' % step_size)
    #print(np.vstack((num_steps_list, dists, offset_list)).T)

    # store the percent location at each line segment where
    # the cut will be made
    cut_list = []
    for num, dist, offset in zip(num_steps_list, dists, offset_list):
        if num == 0:
            cut_list.append([])
            continue
        offset1 = (step_size - offset) / dist
        offset2 = ((num * step_size) - offset) / dist
        cut_locs = (np.linspace(offset1, offset2, num, endpoint=True))
        cut_list.append(cut_locs)
        #print(cut_locs)

    # Cut the segments into new better segments
    approx_pts = []
    nPts = len(uneven_points)
    for count, cut_locs in enumerate(cut_list):
        for loc in cut_locs:
            pt1 = uneven_points[count]
            pt2 = uneven_points[(count + 1) % nPts]
            # Linearly interpolate between points
            new_loc = ((1 - loc) * pt1) + ((loc) * pt2)
            approx_pts.append(new_loc)
    approx_pts = np.array(approx_pts)

    # Warp approx_pts to the unit circle
    print('----------------------------------')
    print('For each aproximate point, find the closet point on the ellipse')
    #new_unit = V.dot(approx_pts.T).T
    ones_ = np.ones(len(approx_pts))
    new_hlocs = np.vstack((approx_pts.T, ones_))
    new_unit = V.dot(new_hlocs).T
    # normalize new_unit
    new_mag = np.sqrt((new_unit ** 2).sum(1))
    new_unorm_unit = new_unit / np.vstack([new_mag] * 3).T
    new_norm_unit = new_unorm_unit / np.vstack([new_unorm_unit[:, 2]] * 3).T
    # Get angle (might not be necessary)
    x_axis = np.array([1, 0, 0])
    arccos_list = x_axis.dot(new_norm_unit.T)
    uniform_theta_list = np.arccos(arccos_list)
    # Maybe this?
    uniform_theta_list = np.arctan2(new_norm_unit[:, 1], new_norm_unit[:, 0])
    #
    unevn_cicrle_pts = np.array([(np.cos(t_), np.sin(t_), 1) for t_ in uniform_theta_list])
    # This is the output. Approximately uniform points sampled along an ellipse
    uniform_ell_pts = invV.dot(unevn_cicrle_pts.T).T
    #uniform_ell_pts = invV.dot(new_norm_unit.T).T

    _plotpts(approx_pts, 0, df2.YELLOW, label='approx points', marker='o-')
    _plotpts(uniform_ell_pts, 0, df2.RED, label='uniform points', marker='o-')

    # Desired number of points
    #ecc = np.sqrt(1 - (b ** 2) / (a ** 2))
    # Total arclength
    #total_arclen = ellipeinc(TAU, ecc)
    #firstquad_arclen = total_arclen / 4
    # Desired arclength between points
    #d = firstquad_arclen / nSamples
    # Initial point
    #x, y = xy_fn(.001)
    #uniform_points = []
    #for count in range(nSamples):
    #    if np.isnan(x_) or np.isnan(y_):
    #        print('nan on count=%r' % count)
    #        break
    #    uniform_points.append((x_, y_))
    # The angle between the major axis and our x axis is:
    #-----------------------
    # DRAWING
    #-----------------------
    print('----------------------------------')
    # Draw the keypoint using the tried and true df2
    # Other things should subsiquently align
    #df2.draw_kpts2(np.array([kp]), ell_linewidth=4,
    #               ell_color=df2.DEEP_PINK, ell_alpha=1, arrow=True, rect=True)

    # Plot ellipse points
    _plotpts(ellipse_pts1, 0, df2.PURPLE, label='invV.dot(cicrle_pts.T).T', marker='x-')

    _plotarrow(x_center, y_center, dx1, -dy1, color=df2.GRAY, label='minor axis')
    _plotarrow(x_center, y_center, dx2, -dy2, color=df2.GRAY, label='major axis')

    # Rotate the ellipse so it is axis aligned and plot that
    rot = ltool.rotation_around_mat3x3(theta, ix, iy)
    ellipse_pts3 = rot.dot(ellipse_pts1.T).T
    #!_plotpts(ellipse_pts3, 0, df2.GREEN, label='axis aligned points')

    # Plot ellipse orientation
    ortho_basis = np.eye(3)[:, 0:2]
    orient_axis = invV.dot(ortho_basis)
    print(orient_axis)
    _dx1, _dx2, _dy1, _dy2, _1, _2 = orient_axis.flatten()
    #!_plotarrow(x_center, y_center, _dx1, _dy1, color=df2.BLUE, label='ellipse rotation')
    #!_plotarrow(x_center, y_center, _dx2, _dy2, color=df2.BLUE)

    #df2.plt.gca().set_xlim(400, 600)
    #df2.plt.gca().set_ylim(300, 500)

    xmin, ymin = ellipse_pts1.min(0)[0:2] - 1
    xmax, ymax = ellipse_pts1.max(0)[0:2] + 1
    df2.plt.gca().set_xlim(xmin, xmax)
    df2.plt.gca().set_ylim(ymin, ymax)
    df2.legend()
    df2.dark_background(doubleit=3)
    df2.gca().invert_yaxis()
    return locals()
    # Algebraic form of connic
    #assert (a * (x ** 2)) + (b * (x * y)) + (c * (y ** 2)) + (d * x) + (e * y) + (f) == 0


if __name__ == '__main__':
    """
    python pyhesaff/tests/test_ellipse.py
    """
    print('__main__ = test_ellipse.py')
    np.set_printoptions(threshold=5000, linewidth=5000, precision=3)

    test_data = pyhestest.load_test_data(short=True)
    kpts = test_data['kpts']
    kp = kpts[0]
    #kp = np.array([0, 0, 10, 10, 10, 0])
    print('Testing kp=%r' % (kp,))
    test_locals = in_depth_ellipse(kp)
    exec(utool.execstr_dict(test_locals, 'test_locals'))
    #if '--cmd' in sys.argv:
    #exec(helpers.execstr_dict(adaptive_locals, 'adaptive_locals'))
    #in_depth_locals = adaptive_locals['in_depth_locals']
    #exec(helpers.execstr_dict(in_depth_locals, 'in_depth_locals'))
    #exec(df2.present(override1=True))
    exec(df2.present())
