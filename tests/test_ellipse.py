#!/usr/bin/env python
import numpy as np


TAU = 2 * np.pi  # References: tauday.com


def draw_expanded_scales(imgL, sel_kpts, exkpts, exdesc_):
    import plottool as pt
    draw_keypoint_patch = pt.draw_keypoint_patch
    get_warped_patch = pt.get_warped_patch  # NOQA

    # Rows are for different scales
    # Cols are for different patches
    # There is a prefix row showing the original image
    nRows, nCols = len(exkpts), len(sel_kpts)
    exkpts_ = np.vstack(exkpts)

    fnum = 1
    pt.figure(fnum=fnum, docla=True, doclf=True)

    nPreRows = 1
    nPreCols = (nPreRows * nCols) + 1

    pt.show_keypoints(imgL, exkpts_, fnum=fnum, pnum=(nRows + nPreRows, 1, 1),
                      color=pt.BLUE)

    px = 0
    for row, kpts_ in enumerate(exkpts):
        for col, kp in enumerate(kpts_):
            sift = exdesc_[px]
            pnum = (nRows + nPreRows, nCols, px + nPreCols)
            draw_keypoint_patch(imgL, kp, sift, warped=True, fnum=fnum, pnum=pnum)
            px += 1
    #df2.draw()

    print('nRows = %r' % nRows)
    print('nCols = %r' % nCols)


def in_depth_ellipse(kp):
    """
    Makes sure that I understand how the ellipse is created form a keypoint
    representation. Walks through the steps I took in coming to an
    understanding.

    CommandLine:
        xdoctest ~/code/pyhesaff/tests/test_ellipse.py --show --num-samples=12

    Example:
        >>> # xdoctest: +REQUIRES(--show)
        >>> # xdoctest: +REQUIRES(module:kwimage)
        >>> # xdoctest: +REQUIRES(module:kwplot)
        >>> import kwimage
        >>> import pyhesaff
        >>> img = kwimage.grab_test_image('astro')
        >>> kpts, vecs = pyhesaff.detect_feats_in_image(img)
        >>> kp = kpts[0]
        >>> import kwplot
        >>> kwplot.autompl()
        >>> test_locals = in_depth_ellipse(kp)
        >>> kwplot.show_if_requested()
    """
    import matplotlib as mpl
    import plottool_ibeis as pt
    import vtool_ibeis.linalg as ltool
    import vtool_ibeis as vt
    import ubelt as ub
    #nSamples = 12
    nSamples = int(ub.argval('--num-samples', default=12))
    kp = np.array(kp, dtype=np.float64)

    kp = np.array([5, 6, 10.8,  3.431, 8.06, 0])
    #-----------------------
    # SETUP
    #-----------------------
    # np.set_printoptions(precision=3)
    #pt.reset()
    pt.figure(9003, docla=True, doclf=True)
    ax = pt.gca()
    ax.invert_yaxis()

    def _plotpts(data, px, color=pt.BLUE, label='', marker='.', **kwargs):
        #pt.figure(9003, docla=True, pnum=(1, 1, px))
        pt.plot2(data.T[0], data.T[1], marker, '', color=color, label=label, **kwargs)
        #pt.update()

    def _plotarrow(x, y, dx, dy, color=pt.BLUE, label=''):
        ax = pt.gca()
        arrowargs = dict(head_width=.5, length_includes_head=True, label=label)
        arrow = mpl.patches.FancyArrow(x, y, dx, dy, **arrowargs)
        arrow.set_edgecolor(color)
        arrow.set_facecolor(color)
        ax.add_patch(arrow)
        #pt.update()

    #-----------------------
    # INPUT
    #-----------------------
    print('kp = %s' % ub.repr2(kp, precision=3))
    print('--------------------------------')
    print('Let V = Perdoch.A')
    print('Let Z = Perdoch.E')
    print('Let invV = Perdoch.invA')
    print('--------------------------------')
    print("Input from Perdoch's detector: ")

    # We are given the keypoint in invA format
    if len(kp) == 5:
        (ix, iy, iv11, iv21, iv22) = kp
        iv12 = 0
    elif len(kp) == 6:
        (ix, iy, iv11, iv21, iv22, ori) = kp
        iv12 = 0
    invV = np.array([[iv11, iv12, ix],
                     [iv21, iv22, iy],
                     [   0,    0,  1]])
    V = np.linalg.inv(invV)
    Z = (V.T).dot(V)

    V_2x2 = V[0:2, 0:2]
    Z_2x2 = Z[0:2, 0:2]
    V_2x2_ = vt.decompose_Z_to_V_2x2(Z_2x2)
    assert np.all(np.isclose(V_2x2, V_2x2_))

    #C = np.linalg.cholesky(Z)
    #np.isclose(C.dot(C.T), Z)
    #Z

    print('invV is a transform from points on a unit-circle to the ellipse')
    print(ub.hzcat(['invV = ', str(invV)]))
    print('--------------------------------')
    print('V is a transformation from points on the ellipse to a unit circle')
    print(ub.hzcat(['V = ', str(V)]))
    print('--------------------------------')
    print('An ellipse is a special case of a conic. For any ellipse:')
    print('Points on the ellipse satisfy (x_ - x_0).T.dot(Z).dot(x_ - x_0) = 1')
    print('where Z = (V.T).dot(V)')
    print(ub.hzcat(['Z = ', str(Z)]))

    # Define points on a unit circle
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
        is_almost_eq_pos1 = [np.allclose(1, check) for check in checks]
        is_almost_eq_neg1 = [np.allclose(-1, check) for check in checks]
        assert all(is_almost_eq_pos1)
    except AssertionError as ex:
        print('circle pts = %r ' % cicrle_pts)
        print(ex)
        print(checks)
        print([np.allclose(-1, check) for check in checks])
        raise
    else:
        #assert all([abs(1 - check) < 1E-11 for check in checks2])
        print('... all of our plotted points satisfy this')

    #=======================
    # THE CONIC SECTION
    #=======================
    # All of this was from the Perdoch paper, now lets move into conic sections
    # We will use the notation from wikipedia
    # References:
    #     http://en.wikipedia.org/wiki/Conic_section
    #     http://en.wikipedia.org/wiki/Matrix_representation_of_conic_sections

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
    print(ub.hzcat(['A matrix A_Q = ', str(con)]))

    # A_Q is our conic section (aka ellipse matrix)
    A_Q = np.array(((    A, B / 2, D / 2),
                    (B / 2,     C, E / 2),
                    (D / 2, E / 2,     F)))

    print(ub.hzcat(['A_Q = ', str(A_Q)]))

    #-----------------------
    # DEGENERATE CONICS
    # References:
    #    http://individual.utoronto.ca/somody/quiz.html
    print('----------------------------------')
    print('As long as det(A_Q) != it is not degenerate.')
    print('If the conic is not degenerate, we can use the 2x2 minor: A_33')
    print('det(A_Q) = %s' % str(np.linalg.det(A_Q)))
    assert np.linalg.det(A_Q) != 0, 'degenerate conic'
    A_33 = np.array(((    A, B / 2),
                     (B / 2,     C)))
    print(ub.hzcat(['A_33 = ', str(A_33)]))

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
    print(ub.hzcat(['x_center = ', str(x_center)]))
    print(ub.hzcat(['y_center = ', str(y_center)]))

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

    try:
        assert np.allclose(a, major.dot(ltool.rotation_mat2x2(theta))[0])
        assert np.allclose(b, minor.dot(ltool.rotation_mat2x2(theta))[1])
    except AssertionError:
        print('WARNING: warped eigenvects do not have same magintude as axis length')

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
    try:
        assert np.allclose(ecc, ecc2), 'ecc does not aggree!'
    except Exception as ex:
        print(f'WARNING ex={ex}')

    #-----------------------
    # APPROXIMATE UNIFORM SAMPLING
    #-----------------------
    # We are given the keypoint in invA format
    print('----------------------------------')
    print('Approximate uniform points an inscribed polygon bondary')

    #def next_xy(x, y, d):
    #    # References:
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

    _plotpts(approx_pts, 0, pt.YELLOW, label='approx points', marker='o-')
    _plotpts(uniform_ell_pts, 0, pt.RED, label='uniform points', marker='o-')

    #### ALTERNATE METHOD
    import scipy.optimize
    import scipy as sp
    def angles_in_ellipse(num, a, b):
        """
        References:
            https://stackoverflow.com/questions/6972331/how-can-i-generate-a-set-of-points-evenly-distributed-along-the-perimeter-of-an
        """
        assert num > 0
        assert a < b
        angles = 2 * np.pi * np.arange(num) / num
        if a != b:
            e2 = (1.0 - a ** 2.0 / b ** 2.0)
            tot_size = sp.special.ellipeinc(2.0 * np.pi, e2)
            arc_size = tot_size / num
            arcs = np.arange(num) * arc_size
            res = sp.optimize.root(
                lambda x: (sp.special.ellipeinc(x, e2) - arcs), angles,
                options={'maxiter': 5}
            )
            angles = res.x
        return angles

    uniform_arclen_thetas = angles_in_ellipse(nSamples, a, b)
    import kwimage
    axis_aligned_optimized_uniform_pts = kwimage.Points(xy=xy_fn(uniform_arclen_thetas))
    optimized_uniform_pts = axis_aligned_optimized_uniform_pts.warp(
        kwimage.Affine.coerce(theta=theta, offset=(x_center, y_center)))

    _plotpts(optimized_uniform_pts.xy, 0, pt.GREEN, label='optimized', marker='o-')

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
    # Draw the keypoint using the tried and true pt
    # Other things should subsiquently align
    #pt.draw_kpts2(np.array([kp]), ell_linewidth=4,
    #               ell_color=pt.DEEP_PINK, ell_alpha=1, arrow=True, rect=True)

    # Plot ellipse points
    _plotpts(ellipse_pts1, 0, pt.LIGHT_BLUE, label='invV.dot(cicrle_pts.T).T', marker='o-')

    _plotarrow(x_center, y_center, dx1, -dy1, color=pt.GRAY, label='minor axis')
    _plotarrow(x_center, y_center, dx2, -dy2, color=pt.GRAY, label='major axis')

    # Rotate the ellipse so it is axis aligned and plot that
    rot = ltool.rotation_around_mat3x3(theta, ix, iy)
    ellipse_pts3 = rot.dot(ellipse_pts1.T).T
    #!_plotpts(ellipse_pts3, 0, pt.GREEN, label='axis aligned points')

    # Plot ellipse orientation
    ortho_basis = np.eye(3)[:, 0:2]
    orient_axis = invV.dot(ortho_basis)
    print(orient_axis)
    _dx1, _dx2, _dy1, _dy2, _1, _2 = orient_axis.flatten()
    #!_plotarrow(x_center, y_center, _dx1, _dy1, color=pt.BLUE, label='ellipse rotation')
    #!_plotarrow(x_center, y_center, _dx2, _dy2, color=pt.BLUE)

    #pt.plt.gca().set_xlim(400, 600)
    #pt.plt.gca().set_ylim(300, 500)

    xmin, ymin = ellipse_pts1.min(0)[0:2] - 1
    xmax, ymax = ellipse_pts1.max(0)[0:2] + 1
    pt.plt.gca().set_xlim(xmin, xmax)
    pt.plt.gca().set_ylim(ymin, ymax)
    pt.legend()
    pt.dark_background(doubleit=3)
    pt.gca().invert_yaxis()

    # Hack in another view
    # It seems like the even points are not actually that even.
    # there must be a bug

    pt.figure(fnum=9003 + 1, docla=True, doclf=True, pnum=(1, 3, 1))
    _plotpts(ellipse_pts1, 0, pt.LIGHT_BLUE, label='invV.dot(cicrle_pts.T).T', marker='o-', title='even')
    pt.plt.gca().set_xlim(xmin, xmax)
    pt.plt.gca().set_ylim(ymin, ymax)
    pt.dark_background(doubleit=3)
    pt.gca().invert_yaxis()
    pt.figure(fnum=9003 + 1, pnum=(1, 3, 2))

    _plotpts(approx_pts, 0, pt.YELLOW, label='approx points', marker='o-', title='approx')
    pt.plt.gca().set_xlim(xmin, xmax)
    pt.plt.gca().set_ylim(ymin, ymax)
    pt.dark_background(doubleit=3)
    pt.gca().invert_yaxis()

    pt.figure(fnum=9003 + 1, pnum=(1, 3, 3))
    _plotpts(uniform_ell_pts, 0, pt.RED, label='uniform points', marker='o-', title='uniform')
    pt.plt.gca().set_xlim(xmin, xmax)
    pt.plt.gca().set_ylim(ymin, ymax)
    pt.dark_background(doubleit=3)
    pt.gca().invert_yaxis()

    return locals()
    # Algebraic form of connic
    #assert (a * (x ** 2)) + (b * (x * y)) + (c * (y ** 2)) + (d * x) + (e * y) + (f) == 0


#def test_ellipse_main():
#    r"""
#    CommandLine:
#        python -m pyhesaff.tests.test_ellipse --test-test_ellipse_main
#        python -m pyhesaff.tests.test_ellipse --test-test_ellipse_main --show

#    Example:
#        >>> # ENABLE_DOCTEST
#        >>> from pyhesaff.tests.test_ellipse import *  # NOQA
#        >>> # build test data
#        >>> # execute function
#        >>> result = test_ellipse_main()
#        >>> # verify results
#        >>> print(result)
#    """
#    print('__main__ = test_ellipse.py')
#    np.set_printoptions(threshold=5000, linewidth=5000, precision=3)
#    import pyhesaff.tests.pyhestest as pyhestest
#    test_data = pyhestest.load_test_data(short=True)
#    kpts = test_data['kpts']
#    kp = kpts[0]
#    #kp = np.array([0, 0, 10, 10, 10, 0])
#    print('Testing kp=%r' % (kp,))
#    test_locals = in_depth_ellipse(kp)
#    exec(ut.execstr_dict(test_locals, 'test_locals'))
#    #if '--cmd' in sys.argv:
#    #exec(helpers.execstr_dict(adaptive_locals, 'adaptive_locals'))
#    #in_depth_locals = adaptive_locals['in_depth_locals']
#    #exec(helpers.execstr_dict(in_depth_locals, 'in_depth_locals'))
#    #exec(pt.present(override1=True))
#    if ut.show_was_requested():
#        exec(pt.present())


if __name__ == '__main__':
    """
    CommandLine:
        python -m pyhesaff.tests.test_ellipse
        python -m pyhesaff.tests.test_ellipse --allexamples
        python -m pyhesaff.tests.test_ellipse --allexamples --noface --nosrc
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
