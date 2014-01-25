from __future__ import print_function, division
# Standard
#from ctypes.util import find_library
from os.path import join, exists, realpath, dirname, expanduser, split
import ctypes_interface
import ctypes as C
import sys
import collections
# Scientific
import numpy as np


# If profiling with kernprof.py
try:
    profile  # NOQA
except NameError:
    profile = lambda func: func


def ensure_hotspotter():
    import matplotlib
    matplotlib.use('Qt4Agg', warn=True, force=True)
    # Look for hotspotter in ~/code
    hotspotter_dir = join(expanduser('~'), 'code', 'hotspotter')
    if not exists(hotspotter_dir):
        print('[jon] hotspotter_dir=%r DOES NOT EXIST!' % (hotspotter_dir,))
    # Append hotspotter location (not dir) to PYTHON_PATH (i.e. sys.path)
    hotspotter_location = split(hotspotter_dir)[0]
    if not hotspotter_location in sys.path:
        sys.path.append(hotspotter_location)


#============================
# hesaff ctypes interface
#============================

# numpy dtypes
kpts_dtype = np.float32
desc_dtype = np.uint8
# ctypes
obj_t = C.c_void_p
kpts_t = np.ctypeslib.ndpointer(dtype=kpts_dtype, ndim=2, flags='aligned, c_contiguous, writeable')
ro_kpts_t = np.ctypeslib.ndpointer(dtype=kpts_dtype, ndim=2, flags='aligned, c_contiguous')
desc_t = np.ctypeslib.ndpointer(dtype=desc_dtype, ndim=2, flags='aligned, c_contiguous, writeable')
str_t = C.c_char_p
int_t = C.c_int
float_t = C.c_float

# THE ORDER OF THIS LIST IS IMPORTANT!
hesaff_typed_params = [
    # Pyramid Params
    (int_t,   'numberOfScales', 3),           # number of scale per octave
    (float_t, 'threshold', 16.0 / 3.0),       # noise dependent threshold on the response (sensitivity)
    (float_t, 'edgeEigenValueRatio', 10.0),   # ratio of the eigenvalues
    (int_t,   'border', 5),                   # number of pixels ignored at the border of image
    # Affine Shape Params
    (int_t,   'maxIterations', 16),           # number of affine shape interations
    (float_t, 'convergenceThreshold', 0.05),  # maximum deviation from isotropic shape at convergence
    (int_t,   'smmWindowSize', 19),           # width and height of the SMM mask
    (float_t, 'mrSize', 3.0 * np.sqrt(3.0)),  # size of the measurement region (as multiple of the feature scale)
    # SIFT params
    (int_t,   'spatialBins', 4),
    (int_t,   'orientationBins', 8),
    (float_t, 'maxBinValue', 0.2),
    # Shared params
    (float_t, 'initialSigma', 1.6),           # amount of smoothing applied to the initial level of first octave
    (int_t,   'patchSize', 41),               # width and height of the patch
    # My params
    (float_t, 'scale_min', -1.0),
    (float_t, 'scale_max', -1.0),
]

OrderedDict = collections.OrderedDict
hesaff_param_dict = OrderedDict([(key, val) for (type_, key, val) in hesaff_typed_params])
hesaff_param_types = [type_ for (type_, key, val) in hesaff_typed_params]


def load_hesaff_clib():
    '''
    Specificially loads the hesaff lib and defines its functions
    '''
    # Get the root directory which should have the dynamic library in it
    #root_dir = realpath(dirname(__file__)) if '__file__' in vars() else realpath(os.getcwd())
    root_dir = realpath(dirname(__file__))
    libname = 'hesaff'
    hesaff_lib, def_cfunc = ctypes_interface.load_clib(libname, root_dir)
    # Expose extern C Functions
    def_cfunc(int_t, 'detect',                 [obj_t])
    def_cfunc(None,  'exportArrays',           [obj_t, int_t, kpts_t, desc_t])
    def_cfunc(None,  'extractDesc',            [obj_t, int_t, kpts_t, desc_t])
    def_cfunc(obj_t, 'new_hesaff',             [str_t])
    def_cfunc(obj_t, 'new_hesaff_from_params', [str_t] + hesaff_param_types)
    return hesaff_lib

# Create a global interface to the hesaff lib
hesaff_lib = load_hesaff_clib()


#============================
# hesaff python interface
#============================


def new_hesaff(img_fpath, **kwargs):
    # Make detector and read image
    hesaff_params = hesaff_param_dict.copy()
    hesaff_params.update(kwargs)
    hesaff_args = hesaff_params.values()
    hesaff_ptr = hesaff_lib.new_hesaff_from_params(realpath(img_fpath), *hesaff_args)
    return hesaff_ptr


@profile
def detect_kpts(img_fpath, **kwargs):
    hesaff_ptr = new_hesaff(img_fpath, **kwargs)
    # Return the number of keypoints detected
    nKpts = hesaff_lib.detect(hesaff_ptr)
    # Allocate arrays
    kpts = np.empty((nKpts, 5), kpts_dtype)
    desc = np.empty((nKpts, 128), desc_dtype)
    # Populate arrays
    hesaff_lib.exportArrays(hesaff_ptr, nKpts, kpts, desc)
    return kpts, desc


def extract_desc(img_fpath, kpts, **kwargs):
    hesaff_ptr = new_hesaff(img_fpath, **kwargs)
    nKpts = len(kpts)
    # allocate memory for new descriptors
    desc = np.empty((nKpts, 128), desc_dtype)
    kpts = np.ascontiguousarray(kpts)  # kpts might not be contiguous
    # extract descriptors at given locations
    hesaff_lib.extractDesc(hesaff_ptr, nKpts, kpts, desc)
    return desc


@profile
def test_hesaff_kpts(img_fpath, **kwargs):
    # Make detector and read image
    hesaff_ptr = new_hesaff(img_fpath, **kwargs)
    # Return the number of keypoints detected
    nKpts = hesaff_lib.detect(hesaff_ptr)
    #print('[pyhesaff] detected: %r keypoints' % nKpts)
    # Allocate arrays
    kpts = np.empty((nKpts, 5), kpts_dtype)
    desc = np.empty((nKpts, 128), desc_dtype)
    # Populate arrays
    hesaff_lib.exportArrays(hesaff_ptr, nKpts, kpts, desc)
    # TODO: Incorporate parameters
    # TODO: Scale Factor
    #hesaff_lib.extractDesc(hesaff_ptr, nKpts, kpts, desc2)
    #hesaff_lib.extractDesc(hesaff_ptr, nKpts, kpts, desc3)
    #print('[hesafflib] returned')
    return kpts, desc


def expand_scales(kpts, nScales=10):
    scales = 2 ** np.linspace(-.5, .5, nScales)
    exkpts = []
    for scale in scales:
        kpts_ = kpts.copy()
        kpts_.T[2] *= scale
        kpts_.T[3] *= scale
        kpts_.T[4] *= scale
        exkpts.append(kpts_)
    return exkpts


def test_adaptive_scale(img_fpath):
    import cv2
    #from hotspotter import extract_patch
    n = 5
    img_fpath = realpath('zebra.png')
    kpts, desc = detect_kpts(img_fpath, scale_min=20, scale_max=100)
    zebra_fxs = [374, 520, 880]
    fxs = np.array(spaced_elements2(kpts, n).tolist() + zebra_fxs)
    sel_kpts = kpts[fxs]
    sel_desc = desc[fxs]

    imgBGR = cv2.imread(img_fpath, flags=cv2.IMREAD_COLOR)
    imgLAB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2LAB)
    imgL = imgLAB[:, :, 0]

    nScales = 5
    exkpts = expand_scales(sel_kpts, nScales)
    exkpts_ = np.vstack(exkpts)
    exdesc_ = extract_desc(img_fpath, exkpts_)

    kp = kpts[zebra_fxs[0]]
    rchip = imgL
    in_depth_locals = in_depth_ellipse(kp)

    #from hotspotter import interaction
    #interaction.rrr()
    #interaction.interact_keypoints(imgBGR, sel_kpts, sel_desc, fnum=3)
    ##interaction.interact_keypoints(imgBGR, kpts, desc, fnum=2)

    #draw_expanded_scales(imgBGR, sel_kpts, exkpts, exdesc_)
    return locals()


def in_depth_ellipse(kp):
    kp = np.array(kp, dtype=np.float64)
    print('kp = %r' % kp)
    #-----------------------
    # SETUP
    #-----------------------
    from hotspotter import draw_func2 as df2
    np.set_printoptions(precision=3)
    tau = 2 * np.pi
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
        arrow = df2.FancyArrow(x, y, dx, dy, **arrowargs)
        arrow.set_edgecolor(color)
        arrow.set_facecolor(color)
        ax.add_patch(arrow)
        df2.update()

    def almost_eq(a, b, thresh=1E-11):
        return abs(a - b) < thresh

    def rotation(theta):
        sin_ = np.sin(theta)
        cos_ = np.cos(theta)
        rot_ = np.array([[cos_, -sin_],
                         [sin_, cos_]])
        return rot_

    def rotation_around(theta, x, y):
        sin_ = np.sin(theta)
        cos_ = np.cos(theta)
        tr1_ = np.array([[1, 0, -x],
                         [0, 1, -y],
                         [0, 0, 1]])
        rot_ = np.array([[cos_, -sin_, 0],
                         [sin_, cos_,  0],
                         [   0,    0,  1]])
        tr2_ = np.array([[1, 0, x],
                         [0, 1, y],
                         [0, 0, 1]])
        rot = tr2_.dot(rot_).dot(tr1_)
        return rot

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
    (ix, iy, ia11, ia21, ia22), ia12 = kp, 0
    invV = np.array([[ia11, ia12, ix],
                     [ia21, ia22, iy],
                     [   0,    0,  1]])
    V = np.linalg.inv(invV)
    # <HACK>
    #invV = V / np.linalg.det(V)
    #V = np.linalg.inv(V)
    # </HACK>
    Z = (V.T).dot(V)

    print('invV is a transform from points on a unit-circle to the ellipse')
    helpers.horiz_print('invV = ', invV)
    print('--------------------------------')
    print('V is a transformation from points on the ellipse to a unit circle')
    helpers.horiz_print('V = ', V)
    print('--------------------------------')
    print('Points on a matrix satisfy (x_ - x_0).T.dot(Z).dot(x_ - x_0) = 1')
    print('where Z = (V.T).dot(V)')
    helpers.horiz_print('Z = ', Z)

    # Define points on a unit circle
    num_pts = 12
    theta_list = np.linspace(0, tau, num_pts)
    cicrle_pts = np.array([(np.cos(t_), np.sin(t_), 1) for t_ in theta_list])

    # Transform those points to the ellipse using invV
    ellipse_pts1 = invV.dot(cicrle_pts.T).T

    #Lets check our assertion: (x_ - x_0).T.dot(Z).dot(x_ - x_0) == 1
    x_0 = np.array([ix, iy, 1])
    checks = [(x_ - x_0).T.dot(Z).dot(x_ - x_0) for x_ in ellipse_pts1]
    try:
        # HELP: The phase is off here. in 3x3 version I'm not sure why
        #assert all([almost_eq(1, check) for check in checks1])
        is_almost_eq_pos1 = [almost_eq(1, check) for check in checks]
        is_almost_eq_neg1 = [almost_eq(-1, check) for check in checks]
        assert all(is_almost_eq_pos1)
    except AssertionError as ex:
        print('circle pts = %r ' % cicrle_pts)
        print(ex)
        print(checks)
        print([almost_eq(-1, check, 1E-9) for check in checks])
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
    helpers.horiz_print('A matrix A_Q = ', con)

    # A_Q is our conic section (aka ellipse matrix)
    A_Q = np.array(((    A, B / 2, D / 2),
                    (B / 2,     C, E / 2),
                    (D / 2, E / 2,     F)))

    helpers.horiz_print('A_Q = ', A_Q)

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
    helpers.horiz_print('A_33 = ', A_33)

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
    helpers.horiz_print('x_center = ', x_center)
    helpers.horiz_print('y_center = ', y_center)

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
    print('theta = ' + str(theta[0] / tau) + ' * 2pi')
    # The warped eigenvects should have the same magintude
    # As the axis lengths
    assert almost_eq(a, major.dot(rotation(theta))[0])
    assert almost_eq(b, minor.dot(rotation(theta))[1])

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
    assert almost_eq(ecc, ecc2)

    #-----------------------
    # APPROXIMATE UNIFORM SAMPLING
    #-----------------------
    from scipy.special import ellipeinc

    #def next_xy(x, y, d):
        ## http://gamedev.stackexchange.com/questions/1692/what-is-a-simple-algorithm-for-calculating-evenly-distributed-points-on-an-ellip
        #num = (b ** 2) * (x ** 2)
        #den = ((a ** 2) * ((a ** 2) - (x ** 2)))
        #dxdenom = np.sqrt(1 + (num / den))
        #deltax = d / dxdenom
        #x_ = x + deltax
        #y_ = b * np.sqrt(1 - (x_ ** 2) / (a ** 2))
        #return x_, y_

    #def xy_fn(t):
        #return np.array((a * np.cos(t), b * np.sin(t))).T

    #num_pts = 16
    #(ix, iy, ia11, ia21, ia22), ia12 = kp, 0
    #invV = np.array([[ia11, ia12, ix],
                     #[ia21, ia22, iy],
                     #[   0,    0,  1]])
    #theta_list = np.linspace(0, tau, num_pts)
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
    multiplier = 2
    step_size = total_dist / (num_pts * multiplier)
    # Walk along edge
    num_steps_list = []
    offset_list = []
    dist_walked = 0
    total_dist = step_size
    for count in xrange(len(dists)):
        segment_len = dists[count]
        offset_list.append(total_dist - dist_walked)
        total_dist += segment_len
        # How many steps can you take?
        num_steps = int((total_dist - dist_walked) // step_size)
        num_steps_list.append(num_steps)
        # Log how much further youve gotten
        dist_walked += (num_steps * step_size)
        # Log how far you got by this timestep
    #print('step_size = %r' % step_size)
    #print(np.vstack((num_steps_list, dists, offset_list)).T)

    # store the percent location at each line segment where
    # the cut will be made
    cut_list = []
    total_steps = 0
    total_dist = 0
    residual = 0
    prev_res = 0
    from itertools import izip
    #num, dist, dist_walked  = iter_.next()
    for num, dist, offset in zip(num_steps_list, dists, offset_list):
        if num == 0:
            cut_list.append([])
            continue
        offset1 = (step_size - offset) / dist
        offset2 = ((num * step_size) - offset) / dist
        cut_locs = (np.linspace(offset1, offset2, num, endpoint=True))
        cut_list.append(cut_locs)
        print(cut_locs)

    # Cut the segments into new better segments
    new_locations = []
    nPts = len(uneven_points)
    for count, cut_locs in enumerate(cut_list):
        for loc in cut_locs:
            pt1 = uneven_points[count]
            pt2 = uneven_points[(count + 1) % nPts]
            # Linearly interpolate between points
            new_loc = ((1 - loc) * pt1) + ((loc) * pt2)
            new_locations.append(new_loc)
    new_locations = np.array(new_locations)
    # FIXME MOVE
    _plotpts(ellipse_pts1, 0, df2.PURPLE, label='invV.dot(cicrle_pts.T).T', marker='x-')

    _plotpts(new_locations, 0, df2.YELLOW, label='new points', marker='.-')
    _plotpts(np.array([new_locations[0]]), 0, df2.RED, label='first points')
    _plotpts(np.array([new_locations[1]]), 0, df2.PINK, label='first points')

    # Desired number of points
    ecc = np.sqrt(1 - (b ** 2) / (a ** 2))
    # Total arclength
    total_arclen = ellipeinc(tau, ecc)
    firstquad_arclen = total_arclen / 4
    # Desired arclength between points
    d = firstquad_arclen / num_pts
    # Initial point
    #x, y = xy_fn(.001)
    #uniform_points = []
    #for count in xrange(num_pts):
        #x_, y_ = next_xy(x, y, d)
        #if np.isnan(x_) or np.isnan(y_):
            #print('nan on count=%r' % count)
            #break
        #uniform_points.append((x_, y_))

    # The angle between the major axis and our x axis is:

    #-----------------------
    # DRAWING
    #-----------------------
    print('----------------------------------')
    # Draw the keypoint using the tried and true df2
    # Other things should subsiquently align
    #df2.draw_kpts2(np.array([kp]), ell_linewidth=4,
                   #ell_color=df2.DEEP_PINK, ell_alpha=1, arrow=True, rect=True)

    # Plot ellipse points

    _plotarrow(x_center, y_center, dx1, -dy1, color=df2.GRAY, label='minor axis')
    _plotarrow(x_center, y_center, dx2, -dy2, color=df2.GRAY, label='major axis')

    # Rotate the ellipse so it is axis aligned and plot that
    rot = rotation_around(theta, ix, iy)
    ellipse_pts3 = rot.dot(ellipse_pts1.T).T
    #!_plotpts(ellipse_pts3, 0, df2.GREEN, label='axis aligned points')

    # Plot ellipse orientation
    ortho_basis = np.eye(3)[:, 0:2]
    orient_axis = invV.dot(ortho_basis)
    print(orient_axis)
    _dx1, _dx2, _dy1, _dy2, _1, _2 = orient_axis.flatten()
    #!_plotarrow(x_center, y_center, _dx1, _dy1, color=df2.BLUE, label='ellipse rotation')
    #!_plotarrow(x_center, y_center, _dx2, _dy2, color=df2.BLUE)

    df2.plt.gca().set_xlim(400, 600)
    df2.plt.gca().set_ylim(300, 500)
    df2.legend()
    df2.dark_background(doubleit=3)
    df2.gca().invert_yaxis()
    return locals()
    # Algebraic form of connic
    #assert (a * (x ** 2)) + (b * (x * y)) + (c * (y ** 2)) + (d * x) + (e * y) + (f) == 0


def draw_expanded_scales(imgL, sel_kpts, exkpts, exdesc_):
    from hotspotter import extract_patch
    draw_keypoint_patch = extract_patch.draw_keypoint_patch
    get_warped_patch = extract_patch.get_warped_patch  # NOQA
    from hotspotter import draw_func2 as df2
    from hotspotter import vizualizations as viz

    # Rows are for different scales
    # Cols are for different patches
    # There is a prefix row showing the original image

    nRows, nCols = len(exkpts), len(sel_kpts)
    exkpts_ = np.vstack(exkpts)

    fnum = 1
    df2.figure(fnum=fnum, docla=True, doclf=True)

    nPreRows = 1
    nPreCols = (nPreRows * nCols) + 1

    viz.show_keypoints(imgL, exkpts_, fnum=fnum, pnum=(nRows + nPreRows, 1, 1),
                       color=df2.BLUE)

    px = 0
    for row, kpts_ in enumerate(exkpts):
        for col, kp in enumerate(kpts_):
            sift = exdesc_[px]
            pnum = (nRows + nPreRows, nCols, px + nPreCols)
            draw_keypoint_patch(imgL, kp, sift, warped=True, fnum=fnum, pnum=pnum)
            #exchip, wkp = extract_patch.get_warped_patch(imgL, kp)
            px += 1
    df2.draw()

    print('nRows = %r' % nRows)
    print('nCols = %r' % nCols)


def adaptive_scale(img_fpath, kpts, desc=None):
    #for channel in xrange(3):
        #import matplotlib.pyplot as plt
        #img = imgLAB[:, :, channel]

        #laplace = cv2.Laplacian(img, cv2.CV_64F)
        #sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        #sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

        #def _imshow(img, name, pnum):
            #plt.subplot(*pnum)
            #plt.imshow(img, cmap='gray')
            #plt.title(name)
            #plt.xticks([])
            #plt.yticks([])

        #plt.figure(channel)
        #_imshow(img,     'Original', (2, 2, 1))
        #_imshow(laplace, 'Laplace',  (2, 2, 2))
        #_imshow(sobelx,  'Sobel X',  (2, 2, 3))
        #_imshow(sobely,  'Sobel Y',  (2, 2, 4))
    #plt.show()

    return kpts, desc


def spaced_elements(list_, n):
    if n is None:
        return 'list'
    indexes = np.arange(len(list_))
    stride = len(indexes) // n
    return list_[indexes[0:-1:stride]]


def spaced_elements2(list_, n):
    if n is None:
        return np.arange(len(list_))
    indexes = np.arange(len(list_))
    stride = len(indexes) // n
    return indexes[0:-1:stride]

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    ensure_hotspotter()
    from hotspotter import fileio as io
    from hotspotter import draw_func2 as df2
    from hotspotter import helpers
    np.set_printoptions(threshold=5000, linewidth=5000, precision=3)

    # Read Image
    img_fpath = realpath('zebra.png')
    image = io.imread(img_fpath)

    def test_hesaff(n=None, fnum=1, **kwargs):
        from hotspotter import interaction
        reextract = kwargs.get('reextrac', False)
        new_exe = kwargs.get('new_exe', False)
        old_exe = kwargs.get('old_exe', False)
        adaptive = kwargs.get('adaptive', False)
        use_exe = new_exe or old_exe

        if use_exe:
            import pyhesaffexe
            if new_exe:
                pyhesaffexe.EXE_FPATH = pyhesaffexe.find_hesaff_fpath(exe_name='hesaffexe')
            if old_exe:
                pyhesaffexe.EXE_FPATH = pyhesaffexe.find_hesaff_fpath(exe_name='hesaff')

        print('[test]---------------------')
        try:
            # Select kpts
            title = split(pyhesaffexe.EXE_FPATH)[1] if use_exe else 'libhesaff'
            detect_func = pyhesaffexe.detect_kpts if use_exe else detect_kpts
            with helpers.Timer(msg=title):
                kpts, desc = detect_func(img_fpath, scale_min=0, scale_max=1000)
            if reextract:
                title = 'reextract'
                with helpers.Timer(msg='reextract'):
                    desc = extract_desc(img_fpath, kpts)
            kpts_ = kpts if n is None else spaced_elements(kpts, n)
            desc_ = desc if n is None else spaced_elements(desc, n)
            if adaptive:
                kpts_, desc_ = adaptive_scale(img_fpath, kpts_, desc_)
            # Print info
            print('detected %d keypoints' % len(kpts))
            print('drawing %d/%d kpts' % (len(kpts_), len(kpts)))
            title += ' ' + str(len(kpts))
            #print(kpts_)
            #print(desc_[:, 0:16])
            # Draw kpts
            interaction.interact_keypoints(image, kpts_, desc_, fnum, nodraw=True)
            df2.set_figtitle(title)
            #df2.imshow(image, fnum=fnum)
            #df2.draw_kpts2(kpts_, ell_alpha=.9, ell_linewidth=4,
                           #ell_color='distinct', arrow=True, rect=True)
        except Exception as ex:
            import traceback
            traceback.format_exc()
            print('EXCEPTION! ' + repr(ex))
            raise
        print('[test]---------------------')
        return locals()

    #n = None
    n = 5
    fnum = 1
    #test_locals = test_hesaff(n, fnum, adaptive=True)
    adaptive_locals = test_adaptive_scale(img_fpath)
    # They seem to work
    #test_locals = test_hesaff(n, fnum + 2, new_exe=True)
    #test_locals = test_hesaff(n, fnum + 3, old_exe=True)
    #test_locals = test_hesaff(n, fnum + 1, use_exe=True, reextract=True)

    #exec(helpers.execstr_dict(test_locals, 'test_locals'))
    if '--cmd' in sys.argv:
        in_depth_locals = adaptive_locals['in_depth_locals']
        exec(helpers.execstr_dict(adaptive_locals, 'adaptive_locals'))
        exec(helpers.execstr_dict(in_depth_locals, 'in_depth_locals'))
    exec(df2.present(override1=True))
    #exec(df2.present())
