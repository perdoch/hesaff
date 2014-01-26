'This module should handle all things elliptical'
from __future__ import print_function, division
# Scientific
import numpy as np
from numpy.core.umath_tests import matrix_multiply
import cv2
# Hotspotter
from hotspotter import __common__
print, print_, print_on, print_off, rrr, profile, printDBG = \
        __common__.init(__name__, module_prefix='[ell]', DEBUG=False, initmpl=False)
from hotspotter import extract_patch
from hotspotter import draw_func2 as df2
from hotspotter import vizualizations as viz
from hotspotter import interaction

def load_test_data():
    import shelve
    shelf = shelve.open('testdata.shelve')
    if not 'kpts' in shelf:
        import pyhesaff
        testdata = pyhesaff.load_test_data()
        shelf['kpts'] = testdata['kpts']
    kpts = shelf['kpts']
    exec(open('ellipse.py').read())


def sample_keypoint_border(imgBGR, exkpts):
    exkpts_ = np.vstack(exkpts)
    uniform_pts = sample_uniform(exkpts_)
    sample_points(imgBGR, uniform_pts)


def sample_points(imgBGR, uniform_pts):
    pass


from numpy.linalg import lapack_lite
from numpy import array, zeros
lapack_routine = lapack_lite.dgesv
# Looking one step deeper, we see that solve performs many sanity checks.
# Stripping these, we have:
def faster_inverse(A):
    #http://stackoverflow.com/questions/11972102/
    #is-there-a-way-to-efficiently-invert-an-array-of-matrices-with-numpy
    b = np.identity(A.shape[2], dtype=A.dtype)
    n_eq = A.shape[1]
    n_rhs = A.shape[2]
    pivots = zeros(n_eq, np.intc)
    identity  = np.eye(n_eq)
    def lapack_inverse(a):
        b = np.copy(identity)
        pivots = zeros(n_eq, np.intc)
        results = lapack_lite.dgesv(n_eq, n_rhs, a, n_eq, pivots, b, n_eq, 0)
        if results['info'] > 0:
            raise LinAlgError('Singular matrix')
        return b
    return array([lapack_inverse(a) for a in A])


@profile
def kpts_matrix(kpts):
    # We are given the keypoint in invA format
    # invV = perdoch.invA
    #    V = perdoch.A
    #    Z = perdoch.E
    nKp = len(kpts)
    (ia13, ia23, ia11, ia21, ia22) = kpts.T
    ia12 = np.zeros(nKp)
    ia31 = np.zeros(nKp)
    ia32 = np.zeros(nKp)
    ia33 = np.ones(nKp)
    # np.dot operates over the -1 and -2 axis of arrays
    # Start with:
    #     invV.shape = (3, 3, nKp)
    invV = np.array([[ia11, ia12, ia13],
                     [ia21, ia22, ia23],
                     [ia31, ia32, ia33]])
    # And roll into:
    #     invV.shape = (nKp, 3, 3)
    invV = np.rollaxis(invV, 2)
    invV = np.ascontiguousarray(invV)
    # invert into V
    #     V.shape = (nKp, 3, 3)
    V = np.linalg.inv(invV)
    #V = faster_inverse(invV)
    # transform into conic matrix Z
    # Z = (V.T).dot(V)
    Vt = array(map(np.transpose, V))
    #     Z.shape = (nKp, 3, 3)
    Z = matrix_multiply(Vt, V)
    return invV, V, Z

def homogenous_circle_pts(nSamples):
    # Make a list of homogenous circle points
    theta_list = np.linspace(0, tau, nSamples)
    cicrle_pts = np.array([(np.cos(t_), np.sin(t_), 1) for t_ in theta_list])
    return cicrle_pts


@profile
def sample_uniform(kpts, nSamples=128):
    nKp = len(kpts)
    invV, V, Z = kpts_matrix(kpts)
    tau = 2 * np.pi
    circle_pts = homogenous_circle_pts(nSamples)
    # assert circle_pts.shape == (nSamples, 3)
    uneven_pts = array([v.dot(circle_pts.T).T for v in invV])
    # assert uneven_pts.shape == (nKp, nSamples, 3)
    # The transformed points are not sampled uniformly... Bummer
    # We will sample points evenly across the sampled polygon
    # then we will project them onto the ellipse
    dists = array([circular_distance(arr) for arr in uneven_pts])
    # assert dists.shape == (nKp, nSamples)
    # perimeter of the polygon
    perimeter = dists.sum(1)
    # assert perimeter.shape == (nKp,)
    # Take a perfect multiple of steps along the perimeter
    multiplier = 1
    step_size = perimeter / (nSamples * multiplier)
    # assert step_size.shape == (nKp,)
    # Walk along edge
    num_steps_list = []
    offset_list = []
    total_dist = np.zeros(step_size.shape)  # step_size.copy()
    dist_walked = np.zeros(step_size.shape)
    # assert dist_walked.shape == (nKp,)
    # assert total_dist.shape == (nKp,)
    distsT = dists.T
    # assert distsT.shape == (nSamples, nKp)

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
    # Check for floating point errors:
    # take an extra step if you need to
    num_steps_list[-1] += np.round((perimeter - dist_walked) / step_size)
    #assert np.all(np.array(num_steps_list).sum(0) == nSample)

    # store the percent location at each line segment where
    # the cut will be made
    # HERE IS NEXT:
    cut_list = []
    # This loops over the pt samples and performs the operation for every keypoint
    for num, dist, offset in izip(num_steps_list, distsT, offset_list):
        #if num == 0:
            #cut_list.append([])
            #continue
        # This was a bitch to keep track of
        offset1 = (step_size - offset) / dist
        offset2 = ((num * step_size) - offset) / dist
        cut_locs = [np.linspace(off1, off2, n, endpoint=True) for (off1, off2, n) in izip(offset1, offset2, num)]
        # post check for divide by 0
        cut_locs = [array([0 if np.isinf(c) else c for c in cut]) for cut in cut_locs]
        cut_list.append(cut_locs)
    cut_list = np.array(cut_list).T
    # assert cut_list.shape == (nKp, nSample)

    # Linearly interpolate between points on the polygons at the places we cut
    def interpolate(loc, uneven, count):
        return ((1 - loc) * uneven[count]) + ((loc) * uneven[(count + 1) % nSamples])
    new_locations = np.array([array([interpolate(loc, uneven, count)
                                for count, locs in enumerate(cuts)
                                for loc in iter(locs)]) for uneven, cuts in izip(uneven_pts, cut_list)])

    # assert new_locations.shape == (nKp, nSamples, 3)
    # Warp new_locations to the unit circle
    #new_unit = V.dot(new_locations.T).T
    new_unit = array([v.dot(newloc.T).T for v, newloc in izip(V, new_locations)])
    # normalize new_unit
    new_mag = np.sqrt((new_unit ** 2).sum(-1))
    new_unorm_unit = new_unit / np.dstack([new_mag]*3)
    new_norm_unit = new_unorm_unit / np.dstack([new_unorm_unit[:, :, 2]]*3)
    # Get angle (might not be necessary)
    #x_axis = np.array([1, 0, 0])
    #arccos_list = x_axis.dot(new_norm_unit.T)
    #uniform_theta_list = np.arccos(arccos_list)
    # Maybe this?
    # Find the angle from the center of the circle
    theta_list2 = np.arctan2(new_norm_unit[:, :, 1], new_norm_unit[:, :, 0])
    # assert uniform_theta_list.shape = (nKp, nSample)
    # Use this angle to unevenly sample the perimeter of the circle
    uneven_cicrle_pts = np.dstack([np.cos(theta_list2), np.sin(theta_list2), np.ones(theta_list2.shape)])
    # The uneven circle points were sampled in such a way that when they are
    # transformeed they will be approximately uniform along the boundary of the
    # ellipse.
    uniform_ell_pts = [v.dot(pts.T).T for (v, pts) in izip(invV, uneven_cicrle_pts)]
    return uniform_ell_pts


def adaptive_scale(imgBGR, exkpts):
    border_intensity = sample_keypoint_border(imgBGR, exkpts)
    #exdesc_ = extract_desc(img_fpath, exkpts_)

    #kp = kpts[zebra_fxs[0]]
    #rchip = imgL
    #
    #interaction.rrr()
    #interaction.interact_keypoints(imgBGR, sel_kpts, sel_desc, fnum=3)
    ##interaction.interact_keypoints(imgBGR, kpts, desc, fnum=2)

    #draw_expanded_scales(imgBGR, sel_kpts, exkpts, exdesc_)

    #from hotspotter import interaction
    #interaction.rrr()
    #interaction.interact_keypoints(imgBGR, sel_kpts, sel_desc, fnum=3)
    ##interaction.interact_keypoints(imgBGR, kpts, desc, fnum=2)

    #draw_expanded_scales(imgBGR, sel_kpts, exkpts, exdesc_)

    #from hotspotter import interaction
    #interaction.rrr()
    #interaction.interact_keypoints(imgBGR, sel_kpts, sel_desc, fnum=3)
    ##interaction.interact_keypoints(imgBGR, kpts, desc, fnum=2)

    #draw_expanded_scales(imgBGR, sel_kpts, exkpts, exdesc_)


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

#----------------
# Numeric Helpers
#----------------

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