#!/usr/bin/env python

"""
The python hessian affine keypoint module

Command Line:
    python -m pyhesaff detect_feats --show --siftPower=0.5 --maxBinValue=-1
    python -m pyhesaff detect_feats --show
    python -m pyhesaff detect_feats --show --siftPower=0.5,
"""
from __future__ import absolute_import, print_function, division, unicode_literals
import six
from six.moves import zip
from six.moves import range
from os.path import realpath, dirname
import ctypes as C
import numpy as np
import ubelt as ub
from collections import OrderedDict
try:
    from pyhesaff import ctypes_interface
except ValueError:
    import ctypes_interface

#============================
# hesaff ctypes interface
#============================

# numpy dtypes
kpts_dtype = np.float32
vecs_dtype = np.uint8
img_dtype  = np.uint8
img32_dtype  = np.float32
# scalar ctypes
obj_t     = C.c_void_p
str_t     = C.c_char_p
#if True or six.PY2:  # HACK ALWAYS ON
int_t     = C.c_int
#else:
#    raise NotImplementedError('PY3')
#if six.PY3:
#    int_t     = C.c_long
bool_t    = C.c_bool
float_t   = C.c_float
#byte_t    = C.c_char
# array ctypes
FLAGS_RW = str('aligned, c_contiguous, writeable')
FLAGS_RO = str('aligned, c_contiguous')
#FLAGS_RW = 'aligned, writeable'
kpts_t       = np.ctypeslib.ndpointer(dtype=kpts_dtype, ndim=2, flags=FLAGS_RW)
vecs_t       = np.ctypeslib.ndpointer(dtype=vecs_dtype, ndim=2, flags=FLAGS_RW)
img_t        = np.ctypeslib.ndpointer(dtype=img_dtype, ndim=3, flags=FLAGS_RO)
img32_t      = np.ctypeslib.ndpointer(dtype=img32_dtype, ndim=3, flags=FLAGS_RO)
kpts_array_t = np.ctypeslib.ndpointer(dtype=kpts_t, ndim=1, flags=FLAGS_RW)
vecs_array_t = np.ctypeslib.ndpointer(dtype=vecs_t, ndim=1, flags=FLAGS_RW)
int_array_t  = np.ctypeslib.ndpointer(dtype=int_t, ndim=1, flags=FLAGS_RW)
str_list_t   = C.POINTER(str_t)

# THE ORDER OF THIS LIST IS IMPORTANT!
HESAFF_TYPED_PARAMS = [
    # Pyramid Params
    (int_t,   'numberOfScales', 3),           # number of scale per octave
    (float_t, 'threshold', 16.0 / 3.0),       # noise dependent threshold on the response (sensitivity)
    (float_t, 'edgeEigenValueRatio', 10.0),   # ratio of the eigenvalues
    (int_t,   'border', 5),                   # number of pixels ignored at the border of image
    (int_t,   'maxPyramidLevels', -1),        # maximum number of pyramid divisions. -1 is no limit
    # Affine Shape Params
    (int_t,   'maxIterations', 16),           # number of affine shape interations
    (float_t, 'convergenceThreshold', 0.05),  # maximum deviation from isotropic shape at convergence
    (int_t,   'smmWindowSize', 19),           # width and height of the SMM (second moment matrix) mask
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
    (bool_t,  'rotation_invariance', False),
    (bool_t,  'augment_orientation', False),
    (float_t, 'ori_maxima_thresh', .8),
    (bool_t,  'affine_invariance', True),
    (bool_t,  'only_count', False),
    #
    (bool_t,  'use_dense', False),
    (int_t,   'dense_stride', 32),
    (float_t, 'siftPower', 1.0),
]

HESAFF_PARAM_DICT = OrderedDict([(key, val) for (type_, key, val) in HESAFF_TYPED_PARAMS])
HESAFF_PARAM_TYPES = [type_ for (type_, key, val) in HESAFF_TYPED_PARAMS]


def grab_test_imgpath(p):
    fpath = ub.grabdata('https://i.imgur.com/KXhKM72.png',
                        fname='astro.png',
                        hash_prefix='160b6e5989d2788c0296eac45b33e90fe612da23',
                        hasher='sha1')
    return fpath


def imread(fpath):
    import cv2
    return cv2.imread(fpath)


def _build_typed_params_kwargs_docstr_block(typed_params):
    r"""
    Args:
        typed_params (dict):

    CommandLine:
        python -m pyhesaff build_typed_params_docstr

    Example:
        >>> # DISABLE_DOCTEST
        >>> from pyhesaff._pyhesaff import *  # NOQA
        >>> typed_params = HESAFF_TYPED_PARAMS
        >>> result = build_typed_params_docstr(typed_params)
        >>> print(result)
    """
    kwargs_lines = []
    for tup in typed_params:
        type_, name, default = tup
        typestr = str(type_).replace('<class \'ctypes.c_', '').replace('\'>', '')
        line_fmtstr = '{name} ({typestr}): default={default}'
        line = line_fmtstr.format(name=name, typestr=typestr, default=default)
        kwargs_lines.append(line)
    kwargs_docstr_block = ('Kwargs:\n' + ub.indent('\n'.join(kwargs_lines), '    '))
    return ub.indent(kwargs_docstr_block, '    ')
hesaff_kwargs_docstr_block = _build_typed_params_kwargs_docstr_block(HESAFF_TYPED_PARAMS)


HESAFF_CLIB = None

def argparse_hesaff_params():
    alias_dict = {'affine_invariance': 'ai'}
    alias_dict = {'rotation_invariance': 'ri'}
    default_dict_ = get_hesaff_default_params()
    try:
        import utool as ut
        hesskw = ut.argparse_dict(default_dict_, alias_dict=alias_dict)
    except Exception as ex:
        print('ex = {!r}'.format(ex))
        return default_dict_
    return hesskw


def _load_hesaff_clib():
    """
    Specificially loads the hesaff lib and defines its functions
    """
    # Get the root directory which should have the dynamic library in it
    #root_dir = realpath(dirname(__file__)) if '__file__' in vars() else realpath(os.getcwd())

    # os.path.dirname(sys.executable)
    #if getattr(sys, 'frozen', False):
    #    # we are running in a |PyInstaller| bundle
    #     root_dir = realpath(sys._MEIPASS)
    #else:
    #    # we are running in a normal Python environment
    #    root_dir = realpath(dirname(__file__))
    root_dir = realpath(dirname(__file__))

    libname = 'hesaff'
    (clib, def_cfunc, lib_fpath) = ctypes_interface.load_clib(libname, root_dir)
    # Expose extern C Functions to hesaff's clib
    #def_cfunc(C.c_char_p, 'cmake_build_type',       [])
    #def_cfunc(None,  'free_char',       [C.c_char_p])
    def_cfunc(int_t, 'get_cpp_version',        [])
    def_cfunc(int_t, 'is_debug_mode',          [])
    def_cfunc(int_t, 'detect',                 [obj_t])
    def_cfunc(int_t, 'get_kpts_dim',           [])
    def_cfunc(int_t, 'get_desc_dim',           [])
    def_cfunc(None,  'exportArrays',           [obj_t, int_t, kpts_t, vecs_t])
    def_cfunc(None,  'extractDesc',            [obj_t, int_t, kpts_t, vecs_t])
    def_cfunc(None,  'extractPatches',         [obj_t, int_t, kpts_t, img32_t])
    def_cfunc(None,  'extractDescFromPatches', [int_t, int_t, int_t, img_t, vecs_t])
    def_cfunc(obj_t, 'new_hesaff_fpath',       [str_t] + HESAFF_PARAM_TYPES)
    def_cfunc(obj_t, 'new_hesaff_image',       [img_t, int_t, int_t, int_t] + HESAFF_PARAM_TYPES)
    def_cfunc(None,  'free_hesaff',            [obj_t])
    def_cfunc(obj_t, 'detectFeaturesListStep1',   [int_t, str_list_t] + HESAFF_PARAM_TYPES)
    def_cfunc(None,  'detectFeaturesListStep2',    [int_t, obj_t, int_array_t])
    def_cfunc(None,  'detectFeaturesListStep3',    [int_t, obj_t, int_array_t, int_array_t, kpts_t, vecs_t])
    return clib, lib_fpath

# Create a global interface to the hesaff lib
try:
    HESAFF_CLIB, __LIB_FPATH__ = _load_hesaff_clib()
except AttributeError as ex:
    print('Need to rebuild hesaff')
    raise

KPTS_DIM = HESAFF_CLIB.get_kpts_dim()
DESC_DIM = HESAFF_CLIB.get_desc_dim()


#============================
# helpers
#============================


def alloc_patches(nKpts, size=41):
    patches = np.empty((nKpts, size, size), np.float32)
    return patches


def alloc_vecs(nKpts):
    # array of bytes
    vecs = np.empty((nKpts, DESC_DIM), vecs_dtype)
    return vecs


def alloc_kpts(nKpts):
    # array of floats
    kpts = np.empty((nKpts, KPTS_DIM), kpts_dtype)
    #kpts = np.zeros((nKpts, KPTS_DIM), kpts_dtype) - 1.0  # array of floats
    return kpts


def _make_hesaff_cpp_params(kwargs):
    hesaff_params = HESAFF_PARAM_DICT.copy()
    for key, val in six.iteritems(kwargs):
        if key in hesaff_params:
            hesaff_params[key] = val
        else:
            print('[pyhesaff] WARNING: key=%r is not known' % key)
    return hesaff_params


def _cast_strlist_to_C(py_strlist):
    """
    Converts a python list of strings into a c array of strings

    References:
        http://stackoverflow.com/questions/3494598/pass-list-of-strings-ctypes
    """
    c_strarr = (str_t * len(py_strlist))()
    c_strarr[:] = py_strlist
    return c_strarr


def _new_fpath_hesaff(img_fpath, **kwargs):
    """ Creates new detector object which reads the image """
    hesaff_params = _make_hesaff_cpp_params(kwargs)
    hesaff_args = hesaff_params.values()  # pass all parameters to HESAFF_CLIB
    img_realpath = realpath(img_fpath)
    if six.PY3:
        # convert out of unicode
        img_realpath = img_realpath.encode('ascii')
    try:
        hesaff_ptr = HESAFF_CLIB.new_hesaff_fpath(img_realpath, *hesaff_args)
    except Exception as ex:
        msg = 'hesaff_ptr = HESAFF_CLIB.new_hesaff_fpath(img_realpath, *hesaff_args)',
        print('msg = {!r}'.format(msg))
        print('hesaff_args = {!r}'.format(hesaff_args))
        raise
    return hesaff_ptr


def _new_image_hesaff(img, **kwargs):
    """ Creates new detector object which reads the image """
    hesaff_params = _make_hesaff_cpp_params(kwargs)
    hesaff_args = hesaff_params.values()  # pass all parameters to HESAFF_CLIB
    rows, cols = img.shape[0:2]
    if len(img.shape) == 2:
        channels = 1
    else:
        channels = img.shape[2]
    try:
        hesaff_ptr = HESAFF_CLIB.new_hesaff_image(
            img, rows, cols, channels, *hesaff_args)
    except Exception as ex:
        msg = ('hesaff_ptr = '
               'HESAFF_CLIB.new_hesaff_image(img_realpath, *hesaff_args)')
        print('msg = {!r}'.format(msg))
        print('hesaff_args = {!r}'.format(hesaff_args))
        raise
    return hesaff_ptr


#============================
# hesaff python interface
#============================


def get_hesaff_default_params():
    return HESAFF_PARAM_DICT.copy()


def get_is_debug_mode():
    return HESAFF_CLIB.is_debug_mode()


def get_cpp_version():
    r"""
    Returns:
        int: cpp_version

    CommandLine:
        python -m pyhesaff get_cpp_version

    Example:
        >>> # ENABLE_DOCTEST
        >>> from pyhesaff._pyhesaff import *  # NOQA
        >>> cpp_version = get_cpp_version()
        >>> isdebug = get_is_debug_mode()
        >>> print('cpp_version = %r' % (cpp_version,))
        >>> print('isdebug = %r' % (isdebug,))
        >>> assert cpp_version == 4, 'cpp version mimatch'
    """

    #str_ptr = HESAFF_CLIB.cmake_build_type()
    # copy c string into python
    #pystr = C.c_char_p(str_ptr).value
    # need to free c string
    #HESAFF_CLIB.free_char(str_ptr)
    #print('pystr = %r' % (pystr,))
    #print('pystr = %s' % (pystr,))
    cpp_version = HESAFF_CLIB.get_cpp_version()
    return cpp_version


# full detection and extraction


def detect_feats(img_fpath, use_adaptive_scale=False, nogravity_hack=False, **kwargs):
    r"""
    driver function for detecting hessian affine keypoints from an image path.
    extra parameters can be passed to the hessian affine detector by using
    kwargs.

    Args:
        img_fpath (str): image file path on disk
        use_adaptive_scale (bool):
        nogravity_hack (bool):

    Kwargs:
        numberOfScales (int)         : default=3
        threshold (float)            : default=5.33333333333
        edgeEigenValueRatio (float)  : default=10.0
        border (int)                 : default=5
        maxIterations (int)          : default=16
        convergenceThreshold (float) : default=0.05
        smmWindowSize (int)          : default=19
        mrSize (float)               : default=5.19615242271
        spatialBins (int)            : default=4
        orientationBins (int)        : default=8
        maxBinValue (float)          : default=0.2
        initialSigma (float)         : default=1.6
        patchSize (int)              : default=41
        scale_min (float)            : default=-1.0
        scale_max (float)            : default=-1.0
        rotation_invariance (bool)   : default=False
        affine_invariance (bool)     : default=True

    Returns:
        tuple : (kpts, vecs)

    CommandLine:
        python -m pyhesaff detect_feats
        python -m pyhesaff detect_feats --show
        python -m pyhesaff detect_feats --show --fname star.png
        python -m pyhesaff detect_feats --show --fname zebra.png
        python -m pyhesaff detect_feats --show --fname astro.png
        python -m pyhesaff detect_feats --show --fname carl.jpg

        python -m pyhesaff detect_feats --show --fname astro.png --ri
        python -m pyhesaff detect_feats --show --fname astro.png --ai

        python -m pyhesaff detect_feats --show --fname easy1.png --no-ai
        python -m pyhesaff detect_feats --show --fname easy1.png --no-ai --numberOfScales=1 --verbose
        python -m pyhesaff detect_feats --show --fname easy1.png --no-ai --scale-max=100 --verbose
        python -m pyhesaff detect_feats --show --fname easy1.png --no-ai --scale-min=20 --verbose
        python -m pyhesaff detect_feats --show --fname easy1.png --no-ai --scale-min=100 --verbose
        python -m pyhesaff detect_feats --show --fname easy1.png --no-ai --scale-max=20 --verbose

        python -m vtool.test_constrained_matching visualize_matches --show
        python -m vtool.tests.dummy testdata_ratio_matches --show

        python -m pyhesaff detect_feats --show --fname easy1.png --ai \
            --verbose --scale-min=35 --scale-max=40

        python -m pyhesaff detect_feats --show --fname easy1.png --ai \
            --verbose --scale-min=35 --scale-max=40&
        python -m pyhesaff detect_feats --show --fname easy1.png --no-ai \
            --verbose --scale-min=35 --scale-max=40&
        python -m pyhesaff detect_feats --show --fname easy1.png --no-ai \
            --verbose --scale-max=40 --darken .5
        python -m pyhesaff detect_feats --show --fname easy1.png --no-ai \
            --verbose --scale-max=30 --darken .5
        python -m pyhesaff detect_feats --show --fname easy1.png --ai \
            --verbose --scale-max=30 --darken .5

        # DENSE KEYPOINTS
        python -m pyhesaff detect_feats --show --fname astro.png \
            --no-affine-invariance --numberOfScales=1 --maxPyramidLevels=1 \
            --use_dense --dense_stride=64
        python -m pyhesaff detect_feats --show --fname astro.png \
            --no-affine-invariance --numberOfScales=1 --maxPyramidLevels=1 \
            --use_dense --dense_stride=64 --rotation-invariance
        python -m pyhesaff detect_feats --show --fname astro.png \
            --affine-invariance --numberOfScales=1 --maxPyramidLevels=1 \
            --use_dense --dense_stride=64
        python -m pyhesaff detect_feats --show --fname astro.png \
            --no-affine-invariance --numberOfScales=3 \
            --maxPyramidLevels=2 --use_dense --dense_stride=32

        python -m pyhesaff detect_feats --show --only_count=False

    Example0:
        >>> # ENABLE_DOCTEST
        >>> # Test simple detect
        >>> from pyhesaff._pyhesaff import *  # NOQA
        >>> TAU = 2 * np.pi
        >>> img_fpath = grab_test_imgpath(ub.argval('--fname', default='astro.png'))
        >>> kwargs = argparse_hesaff_params()
        >>> print('kwargs = %r' % (kwargs,))
        >>> (kpts, vecs) = detect_feats(img_fpath, **kwargs)
        >>> # Show keypoints
        >>> # xdoctest: +REQUIRES(--show)
        >>> imgBGR = imread(img_fpath)
        >>> # take a random stample
        >>> frac = ub.argval('--frac', default=1.0)
        >>> print('frac = %r' % (frac,))
        >>> idxs = vecs[0:int(len(vecs) * frac)]
        >>> vecs, kpts = vecs[idxs], kpts[idxs]
        >>> default_showkw = dict(ori=False, ell=True, ell_linewidth=2,
        >>>                       ell_alpha=.4, ell_color='distinct')
        >>> print('default_showkw = %r' % (default_showkw,))
        >>> #showkw = ut.argparse_dict(default_showkw)
        >>> #import plottool as pt
        >>> #pt.interact_keypoints.ishow_keypoints(imgBGR, kpts, vecs, **showkw)
        >>> #pt.show_if_requested()
    """
    # Load image
    hesaff_ptr = _new_fpath_hesaff(img_fpath, **kwargs)
    # Get num detected
    nKpts = HESAFF_CLIB.detect(hesaff_ptr)
    # Allocate arrays
    kpts = alloc_kpts(nKpts)
    vecs = alloc_vecs(nKpts)
    # Populate arrays
    HESAFF_CLIB.exportArrays(hesaff_ptr, nKpts, kpts, vecs)
    HESAFF_CLIB.free_hesaff(hesaff_ptr)
    if use_adaptive_scale:  # Adapt scale if requested
        kpts, vecs = adapt_scale(img_fpath, kpts)
    if nogravity_hack:
        kpts, vecs = vtool_adapt_rotation(img_fpath, kpts)
    return kpts, vecs


def detect_feats2(img_or_fpath, **kwargs):
    """
    General way of detecting from either an fpath or ndarray

    Args:
        img_or_fpath (str or ndarray):  file path string

    Returns:
        tuple
    """
    if isinstance(img_or_fpath, six.string_types):
        fpath = img_or_fpath
        return detect_feats(fpath, **kwargs)
    else:
        img = img_or_fpath
        return detect_feats_in_image(img, **kwargs)


def detect_feats_list(image_paths_list, **kwargs):
    """
    Args:
        image_paths_list (list): A list of image paths

    Returns:
        tuple: (kpts_list, vecs_list) A tuple of lists of keypoints and decsriptors

    Kwargs:
        numberOfScales (int)         : default=3
        threshold (float)            : default=5.33333333333
        edgeEigenValueRatio (float)  : default=10.0
        border (int)                 : default=5
        maxIterations (int)          : default=16
        convergenceThreshold (float) : default=0.05
        smmWindowSize (int)          : default=19
        mrSize (float)               : default=5.19615242271
        spatialBins (int)            : default=4
        orientationBins (int)        : default=8
        maxBinValue (float)          : default=0.2
        initialSigma (float)         : default=1.6
        patchSize (int)              : default=41
        scale_min (float)            : default=-1.0
        scale_max (float)            : default=-1.0
        rotation_invariance (bool)   : default=False

    CommandLine:
        python -m pyhesaff._pyhesaff detect_feats_list --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from pyhesaff._pyhesaff import *  # NOQA
        >>> fpath = grab_test_imgpath('astro.png')
        >>> image_paths_list = [grab_test_imgpath('carl.jpg'), grab_test_imgpath('star.png'), fpath]
        >>> (kpts_list, vecs_list) = detect_feats_list(image_paths_list)
        >>> #print((kpts_list, vecs_list))
        >>> # Assert that the normal version agrees
        >>> serial_list = [detect_feats(fpath) for fpath in image_paths_list]
        >>> kpts_list2 = [c[0] for c in serial_list]
        >>> vecs_list2 = [c[1] for c in serial_list]
        >>> diff_kpts = [kpts - kpts2 for kpts, kpts2 in zip(kpts_list, kpts_list2)]
        >>> diff_vecs = [vecs - vecs2 for vecs, vecs2 in zip(vecs_list, vecs_list2)]
        >>> assert all([x.sum() == 0 for x in diff_kpts]), 'inconsistent results'
        >>> assert all([x.sum() == 0 for x in diff_vecs]), 'inconsistent results'
    """
    # Get Num Images
    num_imgs = len(image_paths_list)

    # Cast string list to C
    if six.PY2:
        realpaths_list = list(map(realpath, image_paths_list))
    if six.PY3:
        realpaths_list = [realpath(path).encode('ascii') for path in image_paths_list]

    c_strs = _cast_strlist_to_C(realpaths_list)

    # Get algorithm parameters
    hesaff_params = HESAFF_PARAM_DICT.copy()
    hesaff_params.update(kwargs)
    # pass all parameters to HESAFF_CLIB
    hesaff_args = hesaff_params.values()

    length_array = np.empty(num_imgs, dtype=int_t)
    detector_array = HESAFF_CLIB.detectFeaturesListStep1(num_imgs, c_strs, *hesaff_args)
    HESAFF_CLIB.detectFeaturesListStep2(num_imgs, detector_array, length_array)
    total_pts = length_array.sum()
    # allocate arrays
    total_num_arrays = num_imgs * total_pts
    flat_kpts_ptr = alloc_kpts(total_num_arrays)
    flat_vecs_ptr = alloc_vecs(total_num_arrays)
    # TODO: get this working
    offset_array = np.roll(length_array.cumsum(), 1).astype(int_t)
    # np.array([0] + length_array.cumsum().tolist()[0:-1], int_t)
    offset_array[0] = 0
    HESAFF_CLIB.detectFeaturesListStep3(num_imgs, detector_array,
                                         length_array, offset_array,
                                         flat_kpts_ptr, flat_vecs_ptr)

    # reshape into jagged arrays
    kpts_list = [flat_kpts_ptr[o:o + l] for o, l in zip(offset_array, length_array)]
    vecs_list = [flat_vecs_ptr[o:o + l] for o, l in zip(offset_array, length_array)]

    return kpts_list, vecs_list


def detect_feats_in_image(img, **kwargs):
    r"""
    Takes a preloaded image and detects keypoints and descriptors

    Args:
        img (ndarray[uint8_t, ndim=2]):  image data, should be in BGR or grayscale

    Returns:
        tuple: (kpts, vecs)

    CommandLine:
        python -m pyhesaff detect_feats_in_image --show
        python -m pyhesaff detect_feats_in_image --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from pyhesaff._pyhesaff import *  # NOQA
        >>> img_fpath = grab_test_imgpath('astro.png')
        >>> img = imread(img_fpath)
        >>> (kpts, vecs) = detect_feats_in_image(img)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import plottool as pt
        >>> pt.interact_keypoints.ishow_keypoints(img, kpts, vecs, ori=True,
        >>>                                       ell_alpha=.4, color='distinct')
        >>> pt.set_figtitle('Detect Kpts in Image')
        >>> pt.show_if_requested()
    """
    #Valid keyword arguments are: + str(HESAFF_PARAM_DICT.keys())
    hesaff_ptr = _new_image_hesaff(img, **kwargs)
    # Get num detected
    nKpts = HESAFF_CLIB.detect(hesaff_ptr)
    # Allocate arrays
    kpts = alloc_kpts(nKpts)
    vecs = alloc_vecs(nKpts)
    HESAFF_CLIB.exportArrays(hesaff_ptr, nKpts, kpts, vecs)  # Populate arrays
    HESAFF_CLIB.free_hesaff(hesaff_ptr)
    return kpts, vecs


def detect_num_feats_in_image(img, **kwargs):
    r"""
    Just quickly returns how many keypoints are in the image. Does not attempt
    to return or store the values.

    It is a good idea to turn off things like ai and ri here.

    Args:
        img (ndarray[uint8_t, ndim=2]):  image data

    Returns:
        int: nKpts

    ISSUE: there seems to be an inconsistency for jpgs between this and detect_feats

    CommandLine:
        python -m pyhesaff detect_num_feats_in_image:0 --show
        python -m pyhesaff detect_num_feats_in_image:1 --show
        python -m xdoctest pyhesaff detect_num_feats_in_image:0

    Example:
        >>> # ENABLE_DOCTEST
        >>> from pyhesaff._pyhesaff import *  # NOQA
        >>> img_fpath = grab_test_imgpath('zebra.png')
        >>> img = imread(img_fpath)
        >>> nKpts = detect_num_feats_in_image(img)
        >>> kpts, vecs = detect_feats_in_image(img)
        >>> #assert nKpts == len(kpts), 'inconsistency'
        >>> result = ('nKpts = %s' % (ub.repr2(nKpts),))
        >>> print(result)

    Example1:
        >>> # TIMEDOCTEST
        >>> from pyhesaff._pyhesaff import *  # NOQA
        >>> setup = ub.codeblock(
            '''
            import pyhesaff
            img_fpath = grab_test_imgpath('carl.jpg')
            img = imread(img_fpath)
            ''')
        >>> stmt_list = [
        >>>    'pyhesaff.detect_feats_in_image(img)',
        >>>    'pyhesaff.detect_num_feats_in_image(img, affine_invariance=False)',
        >>>    'pyhesaff.detect_num_feats_in_image(img)',
        >>> ]
        >>> iterations = 30
        >>> verbose = True
        >>> #ut.timeit_compare(stmt_list, setup=setup, iterations=iterations,
        >>> #                  verbose=verbose, assertsame=False)

    """
    # We dont need to find vectors at all here
    kwargs['only_count'] = True
    #kwargs['only_count'] = False
    #Valid keyword arguments are: + str(HESAFF_PARAM_DICT.keys())
    hesaff_ptr = _new_image_hesaff(img, **kwargs)
    # Get num detected
    nKpts = HESAFF_CLIB.detect(hesaff_ptr)
    HESAFF_CLIB.free_hesaff(hesaff_ptr)
    return nKpts


# just extraction


def extract_vecs(img_fpath, kpts, **kwargs):
    r"""
    Extract SIFT descriptors at keypoint locations

    Args:
        img_fpath (str):
        kpts (ndarray[float32_t, ndim=2]):  keypoints

    Returns:
        ndarray[uint8_t, ndim=2]: vecs -  descriptor vectors

    CommandLine:
        python -m pyhesaff extract_vecs:0
        python -m pyhesaff extract_vecs:1 --fname=astro.png
        python -m pyhesaff extract_vecs:1 --fname=patsy.jpg --show
        python -m pyhesaff extract_vecs:1 --fname=carl.jpg
        python -m pyhesaff extract_vecs:1 --fname=zebra.png

    Example:
        >>> # ENABLE_DOCTEST
        >>> from pyhesaff._pyhesaff import *  # NOQA
        >>> img_fpath = grab_test_imgpath('carl.jpg')
        >>> kpts = np.array([[20, 25, 5.21657705, -5.11095951, 24.1498699, 0],
        >>>                  [29, 25, 2.35508823, -5.11095952, 24.1498692, 0],
        >>>                  [30, 30, 12.2165705, 12.01909553, 10.5286992, 0],
        >>>                  [31, 29, 13.3555705, 17.63429554, 14.1040992, 0],
        >>>                  [32, 31, 16.0527005, 3.407312351, 11.7353722, 0]], dtype=np.float32)
        >>> vecs = extract_vecs(img_fpath, kpts)
        >>> result = 'vecs = {}'.format(vecs)
        >>> print(result)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from pyhesaff._pyhesaff import *  # NOQA
        >>> img_fpath = grab_test_imgpath(ub.argval('--fname', default='astro.png'))
        >>> # Extract original keypoints
        >>> kpts, vecs1 = detect_feats(img_fpath)
        >>> # Re-extract keypoints
        >>> vecs2 = extract_vecs(img_fpath, kpts)
        >>> # Descriptors should be the same
        >>> errors = (vecs1.astype(np.float) - vecs2.astype(np.float)).sum(axis=1)
        >>> errors_index = np.nonzero(errors)[0]
        >>> print('errors = %r' % (errors,))
        >>> print('errors_index = %r' % (errors_index,))
        >>> print('errors.sum() = %r' % (errors.sum(),))
        >>> # VISUALIZTION
        >>> # xdoctest: +REQUIRES(--show)
        >>> import plottool as pt
        >>> # Extract the underlying grayscale patches
        >>> img = imread(img_fpath)
        >>> patch_list = extract_patches(img, kpts)
        >>> pt.interact_keypoints.ishow_keypoints(img_fpath, kpts[errors_index], vecs1[errors_index], fnum=1)
        >>> ax = pt.draw_patches_and_sifts(patch_list[errors_index], vecs1[errors_index], pnum=(1, 2, 1), fnum=2)
        >>> ax.set_title('patch extracted')
        >>> ax = pt.draw_patches_and_sifts(patch_list[errors_index], vecs2[errors_index], pnum=(1, 2, 2), fnum=2)
        >>> ax.set_title('image extracted')
        >>> pt.set_figtitle('Error Keypoints')
        >>> pt.show_if_requested()
    """
    hesaff_ptr = _new_fpath_hesaff(img_fpath, **kwargs)
    nKpts = len(kpts)
    # allocate memory for new decsriptors
    vecs = alloc_vecs(nKpts)
    # kpts might not be contiguous
    kpts = np.ascontiguousarray(kpts)
    # extract decsriptors at given locations
    HESAFF_CLIB.extractDesc(hesaff_ptr, nKpts, kpts, vecs)
    HESAFF_CLIB.free_hesaff(hesaff_ptr)
    return vecs


def extract_patches(img_or_fpath, kpts, **kwargs):
    r"""
    Extract patches used to compute SIFT descriptors.

    Args:
        img_or_fpath (ndarray or str):
        kpts (ndarray[float32_t, ndim=2]):  keypoints

    CommandLine:
        python -m pyhesaff extract_patches:0 --show
        python -m pyhesaff extract_vecs:1 --fname=astro.png
        python -m pyhesaff extract_vecs:1 --fname=patsy.jpg --show
        python -m pyhesaff extract_vecs:1 --fname=carl.jpg
        python -m pyhesaff extract_vecs:1 --fname=zebra.png

    Example:
        >>> # ENABLE_DOCTEST
        >>> # xdoctest: +REQUIRES(module:vtool)
        >>> from pyhesaff._pyhesaff import *  # NOQA
        >>> import vtool as vt
        >>> kwargs = {}
        >>> img_fpath = grab_test_imgpath('carl.jpg')
        >>> img = imread(img_fpath)
        >>> img_or_fpath = img
        >>> kpts, vecs1 = detect_feats(img_fpath)
        >>> kpts = kpts[1::len(kpts) // 9]
        >>> vecs1 = vecs1[1::len(vecs1) // 9]
        >>> cpp_patch_list = extract_patches(img, kpts)
        >>> py_patch_list_ = np.array(vt.get_warped_patches(img_or_fpath, kpts, patch_size=41)[0])
        >>> py_patch_list = np.array(vt.convert_image_list_colorspace(py_patch_list_, 'gray'))
        >>> # xdoctest: +REQUIRES(--show)
        >>> import plottool as pt
        >>> ax = pt.draw_patches_and_sifts(cpp_patch_list, None, pnum=(1, 2, 1))
        >>> ax.set_title('C++ extracted')
        >>> ax = pt.draw_patches_and_sifts(py_patch_list, None, pnum=(1, 2, 2))
        >>> ax.set_title('Python extracted')
        >>> pt.show_if_requested()
    """
    if isinstance(img_or_fpath, six.string_types):
        hesaff_ptr = _new_fpath_hesaff(img_or_fpath, **kwargs)
    else:
        hesaff_ptr = _new_image_hesaff(img_or_fpath, **kwargs)
    nKpts = len(kpts)
    patch_list = alloc_patches(nKpts, 41)   # allocate memory for patches
    patch_list[:] = 0
    kpts = np.ascontiguousarray(kpts)  # kpts might not be contiguous
    # extract decsriptors at given locations
    HESAFF_CLIB.extractPatches(hesaff_ptr, nKpts, kpts, patch_list)
    HESAFF_CLIB.free_hesaff(hesaff_ptr)
    return patch_list


def extract_desc_from_patches(patch_list):
    r"""
    Careful about the way the patches are extracted here.

    Args:
        patch_list (ndarray[ndims=3]):

    CommandLine:
        python -m pyhesaff extract_desc_from_patches
        python -m pyhesaff extract_desc_from_patches  --show
        python -m pyhesaff extract_desc_from_patches:1 --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> # xdoctest: +REQUIRES(module:vtool)
        >>> from pyhesaff._pyhesaff import *  # NOQA
        >>> import vtool as vt
        >>> img_fpath = grab_test_imgpath(ub.argval('--fname', default='astro.png'))
        >>> # First extract keypoints normally
        >>> (orig_kpts_list, orig_vecs_list) = detect_feats(img_fpath)
        >>> # Take 9 keypoints
        >>> img = imread(img_fpath)
        >>> kpts_list = orig_kpts_list[1::len(orig_kpts_list) // 9]
        >>> vecs_list = orig_vecs_list[1::len(orig_vecs_list) // 9]
        >>> # Extract the underlying grayscale patches (using different patch_size)
        >>> patch_list_ = np.array(vt.get_warped_patches(img, kpts_list, patch_size=64)[0])
        >>> patch_list = np.array(vt.convert_image_list_colorspace(patch_list_, 'gray'))
        >>> # Extract descriptors from the patches
        >>> vecs_array = extract_desc_from_patches(patch_list)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from pyhesaff._pyhesaff import *  # NOQA
        >>> img_fpath = grab_test_imgpath(ub.argval('--fname', default='astro.png'))
        >>> # First extract keypoints normally
        >>> (orig_kpts_list, orig_vecs_list) = detect_feats(img_fpath)
        >>> # Take 9 keypoints
        >>> img = imread(img_fpath)
        >>> kpts_list = orig_kpts_list[1::len(orig_kpts_list) // 9]
        >>> vecs_list = orig_vecs_list[1::len(orig_vecs_list) // 9]
        >>> # Extract the underlying grayscale patches
        >>> patch_list = extract_patches(img, kpts_list)
        >>> patch_list = np.round(patch_list).astype(np.uint8)
        >>> # Currently its impossible to get the correct answer
        >>> # TODO: allow patches to be passed in as float32
        >>> # Extract descriptors from those patches
        >>> vecs_array = extract_desc_from_patches(patch_list)
        >>> # Comparse to see if they are close to the original descriptors
        >>> errors = (vecs_list.astype(np.float) - vecs_array.astype(np.float)).sum(axis=1)
        >>> print('Errors: %r' % (errors,))
        >>> # xdoctest: +REQUIRES(--show)
        >>> import plottool as pt
        >>> ax = pt.draw_patches_and_sifts(patch_list, vecs_array, pnum=(1, 2, 1))
        >>> ax.set_title('patch extracted')
        >>> ax = pt.draw_patches_and_sifts(patch_list, vecs_list, pnum=(1, 2, 2))
        >>> ax.set_title('image extracted')
        >>> pt.show_if_requested()
    """
    ndims = len(patch_list.shape)
    if ndims == 4 and patch_list.shape[-1] == 1:
        print('[pyhesaff] warning need to reshape patch_list')
        # need to remove grayscale dimension, maybe it should be included
        patch_list = patch_list.reshape(patch_list.shape[0:3])
    elif ndims == 4 and patch_list.shape[-1] == 3:
        assert False, 'cannot handle color images yet'
    assert patch_list.flags['C_CONTIGUOUS'], 'patch_list must be contiguous array'
    num_patches, patch_h, patch_w = patch_list.shape[0:3]
    assert patch_h == patch_w, 'must be square patches'
    vecs_array = alloc_vecs(num_patches)
    #vecs_array[:] = 0
    #print('vecs_array = %r' % (vecs_array,))
    # If the input array list is memmaped it is a good idea to process in chunks
    CHUNKS = isinstance(patch_list, np.memmap)
    if not CHUNKS:
        HESAFF_CLIB.extractDescFromPatches(num_patches, patch_h, patch_w,
                                           patch_list, vecs_array)
    else:
        chunksize = 2048
        _iter = range(num_patches // chunksize)
        _progiter = ub.ProgIter(_iter, desc='extracting sift chunk')
        for ix in _progiter:
            lx = ix * chunksize
            rx = (ix + 1) * chunksize
            patch_sublist = np.array(patch_list[lx:rx])
            sublist_size = rx - lx
            HESAFF_CLIB.extractDescFromPatches(sublist_size, patch_h, patch_w,
                                               patch_sublist,
                                               vecs_array[lx:rx])
        last_size = num_patches - rx
        if last_size > 0:
            lx = rx
            rx = lx + last_size
            patch_sublist = np.array(patch_list[lx:rx])
            sublist_size = rx - lx
            HESAFF_CLIB.extractDescFromPatches(sublist_size, patch_h, patch_w,
                                               patch_sublist,
                                               vecs_array[lx:rx])
    #print('vecs_array = %r' % (vecs_array,))
    return vecs_array

#============================
# other
#============================


def test_rot_invar():
    r"""
    CommandLine:
        python -m pyhesaff test_rot_invar --show
        python -m pyhesaff test_rot_invar --show --nocpp

        python -m vtool.tests.dummy testdata_ratio_matches --show --ratio_thresh=1.0 --rotation_invariance
        python -m vtool.tests.dummy testdata_ratio_matches --show --ratio_thresh=1.1 --rotation_invariance

    Example:
        >>> # DISABLE_DODCTEST
        >>> from pyhesaff._pyhesaff import *  # NOQA
        >>> test_rot_invar()
    """
    import cv2
    import vtool as vt
    import plottool as pt
    TAU = 2 * np.pi
    fnum = pt.next_fnum()
    NUM_PTS = 5  # 9
    theta_list = np.linspace(0, TAU, NUM_PTS, endpoint=False)
    nRows, nCols = pt.get_square_row_cols(len(theta_list), fix=True)
    next_pnum = pt.make_pnum_nextgen(nRows, nCols)
    # Expand the border a bit around star.png
    pad_ = 100
    img_fpath = grab_test_imgpath('star.png')
    img_fpath2 = vt.pad_image_ondisk(img_fpath, pad_, value=26)
    for theta in theta_list:
        print('-----------------')
        print('theta = %r' % (theta,))
        img_fpath = vt.rotate_image_ondisk(img_fpath2, theta, border_mode=cv2.BORDER_REPLICATE)
        if not ub.argflag('--nocpp'):
            (kpts_list_ri, vecs_list2) = detect_feats(img_fpath, rotation_invariance=True)
            kpts_ri = kpts_list_ri[0:2]
        (kpts_list_gv, vecs_list1) = detect_feats(img_fpath, rotation_invariance=False)
        kpts_gv = kpts_list_gv[0:2]
        # find_kpts_direction
        imgBGR = imread(img_fpath)
        kpts_ripy = vt.find_kpts_direction(imgBGR, kpts_gv, DEBUG_ROTINVAR=False)
        # Verify results stdout
        #print('nkpts = %r' % (len(kpts_gv)))
        #print(vt.kpts_repr(kpts_gv))
        #print(vt.kpts_repr(kpts_ri))
        #print(vt.kpts_repr(kpts_ripy))
        # Verify results plot
        pt.figure(fnum=fnum, pnum=next_pnum())
        pt.imshow(imgBGR)
        #if len(kpts_gv) > 0:
        #    pt.draw_kpts2(kpts_gv, ori=True, ell_color=pt.BLUE, ell_linewidth=10.5)
        ell = False
        rect = True
        if not ub.argflag('--nocpp'):
            if len(kpts_ri) > 0:
                pt.draw_kpts2(kpts_ri, rect=rect, ell=ell, ori=True,
                              ell_color=pt.RED, ell_linewidth=5.5)
        if len(kpts_ripy) > 0:
            pt.draw_kpts2(kpts_ripy, rect=rect, ell=ell,  ori=True,
                          ell_color=pt.GREEN, ell_linewidth=3.5)
    pt.set_figtitle('green=python, red=C++')
    pt.show_if_requested()


def vtool_adapt_rotation(img_fpath, kpts):
    """ rotation invariance in python """
    import vtool.patch as ptool
    import vtool.image as gtool
    imgBGR = gtool.imread(img_fpath)
    kpts2 = ptool.find_kpts_direction(imgBGR, kpts)
    vecs2 = extract_vecs(img_fpath, kpts2)
    return kpts2, vecs2


def adapt_scale(img_fpath, kpts):
    import vtool.ellipse as etool
    nScales = 16
    nSamples = 16
    low, high = -1, 2
    kpts2 = etool.adaptive_scale(img_fpath, kpts, nScales, low, high, nSamples)
    # passing in 0 orientation results in gravity vector direction keypoint
    vecs2 = extract_vecs(img_fpath, kpts2)
    return kpts2, vecs2


# del type_, key, val

if __name__ == '__main__':
    """
    CommandLine:
        python -m pyhesaff._pyhesaff
        python -m pyhesaff._pyhesaff --allexamples
        python -m pyhesaff._pyhesaff --allexamples --noface --nosrc
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
