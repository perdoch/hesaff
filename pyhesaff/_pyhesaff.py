#!/usr/bin/env python
'the python hessian affine keypoint module'
# TODO: it would be nice to be able to pass around an image
# already in memory instead of having to pass around its path
from __future__ import absolute_import, print_function, division
import __builtin__
# Standard
import sys
from os.path import realpath, dirname
from . import ctypes_interface
import ctypes as C
from collections import OrderedDict
from itertools import izip
# Scientific
import numpy as np

try:
    getattr(__builtin__, 'profile')
except AttributeError:
    def profile(func):
        return func


__DEBUG__ = '--debug-pyhesaff' in sys.argv or '--debug' in sys.argv

#============================
# hesaff ctypes interface
#============================

# numpy dtypes
kpts_dtype = np.float32
desc_dtype = np.uint8
# ctypes
FLAGS_RW = 'aligned, c_contiguous, writeable'
kpts_t    = np.ctypeslib.ndpointer(dtype=kpts_dtype, ndim=2, flags=FLAGS_RW)
desc_t    = np.ctypeslib.ndpointer(dtype=desc_dtype, ndim=2, flags=FLAGS_RW)
kpts_array_t = np.ctypeslib.ndpointer(dtype=kpts_t, ndim=1, flags=FLAGS_RW)
desc_array_t = np.ctypeslib.ndpointer(dtype=desc_t, ndim=1, flags=FLAGS_RW)
int_array_t = np.ctypeslib.ndpointer(dtype=C.c_int, ndim=1, flags=FLAGS_RW)
obj_t     = C.c_void_p
str_t     = C.c_char_p
int_t     = C.c_int
bool_t    = C.c_bool
float_t   = C.c_float

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
]

hesaff_param_dict = OrderedDict([(key, val) for (type_, key, val) in hesaff_typed_params])
hesaff_param_types = [type_ for (type_, key, val) in hesaff_typed_params]


def load_hesaff_clib():
    '''
    Specificially loads the hesaff lib and defines its functions
    '''
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
    hesaff_clib, def_cfunc = ctypes_interface.load_clib(libname, root_dir)
    # Expose extern C Functions
    def_cfunc(int_t, 'detect',                 [obj_t])
    def_cfunc(int_t, 'get_kpts_dim',           [])
    def_cfunc(None,  'exportArrays',           [obj_t, int_t, kpts_t, desc_t])
    def_cfunc(None,  'extractDesc',            [obj_t, int_t, kpts_t, desc_t])
    def_cfunc(obj_t, 'new_hesaff',             [str_t])
    def_cfunc(obj_t, 'new_hesaff_from_params', [str_t] + hesaff_param_types)
    def_cfunc(None,  'detectKeypointsList',    [int_t, C.POINTER(str_t), kpts_array_t, desc_array_t, int_array_t] + hesaff_param_types)
    return hesaff_clib

# Create a global interface to the hesaff lib
HESAFF_CLIB = load_hesaff_clib()
KPTS_DIM = HESAFF_CLIB.get_kpts_dim()
DESC_DIM = HESAFF_CLIB.get_desc_dim()


if __DEBUG__:
    print('[hes] %r KPTS_DIM = %r' % (type(KPTS_DIM), KPTS_DIM))
    print('[hes] %r DESC_DIM = %r' % (type(KPTS_DIM), DESC_DIM))


#============================
# hesaff python interface
#============================


def _alloc_desc(nKpts):
    desc = np.empty((nKpts, DESC_DIM), desc_dtype)
    return desc


def _allocate_kpts_and_desc(nKpts):
    kpts = np.empty((nKpts, KPTS_DIM), kpts_dtype)  # array of floats
    desc = _alloc_desc(nKpts)  # array of bytes
    return kpts, desc


def _make_hesaff_cpp_params(**kwargs):
    hesaff_params = hesaff_param_dict.copy()
    for key, val in kwargs.iteritems():
        if key in hesaff_params:
            hesaff_params[key] = val
        else:
            print('[pyhesaff] WARNING: key=%r is not known' % key)


def _new_hesaff(img_fpath, **kwargs):
    'Creates new detector object which reads the image'
    hesaff_params = hesaff_param_dict.copy()
    hesaff_params.update(kwargs)
    if __DEBUG__:
        print('[hes] New Hesaff')
        print('[hes] hesaff_params=%r' % (hesaff_params,))
    hesaff_args = hesaff_params.values()  # pass all parameters to HESAFF_CLIB
    hesaff_ptr = HESAFF_CLIB.new_hesaff_from_params(realpath(img_fpath), *hesaff_args)
    return hesaff_ptr


def extract_desc(img_fpath, kpts, **kwargs):
    hesaff_ptr = _new_hesaff(img_fpath, **kwargs)
    nKpts = len(kpts)
    desc = _alloc_desc(nKpts)  # allocate memory for new descriptors
    kpts = np.ascontiguousarray(kpts)  # kpts might not be contiguous
    # extract descriptors at given locations
    HESAFF_CLIB.extractDesc(hesaff_ptr, nKpts, kpts, desc)
    return desc


@profile
def detect_kpts(img_fpath,
                use_adaptive_scale=False, nogravity_hack=False,
                **kwargs):
    '''
    main driver function for detecting hessian affine keypoints.
    extra parameters can be passed to the hessian affine detector by using
    kwargs. Valid keyword arguments are:
    ''' + str(hesaff_param_dict.keys())
    if __DEBUG__:
        print('[hes] Detecting Keypoints')
        print('[hes] use_adaptive_scale=%r' % (use_adaptive_scale,))
        print('[hes] nogravity_hack=%r' % (nogravity_hack,))
        print('[hes] kwargs=%r' % (kwargs,))
    hesaff_ptr = _new_hesaff(img_fpath, **kwargs)
    if __DEBUG__:
        print('[hes] detect')
    nKpts = HESAFF_CLIB.detect(hesaff_ptr)  # Get num detected
    if __DEBUG__:
        print('[hes] allocate')
    kpts, desc = _allocate_kpts_and_desc(nKpts)  # Allocate arrays
    if __DEBUG__:
        print('[hes] export')
    HESAFF_CLIB.exportArrays(hesaff_ptr, nKpts, kpts, desc)  # Populate arrays
    if use_adaptive_scale:  # Adapt scale if requested
        #print('Adapting Scale')
        if __DEBUG__:
            print('[hes] adapt_scale')
        kpts, desc = adapt_scale(img_fpath, kpts)
    if nogravity_hack:
        if __DEBUG__:
            print('[hes] adapt_rotation')
        kpts, desc = adapt_rotation(img_fpath, kpts)
    return kpts, desc


#adapted from "http://stackoverflow.com/questions/3494598/passing-a-list-of-strings-to-from-python-ctypes-to-c-function-expecting-char"
def _python_list_to_c_string_array(python_list):
    arr = (C.c_char_p * len(python_list))()
    arr[:] = python_list
    return (len(python_list), arr)


def detect_kpts_list(image_paths_list, **kwargs):
    (listlen, c_strings) = _python_list_to_c_string_array([realpath(path) for path in image_paths_list])
    kpts_array = np.empty((listlen), kpts_t)   # array of float arrays
    desc_array = np.empty((listlen), desc_t)   # array of byte arrays
    length_array = np.empty((listlen), int_t)  # array of ints
    hesaff_params = hesaff_param_dict.copy()
    hesaff_params.update(kwargs)
    hesaff_args = hesaff_params.values()  # pass all parameters to HESAFF_CLIB
    HESAFF_CLIB.detectKeypointsList(listlen, c_strings, kpts_array, desc_array, length_array, *hesaff_args)
    #kpts_array2 = [np.ctypeslib.as_array(C.cast(kpts_ptr.astype(int),C.POINTER(C.c_char * kpts_dtype().itemsize)),(length, KPTS_DIM)) for (kpts_ptr, length) in izip(kpts_array,length_array)]

    as_array = np.ctypeslib.as_array
    kpts_dtype_size = C.POINTER(C.c_char * kpts_dtype().itemsize)
    def _cast_kpts(kpts_ptr, length):
        c_kpts = C.cast(kpts_ptr.astype(int), kpts_dtype_size)
        np_kpts = as_array(c_kpts, (length, KPTS_DIM))
        np_kpts.dtype = kpts_dtype
        return np_kpts
    kpts_array2 = [_cast_kpts(kpts_ptr, length) for (kpts_ptr, length) in izip(kpts_array, length_array)]
    desc_dtype_size = C.POINTER(C.c_char * desc_dtype().itemsize)
    def _cast_desc(desc_ptr, length):
        c_desc = C.cast(desc_ptr.astype(int), desc_dtype_size)
        np_desc = as_array(c_desc, (length, DESC_DIM))
        np_desc.dtype = desc_dtype
        return np_desc
    desc_array2 = [_cast_desc(desc_ptr, length) for (desc_ptr, length) in izip(desc_array, length_array)]
    return (kpts_array2, desc_array2, length_array)


def adapt_rotation(img_fpath, kpts):
    import vtool.patch as ptool
    import vtool.image as gtool
    imgBGR = gtool.imread(img_fpath)
    kpts2 = ptool.find_kpts_direction(imgBGR, kpts)
    desc2 = extract_desc(img_fpath, kpts2)
    return kpts2, desc2


@profile
def adapt_scale(img_fpath, kpts):
    import vtool.ellipse as etool
    nScales = 16
    nSamples = 16
    low, high = -1, 2
    kpts2 = etool.adaptive_scale(img_fpath, kpts, nScales, low, high, nSamples)
    # passing in 0 orientation results in gravity vector direction keypoint
    desc2 = extract_desc(img_fpath, kpts2)
    return kpts2, desc2
