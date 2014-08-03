#!/usr/bin/env python2.7
'the python hessian affine keypoint module'
# TODO: it would be nice to be able to pass around an image
# already in memory instead of having to pass around its path
from __future__ import absolute_import, print_function, division
# Standard
import sys
import six
from six.moves import zip, builtins
from os.path import realpath, dirname
try:
    from . import ctypes_interface
except ValueError:
    import ctypes_interface
import ctypes as C
from collections import OrderedDict
# Scientific
import numpy as np

try:
    getattr(builtins, 'profile')
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
# scalar ctypes
obj_t     = C.c_void_p
str_t     = C.c_char_p
#if six.PY2:
int_t     = C.c_int
#if six.PY3:
#    int_t     = C.c_long
bool_t    = C.c_bool
float_t   = C.c_float
byte_t    = C.c_char
# array ctypes
FLAGS_RW = 'aligned, c_contiguous, writeable'
kpts_t       = np.ctypeslib.ndpointer(dtype=kpts_dtype, ndim=2, flags=FLAGS_RW)
desc_t       = np.ctypeslib.ndpointer(dtype=desc_dtype, ndim=2, flags=FLAGS_RW)
kpts_array_t = np.ctypeslib.ndpointer(dtype=kpts_t, ndim=1, flags=FLAGS_RW)
desc_array_t = np.ctypeslib.ndpointer(dtype=desc_t, ndim=1, flags=FLAGS_RW)
int_array_t  = np.ctypeslib.ndpointer(dtype=int_t, ndim=1, flags=FLAGS_RW)
str_list_t   = C.POINTER(str_t)

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
    def_cfunc(int_t, 'detect',                 [obj_t])
    def_cfunc(int_t, 'get_kpts_dim',           [])
    def_cfunc(None,  'exportArrays',           [obj_t, int_t, kpts_t, desc_t])
    def_cfunc(None,  'extractDesc',            [obj_t, int_t, kpts_t, desc_t])
    def_cfunc(obj_t, 'new_hesaff',             [str_t])
    def_cfunc(obj_t, 'new_hesaff_from_params', [str_t] + hesaff_param_types)
    def_cfunc(None,  'detectKeypointsList',    [int_t, str_list_t, kpts_array_t,
                                                desc_array_t, int_array_t] +
                                                hesaff_param_types)
    return clib, lib_fpath

# Create a global interface to the hesaff lib
HESAFF_CLIB, __LIB_FPATH__ = load_hesaff_clib()
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
    for key, val in six.iteritems(kwargs):
        if key in hesaff_params:
            hesaff_params[key] = val
        else:
            print('[pyhesaff] WARNING: key=%r is not known' % key)


def _new_hesaff(img_fpath, **kwargs):
    """ Creates new detector object which reads the image """
    hesaff_params = hesaff_param_dict.copy()
    hesaff_params.update(kwargs)
    if __DEBUG__:
        print('[hes] New Hesaff')
        print('[hes] hesaff_params=%r' % (hesaff_params,))
    hesaff_args = hesaff_params.values()  # pass all parameters to HESAFF_CLIB
    img_realpath = realpath(img_fpath)
    if six.PY3:
        # convert out of unicode
        img_realpath = img_realpath.encode('ascii')
    hesaff_ptr = HESAFF_CLIB.new_hesaff_from_params(img_realpath,
                                                    *hesaff_args)
    return hesaff_ptr


def extract_desc(img_fpath, kpts, **kwargs):
    hesaff_ptr = _new_hesaff(img_fpath, **kwargs)
    nKpts = len(kpts)
    desc = _alloc_desc(nKpts)  # allocate memory for new descriptors
    kpts = np.ascontiguousarray(kpts)  # kpts might not be contiguous
    # extract descriptors at given locations
    HESAFF_CLIB.extractDesc(hesaff_ptr, nKpts, kpts, desc)
    return desc


def _cast_strlist_to_C(py_strlist):
    """
    Converts a python list of strings into a c array of strings
    adapted from "http://stackoverflow.com/questions/3494598/passing-a-list-of
    -strings-to-from-python-ctypes-to-c-function-expecting-char"
    """
    c_strarr = (str_t * len(py_strlist))()
    c_strarr[:] = py_strlist
    return c_strarr


def arrptr_to_np(c_arrptr, shape, arr_t, dtype):
    """
    Casts an array pointer from C to numpy
    Input:
        c_arrpt - an array pointer returned from C
        shape   - shape of that array pointer
        arr_t   - the ctypes datatype of c_arrptr
    """
    try:
        _byte_t = byte_t  # localize scope
        itemsize_ = dtype().itemsize
        ptr_t = _byte_t * itemsize_
        if six.PY2:
            arr_t_size = C.POINTER(ptr_t)  # size of each item
            typed_c_arrptr = c_arrptr.astype(int)
            c_arr = C.cast(typed_c_arrptr, arr_t_size)   # cast to ctypes
            raise Exception('fuuu. Why does 2.7 work? Why does 3.4 not!?!!!')
        else:
            #ptr_t = _byte_t * itemsize_
            #ptr_t2 = C.c_char * (itemsize_ * 2)
            arr_t_size = C.POINTER(ptr_t)  # size of each item
            #arr_t_size = C.POINTER(C.c_int64)  # size of each item
            #typed_c_arrptr = c_arrptr.astype(C.c_ssize_t)
            #typed_c_arrptr = c_arrptr.astype(int)
            #typed_c_arrptr = c_arrptr.astype(C.c_uint32)
            #typed_c_arrptr = c_arrptr.astype(ptr_t)
            #typed_c_arrptr = c_arrptr.astype(ptr_t2)
            #typed_c_arrptr = c_arrptr.astype(C.c_uint8)
            #typed_c_arrptr = c_arrptr.astype(C.c_void_p)
            #typed_c_arrptr = c_arrptr.astype(C.c_int)
            #typed_c_arrptr = c_arrptr.astype(C.c_char)  # WORKS BUT WRONG
            #typed_c_arrptr = c_arrptr.astype(C.c_char)  # WORKS BUT WRONG
            #typed_c_arrptr = c_arrptr.astype(bytes)  # WORKS BUT WRONG
            #typed_c_arrptr = c_arrptr.astype(int)
            #typed_c_arrptr = c_arrptr
            #typed_c_arrptr = c_arrptr.astype(np.int64)
            typed_c_arrptr = c_arrptr.astype(int)
            c_arr = C.cast(typed_c_arrptr, arr_t_size)   # cast to ctypes
        np_arr = np.ctypeslib.as_array(c_arr, shape)       # cast to numpy
        np_arr.dtype = dtype                               # fix numpy dtype
    except Exception as ex:
        import utool
        varnames = sorted(list(locals().keys()))
        vartypes = [(type, name) for name in varnames]
        spaces    = [None for name in varnames]
        key_list = list(utool.roundrobin(varnames, vartypes, spaces))
        print('itemsize(float) = %r' % np.dtype(float).itemsize)
        print('itemsize(c_char) = %r' % np.dtype(C.c_char).itemsize)
        print('itemsize(c_wchar) = %r' % np.dtype(C.c_wchar).itemsize)
        print('itemsize(c_char_p) = %r' % np.dtype(C.c_char_p).itemsize)
        print('itemsize(c_wchar_p) = %r' % np.dtype(C.c_wchar_p).itemsize)
        print('itemsize(c_int) = %r' % np.dtype(C.c_int).itemsize)
        print('itemsize(c_int32) = %r' % np.dtype(C.c_int32).itemsize)
        print('itemsize(c_int64) = %r' % np.dtype(C.c_int64).itemsize)
        print('itemsize(int) = %r' % np.dtype(int).itemsize)
        print('itemsize(float32) = %r' % np.dtype(np.float32).itemsize)
        print('itemsize(float64) = %r' % np.dtype(np.float64).itemsize)
        utool.printex(ex, key_list=key_list)
        raise
    return np_arr


def extract_2darr_list(size_list, ptr_list, arr_t, arr_dtype,
                        arr_dim):
    """
    size_list - contains the size of each output 2d array
    ptr_list  - an array of pointers to the head of each output 2d
                array (which was allocated in C)
    arr_t     - the C pointer type
    arr_dtype - the numpy array type
    arr_dim   - the number of columns in each output 2d array
    """
    iter_ = ((arr_ptr, (size, arr_dim))
             for (arr_ptr, size) in zip(ptr_list, size_list))
    arr_list = [arrptr_to_np(arr_ptr, shape, arr_t, arr_dtype)
                for arr_ptr, shape in iter_]
    return arr_list


def detect_kpts_list(image_paths_list, **kwargs):
    """
    Input: A list of image paths
    Output: A tuple of lists of keypoints and descriptors
    """
    # Get Num Images
    nImgs = len(image_paths_list)

    # Cast string list to C
    if six.PY2:
        realpaths_list = list(map(realpath, image_paths_list))
    if six.PY3:
        realpaths_list = [realpath(path).encode('ascii') for path in image_paths_list]

    c_strs = _cast_strlist_to_C(realpaths_list)

    # Allocate empty array pointers for each image
    kpts_ptr_array = np.empty(nImgs, dtype=kpts_t)  # array of float arrays
    desc_ptr_array = np.empty(nImgs, dtype=desc_t)  # array of byte arrays
    nDetect_array = np.empty(nImgs, dtype=int_t)  # array of detections per image

    # Get algorithm parameters
    hesaff_params = hesaff_param_dict.copy()
    hesaff_params.update(kwargs)
    hesaff_args = hesaff_params.values()  # pass all parameters to HESAFF_CLIB

    # Detect keypoints in parallel
    HESAFF_CLIB.detectKeypointsList(nImgs, c_strs,
                                    kpts_ptr_array, desc_ptr_array,
                                    nDetect_array, *hesaff_args)

    # Cast keypoint array to list of numpy keypoints
    kpts_list = extract_2darr_list(nDetect_array, kpts_ptr_array, kpts_t, kpts_dtype, KPTS_DIM)
    # Cast descriptor array to list of numpy descriptors
    desc_list = extract_2darr_list(nDetect_array, desc_ptr_array, desc_t, desc_dtype, DESC_DIM)

    #kpts_list = [arrptr_to_np(kpts_ptr, (len_, KPTS_DIM), kpts_t, kpts_dtype)
    #             for (kpts_ptr, len_) in zip(kpts_ptr_array, nDetect_array)]
    #desc_list = [arrptr_to_np(desc_ptr, (len_, DESC_DIM), desc_t, desc_dtype)
    #             for (desc_ptr, len_) in zip(desc_ptr_array, nDetect_array)]

    return kpts_list, desc_list


@profile
def detect_kpts(img_fpath,
                use_adaptive_scale=False, nogravity_hack=False,
                **kwargs):
    """
    main driver function for detecting hessian affine keypoints.
    extra parameters can be passed to the hessian affine detector by using
    kwargs.  """
    #Valid keyword arguments are: + str(hesaff_param_dict.keys())
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
