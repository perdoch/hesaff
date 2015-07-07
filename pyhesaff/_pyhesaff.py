#!/usr/bin/env python2.7
"""
The python hessian affine keypoint module

Command Line:
    python -c "import utool as ut; ut.write_modscript_alias('Fshow.sh', 'pyhesaff._pyhesaff --test-detect_kpts --fname easy1.png --verbose  --show')"
    Fshow.sh --no-affine_invariance --scale-max=150 --darken .5
    Fshow.sh --no-affine_invariance --scale-max=100 --darken .5
    Fshow.sh --no-affine_invariance --scale-max=100
    Fshow.sh --no-affine_invariance --scale-max=100

    sh Fshow.sh --no-affine_invariance --numberOfScales=3 --maxPyramidLevels=-1 --darken .5
    sh Fshow.sh --no-affine_invariance --numberOfScales=1 --maxPyramidLevels=1 --darken .5
    sh Fshow.sh --affine_invariance --numberOfScales=1 --maxPyramidLevels=1 --darken .5
    sh Fshow.sh --affine_invariance --numberOfScales=1 --maxPyramidLevels=1 --darken .5
    sh Fshow.sh --no-affine_invariance --numberOfScales=3 --border=5

    sh Fshow.sh --affine_invariance --numberOfScales=1 --maxPyramidLevels=1 --darken .5 --edgeEigenValueRatio=1.1
    sh Fshow.sh --no-affine_invariance --numberOfScales=3 --maxPyramidLevels=2 --border=5

"""
# TODO: it would be nice to be able to pass around an image
# already in memory instead of having to pass around its path
from __future__ import absolute_import, print_function, division
# Standard
import sys
import six
#from six.moves import zip, builtins
from os.path import realpath, dirname
try:
    from pyhesaff import ctypes_interface
except ValueError:
    import ctypes_interface
import ctypes
import ctypes as C
from collections import OrderedDict
# Scientific
import numpy as np
import utool as ut
#print, print_, printDBG, rrr, profile = ut.inject(__name__, '[hesaff]')

#try:
#    getattr(builtins, 'profile')
#except AttributeError:
#    def profile(func):
#        return func


__DEBUG__ = '--debug-pyhesaff' in sys.argv or '--debug' in sys.argv

#============================
# hesaff ctypes interface
#============================

# numpy dtypes
kpts_dtype = np.float32
vecs_dtype = np.uint8
img_dtype  = np.uint8
# scalar ctypes
obj_t     = C.c_void_p
str_t     = C.c_char_p
PY2 = True  # six.PY2
#if six.PY2:
if PY2:
    int_t     = C.c_int
else:
    raise NotImplementedError('PY3')
#if six.PY3:
#    int_t     = C.c_long
bool_t    = C.c_bool
float_t   = C.c_float
#byte_t    = C.c_char
# array ctypes
FLAGS_RW = 'aligned, c_contiguous, writeable'
FLAGS_RO = 'aligned, c_contiguous'
#FLAGS_RW = 'aligned, writeable'
kpts_t       = np.ctypeslib.ndpointer(dtype=kpts_dtype, ndim=2, flags=FLAGS_RW)
vecs_t       = np.ctypeslib.ndpointer(dtype=vecs_dtype, ndim=2, flags=FLAGS_RW)
img_t        = np.ctypeslib.ndpointer(dtype=img_dtype, ndim=3, flags=FLAGS_RO)
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
    #
    (bool_t,  'use_dense', False),
    (int_t,   'dense_stride', 32),
]

HESAFF_PARAM_DICT = OrderedDict([(key, val) for (type_, key, val) in HESAFF_TYPED_PARAMS])
HESAFF_PARAM_TYPES = [type_ for (type_, key, val) in HESAFF_TYPED_PARAMS]


def build_typed_params_kwargs_docstr_block(typed_params):
    r"""
    Args:
        typed_params (?):

    CommandLine:
        python -m pyhesaff._pyhesaff --test-build_typed_params_docstr

    Example:
        >>> # DISABLE_DOCTEST
        >>> from pyhesaff._pyhesaff import *  # NOQA
        >>> # execute function
        >>> typed_params = HESAFF_TYPED_PARAMS
        >>> result = build_typed_params_docstr(typed_params)
        >>> # verify results
        >>> print(result)
    """
    kwargs_lines = []
    for tup in typed_params:
        type_, name, default = tup
        typestr = str(type_).replace('<class \'ctypes.c_', '').replace('\'>', '')
        line_fmtstr = '{name} ({typestr}): default={default}'
        line = line_fmtstr.format(name=name, typestr=typestr, default=default)
        kwargs_lines.append(line)
    kwargs_docstr_block = ('Kwargs:\n' + ut.indent('\n'.join(kwargs_lines), '    '))
    return ut.indent(kwargs_docstr_block, '    ')
hesaff_kwargs_docstr_block = build_typed_params_kwargs_docstr_block(HESAFF_TYPED_PARAMS)


HESAFF_CLIB = None
REBUILD_ONCE = 0


def load_hesaff_clib(rebuild=None):
    """
    Specificially loads the hesaff lib and defines its functions

    CommandLine:
        python -m pyhesaff._pyhesaff --test-load_hesaff_clib --rebuild

    Example:
        >>> pass
        >>> #import pyhesaff
    """
    global REBUILD_ONCE
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
    if rebuild is not False and REBUILD_ONCE == 0 and __name__ != '__main__':
        REBUILD_ONCE += 1
        rebuild = ut.get_argflag('--rebuild-hesaff')
        if rebuild:
            print('REBUILDING HESAFF')
            repo_dir = realpath(dirname(root_dir))
            ut.std_build_command(repo_dir)

    libname = 'hesaff'
    (clib, def_cfunc, lib_fpath) = ctypes_interface.load_clib(libname, root_dir)
    # Expose extern C Functions to hesaff's clib
    def_cfunc(int_t, 'get_cpp_version',        [])
    def_cfunc(int_t, 'is_debug_mode',          [])
    def_cfunc(int_t, 'detect',                 [obj_t])
    def_cfunc(int_t, 'get_kpts_dim',           [])
    def_cfunc(int_t, 'get_desc_dim',           [])
    def_cfunc(None,  'exportArrays',           [obj_t, int_t, kpts_t, vecs_t])
    def_cfunc(None,  'extractDesc',            [obj_t, int_t, kpts_t, vecs_t])
    def_cfunc(None,  'extractDescFromPatches', [int_t, int_t, int_t, img_t, vecs_t])
    def_cfunc(obj_t, 'new_hesaff',             [str_t])
    def_cfunc(obj_t, 'new_hesaff_from_fpath_and_params', [str_t] + HESAFF_PARAM_TYPES)
    def_cfunc(obj_t, 'new_hesaff_from_image_and_params', [img_t, int_t, int_t, int_t] + HESAFF_PARAM_TYPES)
    def_cfunc(None,  'detectKeypointsList',    [int_t, str_list_t, kpts_array_t,
                                                vecs_array_t, int_array_t] +
                                                HESAFF_PARAM_TYPES)
    def_cfunc(obj_t, 'detectKeypointsListStep1',   [int_t, str_list_t] + HESAFF_PARAM_TYPES)
    def_cfunc(None,  'detectKeypointsListStep2',    [int_t, obj_t, int_array_t])
    def_cfunc(None,  'detectKeypointsListStep3',    [int_t, obj_t, int_array_t, int_array_t, kpts_t, vecs_t])
    return clib, lib_fpath

# Create a global interface to the hesaff lib
HESAFF_CLIB, __LIB_FPATH__ = load_hesaff_clib()
KPTS_DIM = HESAFF_CLIB.get_kpts_dim()
DESC_DIM = HESAFF_CLIB.get_desc_dim()


if __DEBUG__:
    print('[hes] %r KPTS_DIM = %r' % (type(KPTS_DIM), KPTS_DIM))
    print('[hes] %r DESC_DIM = %r' % (type(KPTS_DIM), DESC_DIM))

#============================
# helpers
#============================


def _alloc_vecs(nKpts):
    #vecs = np.empty((nKpts, DESC_DIM), vecs_dtype)
    vecs = np.zeros((nKpts, DESC_DIM), vecs_dtype)
    return vecs


def _allocate_kpts_and_vecs(nKpts):
    #kpts = np.empty((nKpts, KPTS_DIM), kpts_dtype)  # array of floats
    kpts = np.zeros((nKpts, KPTS_DIM), kpts_dtype) - 1.0  # array of floats
    vecs = _alloc_vecs(nKpts)  # array of bytes
    return kpts, vecs


def _make_hesaff_cpp_params(**kwargs):
    hesaff_params = HESAFF_PARAM_DICT.copy()
    for key, val in six.iteritems(kwargs):
        if key in hesaff_params:
            hesaff_params[key] = val
        else:
            print('[pyhesaff] WARNING: key=%r is not known' % key)


def _cast_strlist_to_C(py_strlist):
    """
    Converts a python list of strings into a c array of strings
    adapted from "http://stackoverflow.com/questions/3494598/passing-a-list-of
    -strings-to-from-python-ctypes-to-c-function-expecting-char"
    """
    c_strarr = (str_t * len(py_strlist))()
    c_strarr[:] = py_strlist
    return c_strarr


def arrptr_to_np_OLD(c_arrptr, shape, arr_t, dtype):
    """
    Casts an array pointer from C to numpy

    Args:
        c_arrptr (uint64): a pointer to an array returned from C
        shape (tuple): shape of the underlying array being pointed to
        arr_t (PyCSimpleType): the ctypes datatype of c_arrptr
        dtype (dtype): numpy datatype the array will be to cast into

    CommandLine:
        python2 -m pyhesaff._pyhesaff --test-detect_kpts_list:0 --rebuild-hesaff
        python2 -m pyhesaff._pyhesaff --test-detect_kpts_list:0
        python3 -m pyhesaff._pyhesaff --test-detect_kpts_list:0

    """
    try:
        byte_t = ctypes.c_char
        itemsize_ = dtype().itemsize
        #import utool
        #utool.printvar2('itemsize_')
        ###---------
        #dtype_t1 = C.c_voidp * itemsize_
        #dtype_ptr_t1 = C.POINTER(dtype_t1)  # size of each item
        #dtype_ptr_t = dtype_ptr_t1
        ###---------
        if True or six.PY2:
            # datatype of array elements
            dtype_t = byte_t * itemsize_
            dtype_ptr_t = C.POINTER(dtype_t)  # size of each item
            #typed_c_arrptr = c_arrptr.astype(C.c_long)
            typed_c_arrptr = c_arrptr.astype(int)
            c_arr = C.cast(typed_c_arrptr, dtype_ptr_t)   # cast to ctypes
            #raise Exception('fuuu. Why does 2.7 work? Why does 3.4 not!?!!!')
        else:
            dtype_t = C.c_char * itemsize_
            dtype_ptr_t = C.POINTER(dtype_t)  # size of each item
            #typed_c_arrptr = c_arrptr.astype(int)
            #typed_c_arrptr = c_arrptr.astype(C.c_size_t)
            typed_c_arrptr = c_arrptr.astype(int)
            c_arr = C.cast(c_arrptr.astype(C.c_size_t), dtype_ptr_t)   # cast to ctypes
            c_arr = C.cast(c_arrptr.astype(int), dtype_ptr_t)   # cast to ctypes
            c_arr = C.cast(c_arrptr, dtype_ptr_t)   # cast to ctypes
            #typed_c_arrptr = c_arrptr.astype(int)
            #, order='C', casting='safe')
            #utool.embed()
            #typed_c_arrptr = c_arrptr.astype(dtype_t)
            #typed_c_arrptr = c_arrptr.astype(ptr_t2)
            #typed_c_arrptr = c_arrptr.astype(C.c_uint8)
            #typed_c_arrptr = c_arrptr.astype(C.c_void_p)
            #typed_c_arrptr = c_arrptr.astype(C.c_int)
            #typed_c_arrptr = c_arrptr.astype(C.c_char)  # WORKS BUT WRONG
            #typed_c_arrptr = c_arrptr.astype(bytes)  # WORKS BUT WRONG
            #typed_c_arrptr = c_arrptr.astype(int)
            #typed_c_arrptr = c_arrptr
            #typed_c_arrptr = c_arrptr.astype(np.int64)
            #typed_c_arrptr = c_arrptr.astype(int)

            """
            ctypes.cast(arg1, arg2)

            Input:
                arg1 - a ctypes object that is or can be converted to a pointer
                       of some kind
                arg2 - a ctypes pointer type.
            Output:
                 It returns an instance of the second argument, which references
                 the same memory block as the first argument
            """
            c_arr = C.cast(typed_c_arrptr, dtype_ptr_t)   # cast to ctypes
        np_arr = np.ctypeslib.as_array(c_arr, shape)       # cast to numpy
        np_arr.dtype = dtype                               # fix numpy dtype
    except Exception as ex:
        import utool as ut
        #utool.embed()
        varnames = sorted(list(locals().keys()))
        vartypes = [(type, name) for name in varnames]
        spaces    = [None for name in varnames]
        c_arrptr_dtype = c_arrptr.dtype  # NOQA
        #key_list = list(zip(varnames, vartypes, spaces))
        key_list = ['c_arrptr_dtype'] + 'c_arrptr, shape, arr_t, dtype'.split(', ')
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
        ut.printex(ex, keys=key_list)
        ut.embed()
        raise
    return np_arr


def get_hesaff_default_params():
    return HESAFF_PARAM_DICT.copy()


#============================
# hesaff python interface
#============================


def get_cpp_version():
    r"""
    Returns:
        int: cpp_version

    CommandLine:
        python -m pyhesaff._pyhesaff --test-get_cpp_version

    Example:
        >>> # ENABLE_DOCTEST
        >>> from pyhesaff._pyhesaff import *  # NOQA
        >>> cpp_version = get_cpp_version()
        >>> result = str(cpp_version)
        >>> print(result)
        >>> ut.assert_eq(cpp_version, 2, 'cpp version mimatch')
    """
    cpp_version = HESAFF_CLIB.get_cpp_version()
    return cpp_version


def _new_fpath_hesaff(img_fpath, **kwargs):
    """ Creates new detector object which reads the image """
    hesaff_params = HESAFF_PARAM_DICT.copy()
    hesaff_params.update(kwargs)
    try:
        assert len(hesaff_params) == len(HESAFF_PARAM_DICT), (
            'len(hesaff_params) = %d, len(HESAFF_PARAM_DICT)=%d' % (len(hesaff_params), len(HESAFF_PARAM_DICT)))
    except AssertionError as ex:
        print('Unknown paramaters = %s' % (ut.dict_str(ut.dict_setdiff(kwargs, HESAFF_PARAM_DICT.keys()))))
        raise

    if __DEBUG__:
        print('[hes] New Hesaff')
        print('[hes] hesaff_params=%r' % (hesaff_params,))
    hesaff_args = hesaff_params.values()  # pass all parameters to HESAFF_CLIB
    img_realpath = realpath(img_fpath)
    if six.PY3:
        # convert out of unicode
        img_realpath = img_realpath.encode('ascii')
    try:
        hesaff_ptr = HESAFF_CLIB.new_hesaff_from_fpath_and_params(img_realpath, *hesaff_args)
    except Exception as ex:
        msg = 'hesaff_ptr = HESAFF_CLIB.new_hesaff_from_fpath_and_params(img_realpath, *hesaff_args)',
        print(msg)
        print('hesaff_args = ')
        print(hesaff_args)
        import utool
        utool.printex(ex, msg, keys=['hesaff_args'])
        raise
    return hesaff_ptr


def _new_image_hesaff(img, **kwargs):
    """ Creates new detector object which reads the image """
    hesaff_params = HESAFF_PARAM_DICT.copy()
    hesaff_params.update(kwargs)
    try:
        assert len(hesaff_params) == len(HESAFF_PARAM_DICT), (
            'len(hesaff_params) = %d, len(HESAFF_PARAM_DICT)=%d' % (len(hesaff_params), len(HESAFF_PARAM_DICT)))
    except AssertionError as ex:
        print('Unknown paramaters = %s' % (ut.dict_str(ut.dict_setdiff(kwargs, HESAFF_PARAM_DICT.keys()))))
        raise

    if __DEBUG__:
        print('[hes] New Hesaff')
        print('[hes] hesaff_params=%r' % (hesaff_params,))
    hesaff_args = hesaff_params.values()  # pass all parameters to HESAFF_CLIB
    rows, cols = img.shape[0:2]
    if len(img.shape) == 2:
        channels = 1
    else:
        channels = img.shape[2]
    try:
        hesaff_ptr = HESAFF_CLIB.new_hesaff_from_image_and_params(img, rows, cols, channels, *hesaff_args)
    except Exception as ex:
        msg = 'hesaff_ptr = HESAFF_CLIB.new_hesaff_from_image_and_params(img_realpath, *hesaff_args)',
        print(msg)
        print('hesaff_args = ')
        print(hesaff_args)
        import utool
        utool.printex(ex, msg, keys=['hesaff_args'])
        raise
    return hesaff_ptr


def extract_vecs(img_fpath, kpts, **kwargs):
    r"""
    Extract SIFT descriptors at keypoint locations

    Args:
        img_fpath (?):
        kpts (ndarray[float32_t, ndim=2]):  keypoints

    Returns:
        ndarray[uint8_t, ndim=2]: vecs -  descriptor vectors

    CommandLine:
        python -m pyhesaff._pyhesaff --test-extract_vecs

    Example:
        >>> # ENABLE_DOCTEST
        >>> from pyhesaff._pyhesaff import *  # NOQA
        >>> import vtool as vt
        >>> img_fpath = ut.grab_test_imgpath('carl.jpg')
        >>> kpts = vt.dummy.get_dummy_kpts()
        >>> vecs = extract_vecs(img_fpath, kpts)
        >>> result = ('vecs = %s' % (str(vecs),))
        >>> print(result)
    """
    hesaff_ptr = _new_fpath_hesaff(img_fpath, **kwargs)
    nKpts = len(kpts)
    vecs = _alloc_vecs(nKpts)  # allocate memory for new decsriptors
    kpts = np.ascontiguousarray(kpts)  # kpts might not be contiguous
    # extract decsriptors at given locations
    HESAFF_CLIB.extractDesc(hesaff_ptr, nKpts, kpts, vecs)
    return vecs


def extract_desc_from_patches(patch_list):
    r"""
    Args:
        patch_list (list):

    CommandLine:
        python -m pyhesaff._pyhesaff --test-extract_desc_from_patches  --rebuild-hesaff --no-rmbuild
        python -m pyhesaff._pyhesaff --test-extract_desc_from_patches  --rebuild-hesaff --no-rmbuild --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from pyhesaff._pyhesaff import *  # NOQA
        >>> from pyhesaff._pyhesaff import _alloc_vecs
        >>> import vtool as vt
        >>> img_fpath = ut.grab_test_imgpath(ut.get_argval('--fname', default='lena.png'))
        >>> (kpts_list, vecs_list) = detect_kpts(img_fpath)
        >>> img = vt.imread(img_fpath)
        >>> kpts_list = kpts_list[1::len(kpts_list) // 9]
        >>> patch_list_ = np.array(vt.get_warped_patches(img, kpts_list, patch_size=64)[0])
        >>> patch_list = np.array(vt.convert_image_list_colorspace(patch_list_, 'gray'))
        >>> vecs_array = extract_desc_from_patches(patch_list)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.draw_patches_and_sifts(patch_list, vecs_array)
        >>> ut.show_if_requested()
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
    vecs_array = _alloc_vecs(num_patches)
    #vecs_array[:] = 0
    #print('vecs_array = %r' % (vecs_array,))
    # If the input array list is memmaped it is a good idea to process in chunks
    CHUNKS = isinstance(patch_list, np.memmap)
    if not CHUNKS:
        HESAFF_CLIB.extractDescFromPatches(num_patches, patch_h, patch_w, patch_list, vecs_array)
    else:
        from six.moves import range
        chunksize = 2048
        _iter = range(num_patches // chunksize)
        _progiter = ut.ProgressIter(_iter, lbl='extracting sift chunk')
        for ix in _progiter:
            lx = ix * chunksize
            rx = (ix + 1) * chunksize
            patch_sublist = np.array(patch_list[lx:rx])
            sublist_size = rx - lx
            HESAFF_CLIB.extractDescFromPatches(sublist_size, patch_h, patch_w, patch_sublist, vecs_array[lx:rx])
        last_size = num_patches - rx
        if last_size > 0:
            lx = rx
            rx = lx + last_size
            patch_sublist = np.array(patch_list[lx:rx])
            sublist_size = rx - lx
            HESAFF_CLIB.extractDescFromPatches(sublist_size, patch_h, patch_w, patch_sublist, vecs_array[lx:rx])
    #print('vecs_array = %r' % (vecs_array,))
    return vecs_array


def detect_kpts_list(image_paths_list, **kwargs):
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

    Example:
        >>> # ENABLE_DOCTEST
        >>> from pyhesaff._pyhesaff import *  # NOQA
        >>> import utool as ut
        >>> lena_fpath = ut.grab_test_imgpath('lena.png')  # ut.grab_file_url('http://i.imgur.com/JGrqMnV.png', fname='lena.png')
        >>> image_paths_list = [ut.grab_test_imgpath('carl.jpg'), ut.grab_test_imgpath('star.png'), lena_fpath]  #, ut.grab_test_imgpath('carl.jpg')]
        >>> (kpts_list, vecs_list) = detect_kpts_list(image_paths_list)
        >>> #print((kpts_list, vecs_list))
        >>> print(ut.depth_profile(kpts_list))
        >>> # Assert that the normal version agrees
        >>> serial_list = [detect_kpts(fpath) for fpath in image_paths_list]
        >>> kpts_list2 = ut.get_list_column(serial_list, 0)
        >>> vecs_list2 = ut.get_list_column(serial_list, 1)
        >>> print(ut.depth_profile(kpts_list2))
        >>> diff_kpts = [kpts - kpts2 for kpts, kpts2 in zip(kpts_list, kpts_list2)]
        >>> diff_vecs = [vecs - vecs2 for vecs, vecs2 in zip(vecs_list, vecs_list2)]
        >>> #diff_kpts_xs = [np.nonzero(d.sum(axis=1))[0] for d in diff_kpts]
        >>> #diff_vecs_xs = [np.nonzero(d.sum(axis=1))[0] for d in diff_vecs]
        >>> #print('+---- kpts_list')
        >>> #print(ut.list_str(kpts_list, precision=1, suppress_small=True))
        >>> #print('+---- kpts_list2')
        >>> #print(ut.list_str(kpts_list2, precision=1, suppress_small=True))
        >>> #print('+---- diff_kpts')
        >>> #print(ut.list_str(diff_kpts_xs, precision=1, suppress_small=True))
        >>> #print('+---- vecs_list')
        >>> #print(ut.list_str(vecs_list, precision=1, suppress_small=True))
        >>> #print('+---- vecs_list2')
        >>> #print(ut.list_str(vecs_list2, precision=1, suppress_small=True))
        >>> #print('+---- diff_vecs')
        >>> #print(ut.list_str(diff_vecs_xs, precision=1, suppress_small=True))
        >>> assert all([x.sum() == 0 for x in diff_kpts]), 'inconsistent results'
        >>> assert all([x.sum() == 0 for x in diff_vecs]), 'inconsistent results'

    """
    # Get Num Images
    nImgs = len(image_paths_list)

    # Cast string list to C
    if six.PY2:
        realpaths_list = list(map(realpath, image_paths_list))
    if six.PY3:
        realpaths_list = [realpath(path).encode('ascii') for path in image_paths_list]

    c_strs = _cast_strlist_to_C(realpaths_list)

    # Get algorithm parameters
    hesaff_params = HESAFF_PARAM_DICT.copy()
    hesaff_params.update(kwargs)
    hesaff_args = hesaff_params.values()  # pass all parameters to HESAFF_CLIB

    NEW = True
    if not NEW:
        # Allocate empty array pointers for each image
        kpts_ptr_array = np.empty(nImgs, dtype=kpts_t)  # array of float arrays
        vecs_ptr_array = np.empty(nImgs, dtype=vecs_t)  # array of byte arrays
        nDetect_array = np.empty(nImgs, dtype=int_t)  # array of detections per image
        # Detect keypoints in parallel
        HESAFF_CLIB.detectKeypointsList(nImgs, c_strs,
                                        kpts_ptr_array, vecs_ptr_array,
                                        nDetect_array, *hesaff_args)

        # TODO: this should return the sizes of the arrays, then numpy should allocate
        # the memory and then the memory should be copied from C to numpy using
        # two calls.

        #try:
        # Cast keypoint array to list of numpy keypoints
        kpts_list = extract_2darr_list(nDetect_array, kpts_ptr_array, kpts_t, kpts_dtype, KPTS_DIM)
        # Cast decsriptor array to list of numpy decsriptors
        vecs_list = extract_2darr_list(nDetect_array, vecs_ptr_array, vecs_t, vecs_dtype, DESC_DIM)
        #except Exception:
        #    raise
        #    #argtypes = HESAFF_CLIB.detectKeypointsList.argtypes
        #    #ut.printex(ex, 'error extracting 2darr list', keys=[
        #    #    'argtypes'
        #    #])
        #    #ut.embed()

        #kpts_list = [arrptr_to_np(kpts_ptr, (len_, KPTS_DIM), kpts_t, kpts_dtype)
        #             for (kpts_ptr, len_) in zip(kpts_ptr_array, nDetect_array)]
        #vecs_list = [arrptr_to_np(vecs_ptr, (len_, DESC_DIM), vecs_t, vecs_dtype)
        #             for (vecs_ptr, len_) in zip(vecs_ptr_array, nDetect_array)]
    else:
        length_array = np.empty(nImgs, dtype=int_t)
        detector_array = HESAFF_CLIB.detectKeypointsListStep1(nImgs, c_strs, *hesaff_args)
        HESAFF_CLIB.detectKeypointsListStep2(nImgs, detector_array, length_array)
        total_pts = length_array.sum()
        flat_kpts_ptr, flat_vecs_ptr = _allocate_kpts_and_vecs(nImgs * total_pts)
        # TODO: get this working
        offset_array = np.roll(length_array.cumsum(), 1).astype(int_t)  # np.array([0] + length_array.cumsum().tolist()[0:-1], int_t)
        offset_array[0] = 0
        HESAFF_CLIB.detectKeypointsListStep3(nImgs, detector_array, length_array, offset_array, flat_kpts_ptr, flat_vecs_ptr)

        # reshape into jagged arrays
        kpts_list = [flat_kpts_ptr[o:o + l] for o, l in zip(offset_array, length_array)]
        vecs_list = [flat_vecs_ptr[o:o + l] for o, l in zip(offset_array, length_array)]

    return kpts_list, vecs_list


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


def arrptr_to_np(c_arrptr, shape, arr_t, dtype):
    """
    Casts an array pointer from C to numpy

    Args:
        c_arrptr (uint64): a pointer to an array returned from C
        shape (tuple): shape of the underlying array being pointed to
        arr_t (PyCSimpleType): the ctypes datatype of c_arrptr
        dtype (dtype): numpy datatype the array will be to cast into

    CommandLine:
        python2 -m pyhesaff._pyhesaff --test-detect_kpts_list:0 --rebuild-hesaff
        python2 -m pyhesaff._pyhesaff --test-detect_kpts_list:0
        python3 -m pyhesaff._pyhesaff --test-detect_kpts_list:0

    """
    try:
        byte_t = ctypes.c_char
        itemsize_ = dtype().itemsize  # size of a single byte
        dtype_t = byte_t * itemsize_  # datatype of array elements
        dtype_ptr_t = C.POINTER(dtype_t)  # size of each item
        typed_c_arrptr = c_arrptr.astype(int)
        c_arr = C.cast(typed_c_arrptr, dtype_ptr_t)   # cast to ctypes
        #raise Exception('fuuu. Why does 2.7 work? Why does 3.4 not!?!!!')
        np_arr = np.ctypeslib.as_array(c_arr, shape)       # cast to numpy
        np_arr.dtype = dtype                               # fix numpy dtype
    except Exception as ex:
        import utool as ut
        #utool.embed()
        varnames = sorted(list(locals().keys()))
        vartypes = [(type, name) for name in varnames]
        spaces    = [None for name in varnames]
        c_arrptr_dtype = c_arrptr.dtype  # NOQA
        #key_list = list(zip(varnames, vartypes, spaces))
        key_list = ['c_arrptr_dtype'] + 'c_arrptr, shape, arr_t, dtype'.split(', ')
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
        ut.printex(ex, keys=key_list)
        raise
    return np_arr


def detect_kpts_in_image(img, **kwargs):
    r"""
    Takes a preloaded image and detects keypoints

    Args:
        img (ndarray[uint8_t, ndim=2]):  image data, should be in BGR or grayscale

    Returns:
        tuple: (kpts, vecs)

    CommandLine:
        python -m pyhesaff._pyhesaff --test-detect_kpts_in_image --show
        python -m pyhesaff._pyhesaff --test-detect_kpts --show
        python -m pyhesaff._pyhesaff --test-detect_kpts_in_image --rebuild-hesaff --show --no-rmbuild

    Example:
        >>> # ENABLE_DOCTEST
        >>> from pyhesaff._pyhesaff import *  # NOQA
        >>> import vtool as vt
        >>> img_fpath = ut.grab_test_imgpath('lena.png')
        >>> img= vt.imread(img_fpath)
        >>> (kpts, vecs) = detect_kpts_in_image(img)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.interact_keypoints.ishow_keypoints(img, kpts, vecs, ori=True, ell_alpha=.4, color='distinct')
        >>> pt.set_figtitle('Detect Kpts in Image')
        >>> ut.show_if_requested()
    """
    #Valid keyword arguments are: + str(HESAFF_PARAM_DICT.keys())
    hesaff_ptr = _new_image_hesaff(img, **kwargs)
    if __DEBUG__:
        print('[hes] detect')
    nKpts = HESAFF_CLIB.detect(hesaff_ptr)  # Get num detected
    if __DEBUG__:
        print('[hes] allocate')
    kpts, vecs = _allocate_kpts_and_vecs(nKpts)  # Allocate arrays
    if __DEBUG__:
        print('[hes] export')
    HESAFF_CLIB.exportArrays(hesaff_ptr, nKpts, kpts, vecs)  # Populate arrays
    return kpts, vecs

# detect_kpts_in_fpath = detect_kpts


def detect_kpts2(img_or_fpath, **kwargs):
    if isinstance(img_or_fpath, six.string_types):
        return detect_kpts(img_or_fpath, **kwargs)
    else:
        return detect_kpts_in_image(img_or_fpath, **kwargs)


#@profile
def detect_kpts(img_fpath, use_adaptive_scale=False, nogravity_hack=False, **kwargs):
    r"""
    main driver function for detecting hessian affine keypoints.
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
        python -m pyhesaff._pyhesaff --test-detect_kpts
        python -m pyhesaff._pyhesaff --test-detect_kpts --show
        python -m pyhesaff._pyhesaff --test-detect_kpts --show --fname star.png
        python -m pyhesaff._pyhesaff --test-detect_kpts --show --fname lena.png
        python -m pyhesaff._pyhesaff --test-detect_kpts --show --fname carl.jpg

        python -m pyhesaff._pyhesaff --test-detect_kpts --show --fname lena.png --rotation_invariance
        python -m pyhesaff._pyhesaff --test-detect_kpts --show --fname lena.png --affine_invariance

        python -m pyhesaff._pyhesaff --test-detect_kpts --show --fname easy1.png --no-affine_invariance
        python -m pyhesaff._pyhesaff --test-detect_kpts --show --fname easy1.png --no-affine_invariance --numberOfScales=1 --verbose
        python -m pyhesaff._pyhesaff --test-detect_kpts --show --fname easy1.png --no-affine_invariance --scale-max=100 --verbose
        python -m pyhesaff._pyhesaff --test-detect_kpts --show --fname easy1.png --no-affine_invariance --scale-min=20 --verbose
        python -m pyhesaff._pyhesaff --test-detect_kpts --show --fname easy1.png --no-affine_invariance --scale-min=100 --verbose
        python -m pyhesaff._pyhesaff --test-detect_kpts --show --fname easy1.png --no-affine_invariance --scale-max=20 --verbose

        python -m vtool.test_constrained_matching --test-visualize_matches --show
        python -m vtool.tests.dummy --test-testdata_ratio_matches --show --


        python -m pyhesaff._pyhesaff --test-detect_kpts --show --fname easy1.png --affine_invariance --verbose --rebuild-hesaff --scale-min=35 --scale-max=40 --no-rmbuild

        python -m pyhesaff._pyhesaff --test-detect_kpts --show --fname easy1.png --affine_invariance --verbose --scale-min=35 --scale-max=40&
        python -m pyhesaff._pyhesaff --test-detect_kpts --show --fname easy1.png --no-affine_invariance --verbose --scale-min=35 --scale-max=40&
        python -m pyhesaff._pyhesaff --test-detect_kpts --show --fname easy1.png --no-affine_invariance --verbose --scale-max=40 --darken .5
        python -m pyhesaff._pyhesaff --test-detect_kpts --show --fname easy1.png --no-affine_invariance --verbose --scale-max=30 --darken .5
        python -m pyhesaff._pyhesaff --test-detect_kpts --show --fname easy1.png --affine_invariance --verbose --scale-max=30 --darken .5


        # DENSE KEYPOINTS
        python -m pyhesaff._pyhesaff --test-detect_kpts --show --fname lena.png --no-affine-invariance --numberOfScales=1 --maxPyramidLevels=1 --use_dense --dense_stride=64
        python -m pyhesaff._pyhesaff --test-detect_kpts --show --fname lena.png --no-affine-invariance --numberOfScales=1 --maxPyramidLevels=1 --use_dense --dense_stride=64 --rotation-invariance
        python -m pyhesaff._pyhesaff --test-detect_kpts --show --fname lena.png --affine-invariance --numberOfScales=1 --maxPyramidLevels=1 --use_dense --dense_stride=64
        python -m pyhesaff._pyhesaff --test-detect_kpts --show --fname lena.png --no-affine-invariance --numberOfScales=3 --maxPyramidLevels=2 --use_dense --dense_stride=64


    Example0:
        >>> # ENABLE_DOCTEST
        >>> # Test simple detect
        >>> from pyhesaff._pyhesaff import *  # NOQA
        >>> import plottool as pt
        >>> import utool as ut
        >>> import vtool as vt
        >>> #img_fpath = make_small_test_img_fpath()
        >>> #img_fpath = ut.grab_test_imgpath('lena.png')
        >>> TAU = 2 * np.pi
        >>> fpath = ut.grab_test_imgpath(ut.get_argval('--fname', default='lena.png'))
        >>> theta = ut.get_argval('--theta', float, 0)  # TAU * 3 / 8)
        >>> img_fpath = vt.rotate_image_on_disk(fpath, theta)
        >>> kwargs = ut.parse_dict_from_argv(get_hesaff_default_params())
        >>> (kpts_list, vecs_list) = detect_kpts(img_fpath, **kwargs)
        >>> #print(kpts_list)
        >>> #print(vecs_list)
        >>> kpts = kpts_list
        >>> vecs = vecs_list
        >>> # Show keypoints
        >>> #pt.figure(fnum=1, doclf=True, docla=True)
        >>> imgBGR = vt.imread(img_fpath)
        >>> #pt.imshow(imgBGR)
        >>> ut.quit_if_noshow()
        >>> #pt.draw_kpts2(kpts,
        >>> pt.interact_keypoints.ishow_keypoints(imgBGR, kpts, vecs, ori=True, ell_alpha=.4, color='distinct')
        >>> pt.set_figtitle('Detect Kpts in Image')
        >>> pt.show_if_requested()
    """
    #Valid keyword arguments are: + str(HESAFF_PARAM_DICT.keys())
    if __DEBUG__:
        print('[hes] Detecting Keypoints')
        print('[hes] use_adaptive_scale=%r' % (use_adaptive_scale,))
        print('[hes] nogravity_hack=%r' % (nogravity_hack,))
        print('[hes] kwargs=%s' % (ut.dict_str(kwargs),))
    hesaff_ptr = _new_fpath_hesaff(img_fpath, **kwargs)
    if __DEBUG__:
        print('[hes] detect')
    nKpts = HESAFF_CLIB.detect(hesaff_ptr)  # Get num detected
    if __DEBUG__:
        print('[hes] allocate')
    kpts, vecs = _allocate_kpts_and_vecs(nKpts)  # Allocate arrays
    if __DEBUG__:
        print('[hes] export')
    HESAFF_CLIB.exportArrays(hesaff_ptr, nKpts, kpts, vecs)  # Populate arrays
    if use_adaptive_scale:  # Adapt scale if requested
        #print('Adapting Scale')
        if __DEBUG__:
            print('[hes] adapt_scale')
        kpts, vecs = adapt_scale(img_fpath, kpts)
    if nogravity_hack:
        if __DEBUG__:
            print('[hes] adapt_rotation')
        kpts, vecs = vtool_adapt_rotation(img_fpath, kpts)
    return kpts, vecs


def test_rot_invar():
    r"""
    CommandLine:
        mingw_build.bat
        python -m pyhesaff._pyhesaff --test-test_rot_invar --show --rebuild-hesaff --no-rmbuild
        python -m pyhesaff._pyhesaff --test-test_rot_invar --show --nocpp

        python -m vtool.tests.dummy --test-testdata_ratio_matches --show --ratio_thresh=1.0 --rotation_invariance --rebuild-hesaff
        python -m vtool.tests.dummy --test-testdata_ratio_matches --show --ratio_thresh=1.1 --rotation_invariance --rebuild-hesaff

    Example:
        >>> # ENABLE_DOCTEST
        >>> from pyhesaff._pyhesaff import *  # NOQA
        >>> test_rot_invar()
    """
    #from pyhesaff._pyhesaff import *  # NOQA
    import cv2
    import utool as ut
    import vtool as vt
    import plottool as pt
    #img_fpath = ut.grab_test_imgpath('jeff.png')
    TAU = 2 * np.pi
    fnum = pt.next_fnum()
    NUM_PTS = 5  # 9
    theta_list = np.linspace(0, TAU, NUM_PTS, endpoint=False)
    nRows, nCols = pt.get_square_row_cols(len(theta_list), fix=True)
    next_pnum = pt.make_pnum_nextgen(nRows, nCols)
    # Expand the border a bit around star.png
    pad_ = 100
    img_fpath = ut.grab_test_imgpath('star.png')
    img_fpath2 = vt.pad_image_on_disk(img_fpath, pad_, value=26)
    for theta in theta_list:
        print('-----------------')
        print('theta = %r' % (theta,))
        #theta = ut.get_argval('--theta', type_=float, default=TAU * 3 / 8)
        img_fpath = vt.rotate_image_on_disk(img_fpath2, theta, borderMode=cv2.BORDER_REPLICATE)
        if not ut.get_argflag('--nocpp'):
            (kpts_list_ri, vecs_list2) = detect_kpts(img_fpath, rotation_invariance=True)
            kpts_ri = ut.strided_sample(kpts_list_ri, 2)
        (kpts_list_gv, vecs_list1) = detect_kpts(img_fpath, rotation_invariance=False)
        kpts_gv = ut.strided_sample(kpts_list_gv, 2)
        # find_kpts_direction
        imgBGR = vt.imread(img_fpath)
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
        if not ut.get_argflag('--nocpp'):
            if len(kpts_ri) > 0:
                pt.draw_kpts2(kpts_ri, rect=rect, ell=ell, ori=True, ell_color=pt.RED, ell_linewidth=5.5)
        if len(kpts_ripy) > 0:
            pt.draw_kpts2(kpts_ripy, rect=rect, ell=ell,  ori=True, ell_color=pt.GREEN, ell_linewidth=3.5)
        #print('\n'.join(vt.get_ori_strs(np.vstack([kpts_gv, kpts_ri, kpts_ripy]))))
        #ut.embed(exec_lines=['pt.update()'])
    pt.set_figtitle('green=python, red=C++')
    pt.show_if_requested()


#def make_small_test_img_fpath():
#    import vtool as vt
#    star = np.array(vt.get_star2_patch() * 255)
#    star = vt.resize(star, (64, 64), vt.CV2_INTERPOLATION_TYPES['nearest'])
#    img_fpath = ut.get_app_resource_dir('vtool', 'star.png')
#    vt.imwrite(img_fpath, star)
#    return img_fpath


def vtool_adapt_rotation(img_fpath, kpts):
    # rotation invariance in python
    import vtool.patch as ptool
    import vtool.image as gtool
    imgBGR = gtool.imread(img_fpath)
    kpts2 = ptool.find_kpts_direction(imgBGR, kpts)
    vecs2 = extract_vecs(img_fpath, kpts2)
    return kpts2, vecs2


#@profile
def adapt_scale(img_fpath, kpts):
    import vtool.ellipse as etool
    nScales = 16
    nSamples = 16
    low, high = -1, 2
    kpts2 = etool.adaptive_scale(img_fpath, kpts, nScales, low, high, nSamples)
    # passing in 0 orientation results in gravity vector direction keypoint
    vecs2 = extract_vecs(img_fpath, kpts2)
    return kpts2, vecs2


#if __name__ == '__main__':
#    """
#    CommandLine:
#        python -c "import utool, pyhesaff._pyhesaff; utool.doctest_funcs(pyhesaff._pyhesaff, allexamples=True)"
#        python -c "import utool, pyhesaff._pyhesaff; utool.doctest_funcs(pyhesaff._pyhesaff)"
#        python pyhesaff\_pyhesaff.py
#        python pyhesaff\_pyhesaff.py --allexamples
#    """
#    import multiprocessing
#    multiprocessing.freeze_support()  # for win32
#    import utool as ut  # NOQA
#    ut.doctest_funcs()


if __name__ == '__main__':
    """
    CommandLine:
        python -m pyhesaff._pyhesaff
        python -m pyhesaff._pyhesaff --allexamples
        python -m pyhesaff._pyhesaff --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
