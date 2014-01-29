from __future__ import print_function, division
# Standard
#from itertools import izip
#from ctypes.util import find_library
from os.path import realpath, dirname
import ctypes_interface
import ctypes as C
import collections
# Scientific
import numpy as np
# Hotspotter
from hscom import __common__
print, print_, print_on, print_off, rrr, profile, printDBG =\
    __common__.init(__name__, module_prefix='[hes]', DEBUG=False, initmpl=False)


#============================
# hesaff ctypes interface
#============================

# numpy dtypes
kpts_dtype = np.float32
desc_dtype = np.uint8
# ctypes
FLAGS_RW = 'aligned, c_contiguous, writeable'
obj_t     = C.c_void_p
kpts_t    = np.ctypeslib.ndpointer(dtype=kpts_dtype, ndim=2, flags=FLAGS_RW)
desc_t    = np.ctypeslib.ndpointer(dtype=desc_dtype, ndim=2, flags=FLAGS_RW)
str_t     = C.c_char_p
int_t     = C.c_int
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
