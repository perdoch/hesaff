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
    import extract_patch
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
    extract_patch.get_kp_border(rchip, kp)

    from hotspotter import interaction
    interaction.rrr()
    interaction.interact_keypoints(imgL, sel_kpts, sel_desc, fnum=3)
    interaction.interact_keypoints(imgL, kpts, desc, fnum=2)

    draw_expanded_scales(imgL, sel_kpts, exkpts, exdesc_)


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

    viz.show_keypoints(imgL, exkpts_, fnum=fnum, pnum=(nRows + nPreRows, 1, 1))

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
    test_locals = test_hesaff(n, fnum, adaptive=True)
    # They seem to work
    #test_locals = test_hesaff(n, fnum + 2, new_exe=True)
    #test_locals = test_hesaff(n, fnum + 3, old_exe=True)
    #test_locals = test_hesaff(n, fnum + 1, use_exe=True, reextract=True)

    #exec(helpers.execstr_dict(test_locals, 'test_locals'))

    exec(df2.present())
