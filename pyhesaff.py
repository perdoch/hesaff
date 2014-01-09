from __future__ import print_function, division
# Standard
#from ctypes.util import find_library
from os.path import join, exists, realpath, dirname, normpath, expanduser, split
import ctypes as C
import sys
import collections
# Scientific
import numpy as np

##################
# Global constants
##################

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


# If profiling with kernprof.py
try:
    profile  # NoQA
except NameError:
    profile = lambda func: func


##################
# ctypes interface
##################


def get_lib_fname_list(libname):
    '''
    input <libname>: library name (e.g. 'hesaff', not 'libhesaff')
    returns <libnames>: list of plausible library file names
    '''
    if sys.platform == 'win32':
        libnames = ['lib' + libname + '.dll', libname + '.dll']
    elif sys.platform == 'darwin':
        libnames = ['lib' + libname + '.dylib']
    elif sys.platform == 'linux2':
        libnames = ['lib' + libname + '.so']
    else:
        raise Exception('Unknown operating system: %s' % sys.platform)
    return libnames


def get_lib_dpath_list(root_dir):
    '''
    input <root_dir>: deepest directory to look for a library (dll, so, dylib)
    returns <libnames>: list of plausible directories to look.
    '''
    'returns possible lib locations'
    get_lib_dpath_list = [root_dir,
                          join(root_dir, 'lib'),
                          join(root_dir, 'build'),
                          join(root_dir, 'build', 'lib')]
    return get_lib_dpath_list


def find_lib_fpath(libname, root_dir, recurse_down=True, verbose=False):
    'Search for the library'
    lib_fname_list = get_lib_fname_list(libname)
    tried_fpaths = []
    while root_dir is not None:
        for lib_fname in lib_fname_list:
            for lib_dpath in get_lib_dpath_list(root_dir):
                lib_fpath = normpath(join(lib_dpath, lib_fname))
                if exists(lib_fpath):
                    if verbose:
                        print('\n[c] Checked: '.join(tried_fpaths))
                    print('using: %r' % lib_fpath)
                    return lib_fpath
                else:
                    # Remember which candiate library fpaths did not exist
                    tried_fpaths.append(lib_fpath)
            _new_root = dirname(root_dir)
            if _new_root == root_dir:
                root_dir = None
                break
            else:
                root_dir = _new_root
        if not recurse_down:
            break

    msg = ('\n[C!] load_clib(libname=%r root_dir=%r, recurse_down=%r, verbose=%r)' %
           (libname, root_dir, recurse_down, verbose) +
           '\n[c!] Cannot FIND dynamic library')
    print(msg)
    print('\n[c!] Checked: '.join(tried_fpaths))
    raise ImportError(msg)


def load_clib(libname, root_dir):
    '''
    Does the work.
    Args:
        libname:  library name (e.g. 'hesaff', not 'libhesaff')

        root_dir: the deepest directory searched for the
                  library file (dll, dylib, or so).
    Returns:
        clib: a ctypes object used to interface with the library
    '''
    lib_fpath = find_lib_fpath(libname, root_dir)
    try:
        clib = C.cdll[lib_fpath]

        def def_cfunc(func_name, return_type, arg_type_list):
            'Function to define the types that python needs to talk to c'
            cfunc = getattr(clib, func_name)
            cfunc.restype = return_type
            cfunc.argtypes = arg_type_list
    except Exception as ex:
        print('[C!] Caught exception: %r' % ex)
        print('[C!] load_clib(libname=%r root_dir=%r)' % (libname, root_dir))
        raise ImportError('[C] Cannot LOAD dynamic library. Did you compile HESAFF?')
    return clib, def_cfunc


##################
# hesaff interface
##################


# THE ORDER OF THIS LIST IS IMPORTANT!
hesaff_typed_params = [
    # Pyramid Params
    (int_t, 'numberOfScales', 3),             # number of scale per octave
    (float_t, 'threshold', 16.0 / 3.0),           # noise dependent threshold on the response (sensitivity)
    (float_t, 'edgeEigenValueRatio', 10.0),   # ratio of the eigenvalues
    (int_t, 'border', 5),                     # number of pixels ignored at the border of image
    # Affine Shape Params
    (int_t, 'maxIterations', 16),             # number of affine shape interations
    (float_t, 'convergenceThreshold', 0.05),  # maximum deviation from isotropic shape at convergence
    (int_t, 'smmWindowSize', 19),             # width and height of the SMM mask
    (float_t, 'mrSize', 3.0 * np.sqrt(3.0)),    # size of the measurement region (as multiple of the feature scale)
    # SIFT params
    (int_t, 'spatialBins', 4),
    (int_t, 'orientationBins', 8),
    (float_t, 'maxBinValue', 0.2),
    # Shared params
    (float_t, 'initialSigma', 1.6),           # amount of smoothing applied to the initial level of first octave
    (int_t, 'patchSize', 41),                 # width and height of the patch
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
    hesaff_lib, def_cfunc = load_clib(libname, root_dir)

    def_cfunc(      'detect', int_t, [obj_t])
    def_cfunc('exportArrays',  None, [obj_t, int_t, kpts_t, desc_t])
    def_cfunc('extractDesc',   None, [obj_t, int_t, kpts_t, desc_t])
    def_cfunc(  'new_hesaff', obj_t, [str_t])
    def_cfunc(  'new_hesaff_from_params', obj_t, [str_t] + hesaff_param_types)
    return hesaff_lib

hesaff_lib = load_hesaff_clib()


def new_hesaff(img_fpath, **kwargs):
    # Make detector and read image
    hesaff_params = hesaff_param_dict.copy()
    hesaff_params.update(kwargs)
    #print('[pyhessaff] Override params: %r' % kwargs)
    #print(hesaff_params)
    hesaff_args = hesaff_params.values()
    hesaff_ptr = hesaff_lib.new_hesaff_from_params(realpath(img_fpath), *hesaff_args)
    #hesaff_ptr = hesaff_lib.new_hesaff(realpath(img_fpath))
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
    #print('[pyhesaff] extracting %d desc' % nKpts)
    #print('[pyhesaff] kpts.shape =%r' % (kpts.shape,))
    #print('[pyhesaff] kpts.dtype =%r' % (kpts.dtype,))
    #print('[pyhesaff] desc.shape =%r' % (desc.shape,))
    #print('[pyhesaff] desc.dtype =%r' % (desc.dtype,))
    #desc = np.require(desc, dtype=desc_dtype, requirements=['C_CONTIGUOUS', 'WRITABLE', 'ALIGNED'])
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
    #desc2 = np.empty((nKpts, 128), desc_dtype)
    #desc3 = np.empty((nKpts, 128), desc_dtype)
    #kpts = np.require(kpts, kpts_dtype, ['ALIGNED'])
    #desc = np.require(desc, desc_dtype, ['ALIGNED'])
    # Populate arrays
    hesaff_lib.exportArrays(hesaff_ptr, nKpts, kpts, desc)
    # TODO: Incorporate parameters
    # TODO: Scale Factor
    #hesaff_lib.extractDesc(hesaff_ptr, nKpts, kpts, desc2)
    #hesaff_lib.extractDesc(hesaff_ptr, nKpts, kpts, desc3)
    #print('[hesafflib] returned')
    return kpts, desc


def expand_scale(kpts, scale):
    kpts.T[2] *= scale
    kpts.T[3] *= scale
    kpts.T[4] *= scale
    return kpts


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


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    ensure_hotspotter()
    from hotspotter import fileio as io
    from hotspotter import draw_func2 as df2
    from hotspotter import vizualizations as viz
    from hotspotter import helpers

    # Read Image
    img_fpath = realpath('lena.png')
    image = io.imread(img_fpath)

    def spaced_elements(list_, n):
        if n is None:
            return 'list'
        indexes = np.arange(len(list_))
        stride = len(indexes) // n
        return list_[indexes[0:-1:stride]]

    def test_hesaff(n=None, fnum=1, **kwargs):
        reextract = kwargs.get('reextrac', False)
        new_exe = kwargs.get('new_exe', False)
        old_exe = kwargs.get('old_exe', False)
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
            # Print info
            np.set_printoptions(threshold=5000, linewidth=5000, precision=3)
            print('detected %d keypoints' % len(kpts))
            print('drawing %d/%d kpts' % (len(kpts_), len(kpts)))
            title += ' ' + str(len(kpts))
            #print(kpts_)
            #print(desc_[:, 0:16])
            # Draw kpts
            viz.interact_keypoints(image, kpts_, desc_, fnum, nodraw=True)
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

    n = None
    fnum = 1
    test_locals = test_hesaff(n, fnum)
    # They seem to work
    #test_locals = test_hesaff(n, fnum + 2, new_exe=True)
    #test_locals = test_hesaff(n, fnum + 3, old_exe=True)
    #test_locals = test_hesaff(n, fnum + 1, use_exe=True, reextract=True)

    #exec(helpers.execstr_dict(test_locals, 'test_locals'))

    exec(df2.present())
