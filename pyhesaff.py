from __future__ import print_function, division
if '__file__' not in vars():
    import os
    os.chdir('C:\Users\joncrall\code\hotspotter\_tpl\extern_feat')

# Standard
#from ctypes.util import find_library
from os.path import join, exists, realpath, dirname, normpath, expanduser, split
import ctypes as C
import os
import sys
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
desc_t = np.ctypeslib.ndpointer(dtype=desc_dtype, ndim=2, flags='aligned, c_contiguous, writeable')
str_t = C.c_char_p
int_t = C.c_int

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
    msg = '\n[c!] Cannot find dynamic library: %r' % libname
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
        print('[C] Caught exception: %r' % ex)
        raise ImportError('[C] Cannot load dynamic library. Did you compile HESAFF?')
    return clib, def_cfunc


##################
# hesaff interface
##################


def load_hesaff_clib():
    '''
    Specificially loads the hesaff lib and defines its functions
    '''
    root_dir = realpath(dirname(__file__)) if '__file__' in vars() else realpath(os.getcwd())
    libname = 'hesaff'
    hesaff_lib, def_cfunc = load_clib(libname, root_dir)

    def_cfunc(  'new_hesaff', obj_t, [str_t])
    def_cfunc(      'detect', int_t, [obj_t])
    def_cfunc('exportArrays',  None, [obj_t, int_t, kpts_t, desc_t])
    def_cfunc('extractDesc',   None, [obj_t, int_t, kpts_t, desc_t])
    return hesaff_lib

hesaff_lib = load_hesaff_clib()


@profile
def detect_hesaff_kpts(img_fpath, dict_args={}):
    # Make detector and read image
    hesaff_ptr = hesaff_lib.new_hesaff(realpath(img_fpath))
    # Return the number of keypoints detected
    nKpts = hesaff_lib.detect(hesaff_ptr)
    # Allocate arrays
    kpts = np.empty((nKpts, 5), kpts_dtype)
    desc = np.empty((nKpts, 128), desc_dtype)
    # Populate arrays
    hesaff_lib.exportArrays(hesaff_ptr, nKpts, kpts, desc)
    return kpts, desc


@profile
def test_hesaff_kpts(img_fpath, dict_args={}):
    # Make detector and read image
    hesaff_ptr = hesaff_lib.new_hesaff(realpath(img_fpath))
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
    sys.path.append(hotspotter_location)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    ensure_hotspotter()
    from hotspotter import fileio as io
    from hotspotter import draw_func2 as df2
    from hotspotter import helpers

    # Read Image
    img_fpath = realpath('lena.png')
    image = io.imread(img_fpath)

    def spaced_elements(list_, n):
        indexes = np.arange(len(list_))
        stride = len(indexes) // n
        return list_[indexes[0:-1:stride]]

    def test_hesaff(n=None, fnum=1):
        try:
            # Select kpts
            kpts, desc = detect_hesaff_kpts(img_fpath, {})
            kpts_ = kpts if n is None else spaced_elements(kpts, n)
            # Print info
            np.set_printoptions(threshold=5000, linewidth=5000, precision=3)
            print('----')
            print('detected %d keypoints' % len(kpts))
            print('drawing %d/%d kpts' % (len(kpts_), len(kpts)))
            print(kpts_)
            print('----')
            # Draw kpts
            df2.imshow(image, fnum=fnum)
            df2.draw_kpts2(kpts_, ell_alpha=.9, ell_linewidth=4,
                           ell_color='distinct', arrow=True, rect=True)
        except Exception as ex:
            import traceback
            traceback.format_exc()
            print(ex)
        return locals()

    n = 10
    fnum = 1
    test_locals = test_hesaff(n, fnum)
    exec(helpers.execstr_dict(test_locals, 'test_locals'))
