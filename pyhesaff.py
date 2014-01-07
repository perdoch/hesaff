import matplotlib
matplotlib.use('Qt4Agg', warn=True, force=True)
import ctypes as C
#from ctypes.util import find_library
import numpy as np
from os.path import join, exists, abspath, dirname, normpath, expanduser, split
import os
import sys


def get_lib_fname_list(libname):
    'returns possible library names given the platform'
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
    'returns possible lib locations'
    get_lib_dpath_list = [root_dir,
                          join(root_dir, 'lib'),
                          join(root_dir, 'build'),
                          join(root_dir, 'build', 'lib')]
    return get_lib_dpath_list


def find_lib_fpath(libname, root_dir, recurse_down=True):
    lib_fname_list = get_lib_fname_list(libname)
    while root_dir is not None:
        for lib_fname in lib_fname_list:
            for lib_dpath in get_lib_dpath_list(root_dir):
                lib_fpath = normpath(join(lib_dpath, lib_fname))
                #print('testing: %r' % lib_fpath)
                if exists(lib_fpath):
                    print('using: %r' % lib_fpath)
                    return lib_fpath
            _new_root = dirname(root_dir)
            if _new_root == root_dir:
                root_dir = None
                break
            else:
                root_dir = _new_root
        if not recurse_down:
            break
    raise ImportError('Cannot find dynamic library.')


def load_library2(libname, root_dir):
    lib_fpath = find_lib_fpath(libname, root_dir)
    try:
        clib = C.cdll[lib_fpath]
    except Exception as ex:
        print('Caught exception: %r' % ex)
        raise ImportError('Cannot load dynamic library. Did you compile HESAFF?')
    return clib

#def load_hesaff_lib():
# LOAD LIBRARY
if '__file__' in vars():
    root_dir = abspath(dirname(__file__))
else:
    root_dir = abspath(dirname(os.getcwd()))
libname = 'hesaff'
hesaff_lib = load_library2(libname, root_dir)

# Define types
#str or tuple of str
#Array flags; may be one or more of:
#C_CONTIGUOUS / C / CONTIGUOUS
#F_CONTIGUOUS / F / FORTRAN
#OWNDATA / O
#WRITEABLE / W
#ALIGNED / A
#UPDATEIFCOPY / U
# numpy dtypes
kpts_dtype = np.float32
desc_dtype = np.uint8
# ctypes
obj_t = C.c_void_p
kpts_t = np.ctypeslib.ndpointer(dtype=kpts_dtype, ndim=2, flags='aligned, c_contiguous, writeable')
desc_t = np.ctypeslib.ndpointer(dtype=desc_dtype, ndim=2, flags='aligned, c_contiguous, writeable')
str_t = C.c_char_p
int_t = C.c_int

# Test
hesaff_lib.make_hesaff.restype = obj_t
hesaff_lib.make_hesaff.argtypes = [str_t]
#
hesaff_lib.detect.restype = int_t
hesaff_lib.detect.argtypes = [obj_t]
#
hesaff_lib.exportArrays.restype = None
hesaff_lib.exportArrays.argtypes = [obj_t, int_t, kpts_t, desc_t]
#
hesaff_lib.extractDesc.restype = None
hesaff_lib.extractDesc.argtypes = [obj_t, int_t, kpts_t, desc_t]

# If profiling with kernprof.py
try:
    profile  # NoQA
except NameError:
    profile = lambda func: func


@profile
def detect_hesaff_kpts(img_fpath, dict_args={}):
    # Make detector and read image
    hesaff_ptr = hesaff_lib.make_hesaff(abspath(img_fpath))
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


if __name__ == '__main__':
    def add_hotspotter_to_path():
        # Look for hotspotter in ~/code
        hotspotter_dir = join(expanduser('~'), 'code', 'hotspotter')
        if not exists(hotspotter_dir):
            print('[pyhesaff] hotspotter_dir=%r DOES NOT EXIST!' % (hotspotter_dir,))
        # Append hotspotter location (not dir) to PYTHON_PATH (i.e. sys.path)
        hotspotter_location = split(hotspotter_dir)[0]
        sys.path.append(hotspotter_location)
    add_hotspotter_to_path()

    import multiprocessing
    multiprocessing.freeze_support()
    from hotspotter import fileio as io
    from hotspotter import draw_func2 as df2

    img_fpath = 'build/zebra.jpg'
    img_fpath = '../lena.png'

    img_fpath = abspath('lena.png')
    img_fpath = abspath('zebra.jpg')
    image = io.imread(img_fpath)
    kpts, desc = detect_hesaff_kpts(img_fpath)
    print('detected')
    print('showing')
    df2.imshow(image)
    print('drawing')
    #kpts = np.array([[100, 100, 20, 10, 20]])
    fx = 100
    stride = 42
    kxs = np.arange(len(kpts))
    np.random.shuffle(kxs)
    #kpts2 = kpts[::stride, :]
    kpts2 = kpts[kxs[0:10]]
    cols = df2.distinct_colors(len(kpts2))
    df2.draw_kpts2(kpts2, ell_alpha=.6, ell_linewidth=2,
                   ell_color=cols, rect=True)
    expand_scale(kpts2, 2)
    df2.draw_kpts2(kpts2, ell_alpha=.5, ell_linewidth=3,
                   ell_color=cols, rect=True)

    print('present')
    exec_str = df2.present()
    #print(exec_str)
    exec(exec_str)
#C:\Users\joncrall\code\ell_desc
