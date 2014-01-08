from __future__ import print_function, division
if '__file__' not in vars():
    import os
    os.chdir('C:\Users\joncrall\code\hotspotter\_tpl\extern_feat')

# Standard
#from ctypes.util import find_library
import subprocess
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


def extract_hesaff_kpts(img_fpath, kpts, **kwargs):
    hesaff_ptr = hesaff_lib.new_hesaff(realpath(img_fpath))
    nKpts = len(kpts)
    # allocate memory for new descriptors
    desc = np.empty((nKpts, 128), desc_dtype)
    # extract descriptors at given locations
    hesaff_lib.extractDesc(hesaff_ptr, nKpts, kpts, desc)
    return desc


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


###############
# Old reading
###############
EXE_EXT = {'win32': '.exe', 'darwin': '.mac', 'linux2': ''}[sys.platform]
if not '__file__' in vars():
    __file__ = os.path.realpath('pyhesaff.py')
EXE_PATH = realpath(dirname(__file__))
EXE_FPATH = join(EXE_PATH, 'hesaffexe' + EXE_EXT)
if not os.path.exists(EXE_FPATH):
    EXE_FPATH = join(EXE_PATH, 'build', 'hesaffexe' + EXE_EXT)
if not os.path.exists(EXE_PATH):
    EXE_FPATH = join(EXE_PATH, '_tpl', 'extern_feat', 'hesaffexe' + EXE_EXT)


def execute_extern(cmd):
    'Executes a system call'
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (out, err) = proc.communicate()
    if proc.returncode != 0:
        raise Exception('\n'.join(['* External detector returned 0',
                                   '* Failed calling: ' + cmd, '* Process output: ',
                                   '------------------', out, '------------------']))


def detect_hesaff_kpts_exeversion(rchip_fpath, dict_args):
    'Runs external perdoch detector'
    outname = rchip_fpath + '.hesaff.sift'
    args = '"' + rchip_fpath + '"'
    cmd  = EXE_FPATH + ' ' + args
    execute_extern(cmd)
    kpts, desc = read_text_feat_file(outname)
    if len(kpts) == 0:
        return np.empty((0, 5), dtype=kpts_dtype), np.empty((0, 5), dtype=desc_dtype)
    kpts = fix_kpts_hack(kpts)
    kpts, desc = filter_kpts_scale(kpts, desc, **dict_args)
    return kpts, desc


def read_text_feat_file(outname, be_clean=True):
    'Reads output from external keypoint detectors like hesaff'
    file = open(outname, 'r')
    # Read header
    ndims = int(file.readline())  # assert ndims == 128
    nkpts = int(file.readline())  #
    lines = file.readlines()
    file.close()
    if be_clean:
        os.remove(outname)
    # Preallocate output
    kpts = np.zeros((nkpts, 5), dtype=kpts_dtype)
    desc = np.zeros((nkpts, ndims), dtype=desc_dtype)
    for kx, line in enumerate(lines):
        data = line.split(' ')
        kpts[kx, :] = np.array([kpts_dtype(_) for _ in data[0:5]], dtype=kpts_dtype)
        desc[kx, :] = np.array([desc_dtype(_) for _ in data[5:]],  dtype=desc_dtype)
    return (kpts, desc)


def filter_kpts_scale(kpts, desc, scale_max=None, scale_min=None, **kwargs):
    #max_scale=1E-3, min_scale=1E-7
    #from hotspotter import helpers
    if len(kpts) == 0 or \
       scale_max is None or scale_min is None or\
       scale_max < 0 or scale_min < 0 or\
       scale_max < scale_min:
        return kpts, desc
    acd = kpts.T[2:5]
    det_ = acd[0] * acd[2]
    scale = np.sqrt(det_)
    #print('scale.stats()=%r' % helpers.printable_mystats(scale))
    #is_valid = np.bitwise_and(scale_min < scale, scale < scale_max).flatten()
    is_valid = np.logical_and(scale_min < scale, scale < scale_max).flatten()
    #scale = scale[is_valid]
    kpts = kpts[is_valid]
    desc = desc[is_valid]
    #print('scale.stats() = %s' % str(helpers.printable_mystats(scale)))
    return kpts, desc


def expand_invET(invET):
    # Put the inverse elleq in a list of matrix structure
    e11 = invET[0]
    e12 = invET[1]
    e21 = invET[1]
    e22 = invET[2]
    invE_list = np.array(((e11, e12), (e21, e22))).T
    return invE_list


def fix_kpts_hack(kpts, method=1):
    ''' Transforms:
        [E_a, E_b]        [A_a,   0]
        [E_b, E_d]  --->  [A_c, A_d]
    '''
    'Hack to put things into acd foramat'
    xyT   = kpts.T[0:2]
    invET = kpts.T[2:5]
    # Expand into full matrix
    invE_list = expand_invET(invET)
    # Decompose using singular value decomposition
    invXWYt_list = [np.linalg.svd(invE) for invE in invE_list]
    # Rebuild the ellipse -> circle matrix
    A_list = [invX.dot(np.diag(1 / np.sqrt(invW))) for (invX, invW, invYt) in invXWYt_list]
    # Flatten the shapes for fast rectification
    abcd  = np.vstack([A.flatten() for A in A_list])
    # Rectify up
    acd = rectify_up_abcd(abcd)
    kpts = np.vstack((xyT, acd.T)).T
    return kpts


def rectify_up_abcd(abcd):
    ''' Based on:
    void rectifyAffineTransformationUpIsUp(float &a11, float &a12, float &a21, float &a22)
    {
    double a = a11, b = a12, c = a21, d = a22;
    double det = sqrt(abs(a*d-b*c));
    double b2a2 = sqrt(b*b + a*a);
    a11 = b2a2/det;             a12 = 0;
    a21 = (d*b+c*a)/(b2a2*det); a22 = det/b2a2;
    } '''
    (a, b, c, d) = abcd.T
    absdet_ = np.abs(a * d - b * c)
    #sqtdet_ = np.sqrt(absdet_)
    b2a2 = np.sqrt(b * b + a * a)
    # Build rectified ellipse matrix
    a11 = b2a2
    a21 = (d * b + c * a) / (b2a2)
    a22 = absdet_ / b2a2
    acd = np.vstack([a11, a21, a22]).T
    return acd


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
    from hotspotter import vizualizations as viz
    from hotspotter import helpers

    # Read Image
    img_fpath = realpath('lena.png')
    image = io.imread(img_fpath)

    def spaced_elements(list_, n):
        indexes = np.arange(len(list_))
        stride = len(indexes) // n
        return list_[indexes[0:-1:stride]]

    def test_hesaff(n=None, fnum=1, use_exe=False, reextract=False):
        try:
            # Select kpts
            if use_exe:
                title = 'exe'
                func = detect_hesaff_kpts_exeversion
            else:
                func = detect_hesaff_kpts
                title = 'lib'
            with helpers.Timer(msg=title):
                kpts, desc = func(img_fpath, {})

            if reextract:
                with helpers.Timer(msg='reextract'):
                    desc = extract_hesaff_kpts(img_fpath, kpts)
            kpts_ = kpts if n is None else spaced_elements(kpts, n)
            desc_ = desc if n is None else spaced_elements(desc, n)
            # Print info
            np.set_printoptions(threshold=5000, linewidth=5000, precision=3)
            print('----')
            print('detected %d keypoints' % len(kpts))
            print('drawing %d/%d kpts' % (len(kpts_), len(kpts)))
            print(kpts_)
            print(desc_[:, 0:16])
            print('----')
            # Draw kpts
            viz.interact_keypoints(image, kpts_, desc_, fnum, nodraw=True)
            df2.set_figtitle(title)
            #df2.imshow(image, fnum=fnum)
            #df2.draw_kpts2(kpts_, ell_alpha=.9, ell_linewidth=4,
                           #ell_color='distinct', arrow=True, rect=True)
        except Exception as ex:
            import traceback
            traceback.format_exc()
            print(ex)
        return locals()

    n = 10
    fnum = 1
    test_locals = test_hesaff(n, fnum)
    # They seem to work
    #test_locals = test_hesaff(n, fnum + 1, use_exe=True)
    #test_locals = test_hesaff(n, fnum + 2, reextract=True)

    exec(helpers.execstr_dict(test_locals, 'test_locals'))

    exec(df2.present())
