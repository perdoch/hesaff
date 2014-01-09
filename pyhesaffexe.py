from __future__ import print_function, division
# Standard
#from ctypes.util import find_library
import subprocess
from os.path import join, realpath, dirname
import os
import sys
# Scientific
import cv2
import numpy as np

###############
# Old interface
###############

kpts_dtype = np.float32
desc_dtype = np.uint8


def find_hesaff_fpath(exe_name='hesaffexe'):
    exe_ext = {'win32': '.exe', 'darwin': '.mac', 'linux2': ''}[sys.platform]
    if not '__file__' in vars():
        __file__ = os.path.realpath('pyhesaff.py')
    exe_path = realpath(dirname(__file__))
    exe_fpath = join(exe_path, exe_name + exe_ext)
    if not os.path.exists(exe_fpath):
        exe_fpath = join(exe_path, 'build', exe_name + exe_ext)
    if not os.path.exists(exe_path):
        exe_fpath = join(exe_path, '_tpl', 'extern_feat', exe_name + exe_ext)
    return exe_fpath

EXE_FPATH = find_hesaff_fpath()


def __cmd(args, verbose=True):
    print('[setup] Running: %r' % args)
    sys.stdout.flush()
    PIPE = subprocess.PIPE
    proc = subprocess.Popen(args, stdout=PIPE, stderr=PIPE, shell=True)
    if verbose:
        logged_list = []
        append = logged_list.append
        write = sys.stdout.write
        flush = sys.stdout.flush
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            write(line)
            flush()
            append(line)
        out = '\n'.join(logged_list)
        (out_, err) = proc.communicate()
    else:
        # Surpress output
        (out, err) = proc.communicate()
    # Make sure process if finished
    ret = proc.wait()
    if ret != 0:
        raise Exception('\n'.join(['* External detector returned 0',
                                   '* Failed calling: ' + str(args), '* Process output: ',
                                   '------------------', out, '------------------']))
    return out, err, ret


def detect_kpts(rchip_fpath, **kwargs):
    'Runs external perdoch detector'
    outname = rchip_fpath + '.hesaff.sift'
    args = '"' + rchip_fpath + '"'
    cmd  = EXE_FPATH + ' ' + args
    __cmd(cmd)
    kpts, desc = read_text_feat_file(outname)
    if len(kpts) == 0:
        return np.empty((0, 5), dtype=kpts_dtype), np.empty((0, 5), dtype=desc_dtype)
    kpts = fix_kpts_hack(kpts)
    kpts, desc = filter_kpts_scale(kpts, desc, **kwargs)
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


def svd(M):
    #U, S, V = np.linalg.svd(M)
    flags = cv2.SVD_FULL_UV
    S, U, V = cv2.SVDecomp(M, flags=flags)
    S = S.flatten()
    return U, S, V


def fix_kpts_hack(kpts, method=1):
    ''' Transforms:
        [E_a, E_b]        [A_a,   0]
        [E_b, E_d]  --->  [A_c, A_d]
    '''
    'Hack to put things into acd foramat'
    xyT   = kpts.T[0:2]
    invET = kpts.T[2:5]
    # Expand into full matrix
    e11 = invET[0]
    e12 = invET[1]
    e21 = invET[1]
    e22 = invET[2]
    invE_list = np.array(((e11, e12), (e21, e22))).T
    # Decompose using singular value decomposition
    invXWYt_list = [svd(invE) for invE in invE_list]
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


