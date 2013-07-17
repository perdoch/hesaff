import subprocess
import numpy as np
import os
from os.path import dirname, realpath, join
from PIL import Image
#import threading
#__hesaff_lock = threading.Lock()

hesaff_exe = join(realpath(dirname(__file__)), 'hesaff')
__HOME__ = os.path.expanduser('~')
hesaff_tmp_dir = os.path.join(__HOME__, '.tmp_hesaff') 
if not os.path.exists(hesaff_tmp_dir):
    print('Making directory: '+hesaff_tmp_dir)
    os.mkdir(hesaff_tmp_dir)

def compute_hesaff(rchip):
    #__hesaff_lock.acquire_lock()
    #__hesaff_lock.release_lock()
    tmp_fpath = hesaff_tmp_dir + '/tmp.ppm'
    rchip_pil = Image.fromarray(rchip)
    rchip_pil.save(tmp_fpath, 'PPM')
    (kpts, desc) = __compute_hesaff(tmp_fpath)
    return (kpts, desc)
    
def precompute_hesaff(rchip_fpath, chiprep_fpath):
    kpts, desc = __compute_hesaff(rchip_fpath)
    np.savez(chiprep_fpath, kpts, desc)
    return kpts, desc

def __compute_hesaff(rchip_fpath):
    'Runs external keypoint detetectors like hesaff'
    outname = rchip_fpath + '.hesaff.sift'
    args = '"' + rchip_fpath + '"'
    cmd  = hesaff_exe + ' ' + args
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (out, err) = proc.communicate()
    if proc.returncode != 0:
        raise Exception('  * Failed to execute '+cmd+'\n  * OUTPUT: '+out)
    if not os.path.exists(outname):
        raise Exception('  * The output file doesnt exist: '+outname)
    kpts, desc = read_text_chiprep_file(outname)
    return kpts, desc

def read_text_chiprep_file(outname):
    'Reads output from external keypoint detectors like hesaff'
    with open(outname, 'r') as file:
        # Read header
        ndims = int(file.readline())
        nkpts = int(file.readline())
        # Preallocate output
        kpts = np.zeros((nkpts, 5), dtype=np.float32)
        desc = np.zeros((nkpts, ndims), dtype=np.uint8)
        # iterate over lines
        lines = file.readlines()
        for kx, line in enumerate(lines):
            data = line.split(' ')
            kpts[kx,:] = np.array([np.float32(_)\
                                   for _ in data[0:5]], dtype=np.float32)
            desc[kx,:] = np.array([np.uint8(_)\
                                   for _ in data[5: ]], dtype=np.uint8)
        return (kpts, desc)
