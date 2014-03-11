from __future__ import print_function, division
import sys
from os.path import split, exists, join, dirname
import os


DEBUG = '--debug' in sys.argv


def locate_path(dname, recurse_down=True):
    'Search for a path'
    tried_fpaths = []
    root_dir = os.getcwd()
    while root_dir is not None:
        dpath = join(root_dir, dname)
        if exists(dpath):
            return dpath
        else:
            tried_fpaths.append(dpath)
        _new_root = dirname(root_dir)
        if _new_root == root_dir:
            root_dir = None
            break
        else:
            root_dir = _new_root
        if not recurse_down:
            break
    msg = ('\n[sysreq!] Checked: '.join(tried_fpaths))
    print(msg)
    raise ImportError(msg)


def ensure_path(dname):
    dname_list = [split(dpath)[1] for dpath in sys.path]
    if not dname in dname_list:
        dpath = locate_path(dname)
        print('[sysreq] appending %r to PYTHONPATH' % dpath)
        sys.path.append(dpath)
    elif DEBUG:
        print('[sysreq] PYTHONPATH has %r' % dname)


ensure_path('hesaff')
ensure_path('hotspotter')
