from __future__ import absolute_import, division, print_function, unicode_literals
from os.path import join, exists, dirname, normpath
import sys
import os
import ctypes as C


#============================
# general ctypes interface
#============================

__DEBUG_CLIB__ = '--debug' in sys.argv or '--debug-clib' in sys.argv


def get_plat_specifier():
    """
    Standard platform specifier used by distutils
    """
    import distutils
    try:
        plat_name = distutils.util.get_platform()
    except AttributeError:
        plat_name = distutils.sys.platform
    plat_specifier = ".%s-%s" % (plat_name, sys.version[0:3])
    if hasattr(sys, 'gettotalrefcount'):
        plat_specifier += '-pydebug'
    return plat_specifier


def get_lib_fname_list(libname):
    """
    Args:
        libname (str): library name (e.g. 'hesaff', not 'libhesaff')

    Returns:
        list: libnames - list of plausible library file names

    CommandLine:
        python -m pyhesaff.ctypes_interface get_lib_fname_list

    Example:
        >>> # DISABLE_DOCTEST
        >>> from pyhesaff.ctypes_interface import *  # NOQA
        >>> libname = 'hesaff'
        >>> libnames = get_lib_fname_list(libname)
        >>> result = ('libnames = %s' % (ut.repr2(libnames),))
        >>> print(result)
    """
    spec_list = [get_plat_specifier(), '']
    prefix_list = ['lib' + libname]
    if sys.platform.startswith('win32'):
        prefix_list.append(libname)
        ext = '.dll'
    elif sys.platform.startswith('darwin'):
        ext = '.dylib'
    elif sys.platform.startswith('linux'):
        ext = '.so'
    else:
        raise Exception('Unknown operating system: %s' % sys.platform)
    # Construct priority ordering of libnames
    libnames = [''.join((prefix, spec, ext))
                for spec in spec_list
                for prefix in prefix_list]
    return libnames


def get_lib_dpath_list(root_dir):
    """
    input <root_dir>: deepest directory to look for a library (dll, so, dylib)
    returns <libnames>: list of plausible directories to look.
    """
    'returns possible lib locations'
    get_lib_dpath_list = [root_dir,
                          join(root_dir, 'lib'),
                          join(root_dir, 'build'),
                          join(root_dir, 'build', 'lib')]
    return get_lib_dpath_list


def find_lib_fpath(libname, root_dir, recurse_down=True, verbose=False):
    """ Search for the library """
    lib_fname_list = get_lib_fname_list(libname)
    tried_fpaths = []
    while root_dir is not None:
        for lib_fname in lib_fname_list:
            for lib_dpath in get_lib_dpath_list(root_dir):
                lib_fpath = normpath(join(lib_dpath, lib_fname))
                if exists(lib_fpath):
                    if verbose:
                        print('\n[c] Checked: '.join(tried_fpaths))
                    if __DEBUG_CLIB__:
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
    """
    Searches for a library matching libname and loads it

    Args:
        libname:  library name (e.g. 'hesaff', not 'libhesaff')

        root_dir: the deepest directory searched for the
                  library file (dll, dylib, or so).
    Returns:
        clib: a ctypes object used to interface with the library
    """
    lib_fpath = find_lib_fpath(libname, root_dir)
    try:
        clib = C.cdll[lib_fpath]

        def def_cfunc(return_type, func_name, arg_type_list):
            'Function to define the types that python needs to talk to c'
            cfunc = getattr(clib, func_name)
            cfunc.restype = return_type
            cfunc.argtypes = arg_type_list
        clib.__LIB_FPATH__ = lib_fpath
        return clib, def_cfunc, lib_fpath
    except OSError as ex:
        print('[C!] Caught OSError:\n%s' % ex)
        errsuffix = 'Is there a missing dependency?'
    except Exception as ex:
        print('[C!] Caught Exception:\n%s' % ex)
        errsuffix = 'Was the library correctly compiled?'
    print('[C!] cwd=%r' % os.getcwd())
    print('[C!] load_clib(libname=%r root_dir=%r)' % (libname, root_dir))
    print('[C!] lib_fpath = %r' % lib_fpath)
    errmsg = '[C] Cannot LOAD %r dynamic library. ' % (libname,) + errsuffix
    print(errmsg)
    raise ImportError(errmsg)


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m pyhesaff.ctypes_interface
        python -m pyhesaff.ctypes_interface --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
