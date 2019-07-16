# -*- coding: utf-8 -*-
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


def get_candidate_plat_specifiers():
    import distutils
    if sys.maxsize > 2 ** 32:
        arch = 'x86_64'  # TODO: get correct arch spec
    else:
        arch = 'i686'  # TODO: get correct arch spec

    py_ver = sys.version[0:3]

    try:
        plat_name = distutils.util.get_platform()
    except AttributeError:
        plat_name = distutils.sys.platform

    plat_name_cands = [plat_name]
    if sys.platform.startswith('linux'):
        plat_name_cands.append('linux')
        plat_name_cands.append('manylinux1')
        plat_name_cands.append('manylinux')
    elif sys.platform.startswith('darwin'):
        # HACK:
        # on travis, wheel builds as libhesaff.macosx-10.12-x86_64-2.7.dylib,
        # but we seem to want libhesaff.macosx-10.6-intel-2.7.dylib
        # TODO: what is the proper way to determine the ABI tag?
        plat_name_cands.append('macosx-10.6')
        plat_name_cands.append('macosx-10.7')
        plat_name_cands.append('macosx-10.9')
        plat_name_cands.append('macosx-10.12')
        plat_name_cands.append('macosx-10.6-intel')
        plat_name_cands.append('macosx-10.7-intel')
        plat_name_cands.append('macosx-10.9-intel')
        plat_name_cands.append('macosx-10.12-intel')

    spec_list = []
    for plat_name in plat_name_cands:
        spec_list.extend([
            '.{}-{}'.format(plat_name, sys.version[0:3]),
            '.{}-{}-{}'.format(plat_name, arch, py_ver),
        ])
    spec_list.append('')
    return spec_list


def get_lib_fname_list(libname):
    """
    Args:
        libname (str): library name (e.g. 'hesaff', not 'libhesaff')

    Returns:
        list: libnames - list of plausible library file names

    CommandLine:
        python -m pyhesaff.ctypes_interface get_lib_fname_list

    Example:
        >>> from pyhesaff.ctypes_interface import *  # NOQA
        >>> libname = 'hesaff'
        >>> libnames = get_lib_fname_list(libname)
        >>> import ubelt as ub
        >>> print('libnames = {}'.format(ub.repr2(libnames)))
    """
    spec_list = get_candidate_plat_specifiers()

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
    get_lib_dpath_list = [
        root_dir,
        # join(root_dir, 'lib'),
        # join(root_dir, 'build'),
        # join(root_dir, 'build', 'lib'),
    ]
    return get_lib_dpath_list


def find_lib_fpath(libname, root_dir, recurse_down=False, verbose=False):
    """ Search for the library """
    lib_fname_list = get_lib_fname_list(libname)
    tried_fpaths = []

    class FoundLib(Exception):
        pass

    FINAL_LIB_FPATH = None

    try:
        for lib_fname in lib_fname_list:
            if verbose:
                print('--')
            curr_dpath = root_dir
            max_depth = 0
            while curr_dpath is not None:

                for lib_dpath in get_lib_dpath_list(curr_dpath):
                    lib_fpath = normpath(join(lib_dpath, lib_fname))
                    tried_fpaths.append(lib_fpath)
                    flag = exists(lib_fpath)
                    if verbose:
                        print('[c] Check: {}, exists={}'.format(lib_fpath, int(flag)))
                    if flag:
                        if verbose:
                            print('using: {}'.format(lib_fpath))
                        FINAL_LIB_FPATH = lib_fpath
                        raise FoundLib

                max_depth -= 1
                if max_depth < 0:
                    curr_dpath = None
                    break

                _new_dpath = dirname(curr_dpath)
                if _new_dpath == curr_dpath:
                    curr_dpath = None
                    break
                else:
                    curr_dpath = _new_dpath
                if not recurse_down:
                    break
    except FoundLib:
        pass
        return FINAL_LIB_FPATH

    msg = ('\n[C!] find_lib_fpath(libname=%r root_dir=%r, recurse_down=%r, verbose=%r)' %
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
    import xdoctest
    xdoctest.doctest_module(__file__)
