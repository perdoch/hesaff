#!/usr/bin/env python2.7
from __future__ import absolute_import, division, print_function
import utool as ut


def run_tests():
    # Build module list and run tests
    import sys
    ut.change_term_title('RUN HESAFF TESTS')
    exclude_doctests_fnames = set([
    ])
    exclude_dirs = [
        '_broken',
        'old',
        #'tests',
        'timeits',
        '_scripts',
        '_timeits',
        '_doc',
        'notebook',
    ]
    dpath_list = ['pyhesaff']
    doctest_modname_list = ut.find_doctestable_modnames(
        dpath_list, exclude_doctests_fnames, exclude_dirs)

    #print(doctest_modname_list)
    #return

    for modname in doctest_modname_list:
        exec('import ' + modname, globals(), locals())
    module_list = [sys.modules[name] for name in doctest_modname_list]
    ut.doctest_module_list(module_list)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    run_tests()
