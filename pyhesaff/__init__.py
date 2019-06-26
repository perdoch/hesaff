# flake8: noqa
from __future__ import absolute_import, print_function, division
__version__ = '2.0.1'


from pyhesaff import ctypes_interface
from pyhesaff import _pyhesaff
from pyhesaff._pyhesaff import (DESC_DIM, HESAFF_CLIB,
                                HESAFF_PARAM_DICT, HESAFF_PARAM_TYPES,
                                HESAFF_TYPED_PARAMS, KPTS_DIM,
                                adapt_scale, alloc_kpts,
                                alloc_patches, alloc_vecs,
                                argparse_hesaff_params,
                                detect_feats, detect_feats2,
                                detect_feats_in_image, detect_feats_list,
                                detect_num_feats_in_image,
                                extract_desc_from_patches, extract_patches,
                                extract_vecs, get_cpp_version,
                                get_hesaff_default_params,
                                get_is_debug_mode,
                                hesaff_kwargs_docstr_block, img32_dtype,
                                test_rot_invar,
                                vtool_adapt_rotation,)

#from pyhesaff._pyhesaff import (extract_vecs, detect_feats, detect_feats_list,
#                                adapt_scale, vtool_adapt_rotation, kpts_dtype, vecs_dtype,
#                                get_hesaff_default_params, extract_desc_from_patches,
#                                KPTS_DIM, DESC_DIM, __LIB_FPATH__, HESAFF_CLIB)
#from pyhesaff._pyhesaff import *  # NOQA
"""
# ut.util_inspect.get_module_owned_functions(pyhesaff._pyhesaff)
import pyhesaff
modname = pyhesaff.__name__
IMPORT_TUPLES = [('_pyhesaff', None)]
ut.exec_funckw(ut._internal.util_importer.dynamic_import, globals())
x = ut._internal.util_importer.dynamic_import(modname, IMPORT_TUPLES, dump=True, check_not_imported=False, verbose=True)
"""
