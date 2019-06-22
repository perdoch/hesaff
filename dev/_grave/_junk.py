def arrptr_to_np_OLD(c_arrptr, shape, arr_t, dtype):
    """
    Casts an array pointer from C to numpy

    Args:
        c_arrptr (uint64): a pointer to an array returned from C
        shape (tuple): shape of the underlying array being pointed to
        arr_t (PyCSimpleType): the ctypes datatype of c_arrptr
        dtype (dtype): numpy datatype the array will be to cast into

    CommandLine:
        python2 -m pyhesaff._pyhesaff --test-detect_feats_list:0 --rebuild-hesaff
        python2 -m pyhesaff._pyhesaff --test-detect_feats_list:0
        python3 -m pyhesaff._pyhesaff --test-detect_feats_list:0

    """
    try:
        byte_t = ctypes.c_char
        itemsize_ = dtype().itemsize
        #import utool
        #utool.printvar2('itemsize_')
        ###---------
        #dtype_t1 = C.c_voidp * itemsize_
        #dtype_ptr_t1 = C.POINTER(dtype_t1)  # size of each item
        #dtype_ptr_t = dtype_ptr_t1
        ###---------
        if True or six.PY2:
            # datatype of array elements
            dtype_t = byte_t * itemsize_
            dtype_ptr_t = C.POINTER(dtype_t)  # size of each item
            #typed_c_arrptr = c_arrptr.astype(C.c_long)
            typed_c_arrptr = c_arrptr.astype(int)
            c_arr = C.cast(typed_c_arrptr, dtype_ptr_t)   # cast to ctypes
            #raise Exception('fuuu. Why does 2.7 work? Why does 3.4 not!?!!!')
        else:
            dtype_t = C.c_char * itemsize_
            dtype_ptr_t = C.POINTER(dtype_t)  # size of each item
            #typed_c_arrptr = c_arrptr.astype(int)
            #typed_c_arrptr = c_arrptr.astype(C.c_size_t)
            typed_c_arrptr = c_arrptr.astype(int)
            c_arr = C.cast(c_arrptr.astype(C.c_size_t), dtype_ptr_t)   # cast to ctypes
            c_arr = C.cast(c_arrptr.astype(int), dtype_ptr_t)   # cast to ctypes
            c_arr = C.cast(c_arrptr, dtype_ptr_t)   # cast to ctypes
            #typed_c_arrptr = c_arrptr.astype(int)
            #, order='C', casting='safe')
            #utool.embed()
            #typed_c_arrptr = c_arrptr.astype(dtype_t)
            #typed_c_arrptr = c_arrptr.astype(ptr_t2)
            #typed_c_arrptr = c_arrptr.astype(C.c_uint8)
            #typed_c_arrptr = c_arrptr.astype(C.c_void_p)
            #typed_c_arrptr = c_arrptr.astype(C.c_int)
            #typed_c_arrptr = c_arrptr.astype(C.c_char)  # WORKS BUT WRONG
            #typed_c_arrptr = c_arrptr.astype(bytes)  # WORKS BUT WRONG
            #typed_c_arrptr = c_arrptr.astype(int)
            #typed_c_arrptr = c_arrptr
            #typed_c_arrptr = c_arrptr.astype(np.int64)
            #typed_c_arrptr = c_arrptr.astype(int)

            """
            ctypes.cast(arg1, arg2)

            Input:
                arg1 - a ctypes object that is or can be converted to a pointer
                       of some kind
                arg2 - a ctypes pointer type.
            Output:
                 It returns an instance of the second argument, which references
                 the same memory block as the first argument
            """
            c_arr = C.cast(typed_c_arrptr, dtype_ptr_t)   # cast to ctypes
        np_arr = np.ctypeslib.as_array(c_arr, shape)       # cast to numpy
        np_arr.dtype = dtype                               # fix numpy dtype
    except Exception as ex:
        import utool as ut
        #utool.embed()
        varnames = sorted(list(locals().keys()))
        vartypes = [(type, name) for name in varnames]
        spaces    = [None for name in varnames]
        c_arrptr_dtype = c_arrptr.dtype  # NOQA
        #key_list = list(zip(varnames, vartypes, spaces))
        key_list = ['c_arrptr_dtype'] + 'c_arrptr, shape, arr_t, dtype'.split(', ')
        print('itemsize(float) = %r' % np.dtype(float).itemsize)
        print('itemsize(c_char) = %r' % np.dtype(C.c_char).itemsize)
        print('itemsize(c_wchar) = %r' % np.dtype(C.c_wchar).itemsize)
        print('itemsize(c_char_p) = %r' % np.dtype(C.c_char_p).itemsize)
        print('itemsize(c_wchar_p) = %r' % np.dtype(C.c_wchar_p).itemsize)
        print('itemsize(c_int) = %r' % np.dtype(C.c_int).itemsize)
        print('itemsize(c_int32) = %r' % np.dtype(C.c_int32).itemsize)
        print('itemsize(c_int64) = %r' % np.dtype(C.c_int64).itemsize)
        print('itemsize(int) = %r' % np.dtype(int).itemsize)
        print('itemsize(float32) = %r' % np.dtype(np.float32).itemsize)
        print('itemsize(float64) = %r' % np.dtype(np.float64).itemsize)
        ut.printex(ex, keys=key_list)
        ut.embed()
        raise
    return np_arr


def extract_2darr_list(size_list, ptr_list, arr_t, arr_dtype,
                        arr_dim):
    """
    size_list - contains the size of each output 2d array
    ptr_list  - an array of pointers to the head of each output 2d
                array (which was allocated in C)
    arr_t     - the C pointer type
    arr_dtype - the numpy array type
    arr_dim   - the number of columns in each output 2d array
    """
    iter_ = ((arr_ptr, (size, arr_dim))
             for (arr_ptr, size) in zip(ptr_list, size_list))
    arr_list = [arrptr_to_np(arr_ptr, shape, arr_t, arr_dtype)
                for arr_ptr, shape in iter_]
    return arr_list


def arrptr_to_np(c_arrptr, shape, arr_t, dtype):
    """
    Casts an array pointer from C to numpy

    Args:
        c_arrptr (uint64): a pointer to an array returned from C
        shape (tuple): shape of the underlying array being pointed to
        arr_t (PyCSimpleType): the ctypes datatype of c_arrptr
        dtype (dtype): numpy datatype the array will be to cast into

    CommandLine:
        python2 -m pyhesaff._pyhesaff --test-detect_feats_list:0 --rebuild-hesaff
        python2 -m pyhesaff._pyhesaff --test-detect_feats_list:0
        python3 -m pyhesaff._pyhesaff --test-detect_feats_list:0

    """
    try:
        byte_t = ctypes.c_char
        itemsize_ = dtype().itemsize  # size of a single byte
        dtype_t = byte_t * itemsize_  # datatype of array elements
        dtype_ptr_t = C.POINTER(dtype_t)  # size of each item
        typed_c_arrptr = c_arrptr.astype(int)
        c_arr = C.cast(typed_c_arrptr, dtype_ptr_t)   # cast to ctypes
        #raise Exception('fuuu. Why does 2.7 work? Why does 3.4 not!?!!!')
        np_arr = np.ctypeslib.as_array(c_arr, shape)       # cast to numpy
        np_arr.dtype = dtype                               # fix numpy dtype
    except Exception as ex:
        import utool as ut
        #utool.embed()
        varnames = sorted(list(locals().keys()))
        vartypes = [(type, name) for name in varnames]
        spaces    = [None for name in varnames]
        c_arrptr_dtype = c_arrptr.dtype  # NOQA
        #key_list = list(zip(varnames, vartypes, spaces))
        key_list = ['c_arrptr_dtype'] + 'c_arrptr, shape, arr_t, dtype'.split(', ')
        print('itemsize(float) = %r' % np.dtype(float).itemsize)
        print('itemsize(c_char) = %r' % np.dtype(C.c_char).itemsize)
        print('itemsize(c_wchar) = %r' % np.dtype(C.c_wchar).itemsize)
        print('itemsize(c_char_p) = %r' % np.dtype(C.c_char_p).itemsize)
        print('itemsize(c_wchar_p) = %r' % np.dtype(C.c_wchar_p).itemsize)
        print('itemsize(c_int) = %r' % np.dtype(C.c_int).itemsize)
        print('itemsize(c_int32) = %r' % np.dtype(C.c_int32).itemsize)
        print('itemsize(c_int64) = %r' % np.dtype(C.c_int64).itemsize)
        print('itemsize(int) = %r' % np.dtype(int).itemsize)
        print('itemsize(float32) = %r' % np.dtype(np.float32).itemsize)
        print('itemsize(float64) = %r' % np.dtype(np.float64).itemsize)
        ut.printex(ex, keys=key_list)
        raise
    return np_arr


