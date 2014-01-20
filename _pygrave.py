    #print('[pyhesaff] extracting %d desc' % nKpts)
    #print('[pyhesaff] kpts.shape =%r' % (kpts.shape,))
    #print('[pyhesaff] kpts.dtype =%r' % (kpts.dtype,))
    #print('[pyhesaff] desc.shape =%r' % (desc.shape,))
    #print('[pyhesaff] desc.dtype =%r' % (desc.dtype,))
    #desc = np.require(desc, dtype=desc_dtype, requirements=['C_CONTIGUOUS', 'WRITABLE', 'ALIGNED'])


    #desc2 = np.empty((nKpts, 128), desc_dtype)
    #desc3 = np.empty((nKpts, 128), desc_dtype)
    #kpts = np.require(kpts, kpts_dtype, ['ALIGNED'])
    #desc = np.require(desc, desc_dtype, ['ALIGNED'])

    #print('[pyhessaff] Override params: %r' % kwargs)
    #print(hesaff_params)
    #hesaff_ptr = hesaff_lib.new_hesaff(realpath(img_fpath))
