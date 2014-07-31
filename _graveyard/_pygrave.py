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


def test_hesaff(n=None, fnum=1, **kwargs):
    from hotspotter import interaction
    reextract = kwargs.get('reextrac', False)
    new_exe = kwargs.get('new_exe', False)
    old_exe = kwargs.get('old_exe', False)
    adaptive = kwargs.get('adaptive', False)
    use_exe = new_exe or old_exe

    if use_exe:
        import _pyhesaffexe
        if new_exe:
            _pyhesaffexe.EXE_FPATH = _pyhesaffexe.find_hesaff_fpath(exe_name='hesaffexe')
        if old_exe:
            _pyhesaffexe.EXE_FPATH = _pyhesaffexe.find_hesaff_fpath(exe_name='hesaff')

    print('[test]---------------------')
    try:
        # Select kpts
        title = split(_pyhesaffexe.EXE_FPATH)[1] if use_exe else 'libhesaff'
        detect_func = _pyhesaffexe.detect_kpts if use_exe else detect_kpts
        with helpers.Timer(msg=title):
            kpts, desc = detect_func(img_fpath, scale_min=0, scale_max=1000)
        if reextract:
            title = 'reextract'
            with helpers.Timer(msg='reextract'):
                desc = extract_desc(img_fpath, kpts)
        kpts_ = kpts if n is None else spaced_elements(kpts, n)
        desc_ = desc if n is None else spaced_elements(desc, n)
        if adaptive:
            kpts_, desc_ = adaptive_scale(img_fpath, kpts_, desc_)
        # Print info
        print('detected %d keypoints' % len(kpts))
        print('drawing %d/%d kpts' % (len(kpts_), len(kpts)))
        title += ' ' + str(len(kpts))
        #print(kpts_)
        #print(desc_[:, 0:16])
        # Draw kpts
        interaction.interact_keypoints(image, kpts_, desc_, fnum, nodraw=True)
        df2.set_figtitle(title)
        #df2.imshow(image, fnum=fnum)
        #df2.draw_kpts2(kpts_, ell_alpha=.9, ell_linewidth=4,
                        #ell_color='distinct', arrow=True, rect=True)
    except Exception as ex:
        import traceback
        traceback.format_exc()
        print('EXCEPTION! ' + repr(ex))
        raise
    print('[test]---------------------')
    return locals()
