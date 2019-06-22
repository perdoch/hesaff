

def sift_test():
    """
    Play with SIFT equations using python so I can see and compare results.
    """
    import numpy as np
    # Sample measurement from lena
    sift_raw = np.array([
        48.0168, 130.017, 159.065, 54.5727, 63.7103, 14.3629, 27.0228, 15.3527,
        40.5067, 165.721, 511.036, 196.888, 4.72748, 8.85093, 15.9457, 14.4198,
        49.7571, 209.104, 452.047, 223.972, 2.66391, 16.8975, 21.7488, 13.6855,
        0.700244, 10.2518, 312.483, 282.647, 1.82898, 3.01759, 0.448028, 0,
        144.834, 300.438, 131.837, 40.3284, 11.1998, 9.68647, 7.68484, 29.166,
        425.953, 386.903, 352.388, 267.883, 12.9652, 18.833, 8.55462, 71.7924,
        112.282, 295.512, 678.599, 419.405, 21.3151, 91.9408, 22.8681, 9.83749,
        3.06347, 97.6562, 458.799, 221.873, 68.1473, 410.764, 48.9493, 2.01682,
        194.794, 43.7171, 16.2078, 17.5604, 48.8504, 48.3823, 45.7636, 299.432,
        901.565, 188.732, 32.6512, 23.6874, 55.379, 272.264, 68.2334, 221.37,
        159.631, 44.1475, 126.636, 95.1978, 74.1097, 1353.24, 239.319, 33.5368,
        5.62254, 69.0013, 51.7629, 9.55458, 26.4599, 699.623, 208.78, 2.09156,
        135.278, 19.5378, 52.0265, 51.8445, 49.1938, 9.04161, 11.6605, 87.4498,
        604.012, 85.6801, 42.9738, 75.8549, 183.65, 206.912, 34.2781, 95.0146,
        13.4201, 83.7426, 440.322, 83.0038, 125.663, 457.333, 52.6424, 4.93713,
        0.38947, 244.762, 291.113, 7.50165, 8.16208, 73.2169, 21.9674,
        0.00429259, ])

    import vtool as vt

    # CONFIRMED: One normalization followed by another does not do anything
    #sift_root1 = vt.normalize(sift_root1, ord=2)
    #sift_root1 = vt.normalize(sift_root1, ord=1)

    sift_clip = sift_raw.copy()
    sift_clip = vt.normalize(sift_clip, ord=2)
    sift_clip[sift_clip > .2] = .2
    sift_clip = vt.normalize(sift_clip, ord=2)

    siff_ell2 = vt.normalize(sift_raw, ord=2)

    siff_ell1 = vt.normalize(sift_raw, ord=1)

    # Two versions of root SIFT
    # They are equlivalent
    # taken from https://hal.inria.fr/hal-00840721/PDF/RR-8325.pdf
    normalize1 = lambda x: vt.normalize(x, ord=1)  # NOQA
    normalize2 = lambda x: vt.normalize(x, ord=2)  # NOQA

    assert np.all(np.isclose(np.sqrt(normalize1(sift_raw)),
                             normalize2(np.sqrt(sift_raw))))

    # How do we genralize this for alpha != .5?
    # Just always L2 normalize afterwords?
    alpha = .2
    powerlaw = lambda x: np.power(x, alpha)  # NOQA
    sift_root1 = normalize2(powerlaw(normalize1(sift_raw)))
    sift_root2 = normalize2(powerlaw(sift_raw))
    flags = np.isclose(sift_root1, sift_root2)
    print(flags)
    assert np.all(flags)

    #sift_root_quant = np.clip((sift_root1 * 512), 0, 255).astype(np.uint8)
    #p = (np.bincount(sift_root_quant) / 128)
    #entropy = -np.nansum(p * np.log2(p))

    s = sift_raw[0:10]
    np.sqrt(s) / (np.sqrt(s).sum() ** 2)
    np.power(normalize1(s), 2)
    #b = powerlaw(normalize1(s))
    #print(np.isclose(a, b))

    np.isclose(normalize1(s), normalize1(normalize2(s)))

    # Another root SIFT version from
    # https://hal.inria.fr/hal-00688169/document
    # but this doesnt seem to work with uint8 representations
    sift_root3 = np.sqrt(sift_raw)
    sift_root3 = sift_root3 / np.sqrt(np.linalg.norm(sift_root3))

    import plottool as pt
    import utool as ut
    ut.qtensure()

    fig = pt.figure(fnum=1, pnum=None)
    def draw_sift(sift, pnum, title, **kwargs):
        ax = fig.add_subplot(*pnum)
        pt.draw_sifts(ax, sift[None, :], **kwargs)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.grid(False)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        if title:
            ax.set_title(title)

    fig.clf()
    pnum_ = pt.make_pnum_nextgen(2, 4)
    draw_sift(sift_raw, pnum_(), 'raw/max(raw)', fidelity=sift_raw.max())
    draw_sift(sift_clip, pnum_(), 'clip', fidelity=1.0)
    draw_sift(siff_ell2, pnum_(), 'l2', fidelity=1.0)
    draw_sift(siff_ell1, pnum_(), 'l1', fidelity=1.0)
    draw_sift(sift_root1, pnum_(), 'root1', fidelity=1.0)
    draw_sift(sift_root2, pnum_(), 'root2', fidelity=1.0)
    draw_sift(sift_root3, pnum_(), 'root3', fidelity=2.0)
