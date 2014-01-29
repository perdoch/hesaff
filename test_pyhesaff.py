from __future__ import print_function, division
# Standard
import sys
from os.path import join, exists, realpath, expanduser, split
import multiprocessing
from itertools import izip
# Scientific
import numpy as np
import cv2
# Hotspotter
from hotspotter import draw_func2 as df2
from hotspotter import helpers  # NOQA
from hotspotter import vizualizations as viz  # NOQA
from hotspotter import interaction
from hotspotter import fileio as io
from hotspotter import __common__
# TPL
import pyhesaff
import ellipse
print, print_, print_on, print_off, rrr, profile, printDBG =\
    __common__.init(__name__, module_prefix='[testhesaff]', DEBUG=False, initmpl=False)


def ensure_hotspotter():
    import matplotlib
    matplotlib.use('Qt4Agg', warn=True, force=True)
    # Look for hotspotter in ~/code
    hotspotter_dir = join(expanduser('~'), 'code', 'hotspotter')
    if not exists(hotspotter_dir):
        print('[jon] hotspotter_dir=%r DOES NOT EXIST!' % (hotspotter_dir,))
    # Append hotspotter to PYTHON_PATH (i.e. sys.path)
    if not hotspotter_dir in sys.path:
        sys.path.append(hotspotter_dir)


@profile
def test_hesaff_kpts(img_fpath, **kwargs):
    if not 'kwargs' in vars():
        kwargs = {}
    # Make detector and read image
    hesaff_ptr = pyhesaff.new_hesaff(img_fpath, **kwargs)
    # Return the number of keypoints detected
    nKpts = pyhesaff.hesaff_lib.detect(hesaff_ptr)
    #print('[pyhesaff] detected: %r keypoints' % nKpts)
    # Allocate arrays
    kpts = np.empty((nKpts, 5), pyhesaff.kpts_dtype)
    desc = np.empty((nKpts, 128), pyhesaff.desc_dtype)
    # Populate arrays
    pyhesaff.hesaff_lib.exportArrays(hesaff_ptr, nKpts, kpts, desc)
    # TODO: Incorporate parameters
    # TODO: Scale Factor
    #hesaff_lib.extractDesc(hesaff_ptr, nKpts, kpts, desc2)
    #hesaff_lib.extractDesc(hesaff_ptr, nKpts, kpts, desc3)
    #print('[hesafflib] returned')
    return kpts, desc


def spaced_elements2(list_, n):
    if n is None:
        return np.arange(len(list_))
    if n == 0:
        return np.empty(0)
    indexes = np.arange(len(list_))
    stride = len(indexes) // n
    return indexes[0:-1:stride]


def spaced_elements(list_, n):
    if n is None:
        return 'list'
    indexes = np.arange(len(list_))
    stride = len(indexes) // n
    return list_[indexes[0:-1:stride]]


def load_test_data(short=False, n=0):
    if not 'short' in vars():
        short = False
    # Read Image
    #ellipse.rrr()
    nScales = 4
    nSamples = 16
    img_fname = 'zebra.png'
    img_fpath = realpath(img_fname)
    imgBGR = io.imread(img_fpath)
    imgLAB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2LAB)
    imgL = imgLAB[:, :, 0]
    kpts, desc = pyhesaff.detect_kpts(img_fpath, scale_min=20, scale_max=100)
    if short:
        extra_fxs = []
        if img_fname == 'zebra.png':
            extra_fxs = [374, 520, 880][0:1]
        fxs = np.array(spaced_elements2(kpts, n).tolist() + extra_fxs)
        kpts = kpts[fxs]
        desc = desc[fxs]
    test_data = locals()
    return test_data


def test_adaptive_scale():
    # Get relevant test data
    '''
    __name__ = 'IPython'
    exec(open('test_pyhesaff.py').read())
    exec(open('ellipse.py').read())
    '''
    test_data = load_test_data(short=True)
    img_fpath = test_data['img_fpath']
    imgL = test_data['imgL']
    imgBGR = test_data['imgBGR']
    kpts = test_data['kpts']
    desc = test_data['desc']
    # WHY ARENT THESE 1s?
    #np.sqrt(((desc / 256.0) ** 2).sum(1))

    nScales = 16
    nSamples = 16
    low, high = -1, 2
    nKp = len(kpts)

    df2.figure(fnum=1, doclf=True, docla=True)

    def show_kpts(kpts_, px, title):
        viz.show_keypoints(imgBGR, kpts_, pnum=(2, 3, px + 3), fnum=1,
                           color=df2.BLUE, title=title)

    def plot_line(vals, title):
        df2.figure(fnum=1, pnum=(2, 1, 1))
        df2.plot(vals, 'bo-')

    def plot_marks(vals, marker, title):
        df2.figure(fnum=1, pnum=(2, 1, 1))
        df2.plot(vals[0], vals[1], marker)

    fx = 0

    # INPUT
    kpts = np.array(kpts, dtype=np.float64)
    show_kpts(kpts, 1, 'original keypoints')

    # STEP1: EXPAND KEYPOINTS
    expanded_kpts = ellipse.expand_scales(kpts, nScales, low, high)
    show_kpts(expanded_kpts, 2, 'expanded keypoint')

    # STEP2: UNIFORM SAMPLE / INTERPOLATE MAXIMA
    border_vals_sum = ellipse.sample_ell_border_vals(imgBGR, expanded_kpts, nKp, nScales, nSamples)
    x_data_list, y_data_list = ellipse.find_maxima_with_neighbors(border_vals_sum)
    peak_list = ellipse.interpolate_peaks(x_data_list, y_data_list)

    plot_line(border_vals_sum[fx], 'gradient mag')
    plot_marks([x_data_list[fx][0], y_data_list[fx][0]], 'go', 'left')
    plot_marks([x_data_list[fx][1], y_data_list[fx][1]], 'ro', 'extreme')
    plot_marks([x_data_list[fx][2], y_data_list[fx][2]], 'go', 'right')
    plot_marks(peak_list[fx].T, 'rx', 'interpolated peaks')

    parabolas_list = []
    for x_data, y_data in izip(x_data_list, y_data_list):
        parabola_points = []
        for (x1, x2, x3), (y1, y2, y3) in izip(x_data.T, y_data.T):
            coeff = np.polyfit((x1, x2, x3), (y1, y2, y3), 2)
            xpoints = np.linspace(x1, x3, 50)
            ypoints = np.polyval(coeff, xpoints)
            parabola_points.append((xpoints, ypoints))
        parabolas_list.append(parabola_points)

    df2.set_ylabel('gradient magnitude')
    df2.set_xlabel('scale index')

    for xdat, ydat in parabolas_list[0]:
        plot_marks([xdat, ydat], 'g--', '')

    # STEP 3: INTERPOLATE SCALES
    subscale_list = ellipse.interpolate_between(peak_list, nScales, high, low)
    subscale_kpts = ellipse.expand_subscales(kpts, subscale_list)
    #show_kpts(subscale_kpts, 3, 'subscale keypoint')

    # STEP 4: Check Image Bounds
    # Make sure that the new shapes are in bounds
    height, width = imgBGR.shape[0:2]
    isvalid = ellipse.check_kpts_in_bounds(subscale_kpts, width, height)
    adapted_kpts = np.array(subscale_kpts[isvalid], dtype=np.float32)
    show_kpts(adapted_kpts, 3, 'adapted keypoint')

    df2.update()

    scales = 2 ** np.linspace(low, high, nScales)
    adapted_kpts = ellipse.adaptive_scale(img_fpath, kpts, nScales, low, high, nSamples)

    #plot_vals(adapted_kpts, pnum=(3

    #viz.show_keypoints(imgBGR, adapted_kpts, pnum=(3, 1, 3), fnum=1, color=df2.BLUE, title='adapted keypoints')

    #adapted_desc = pyhesaff.extract_desc(img_fpath, adapted_kpts)
    #desc = pyhesaff.extract_desc(img_fpath, kpts)
    ##interaction.interact_keypoints(imgBGR, adapted_kpts, adapted_desc)
    #df2.update()
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    ensure_hotspotter()
    np.set_printoptions(threshold=5000, linewidth=5000, precision=3)

    #adaptive_locals = test_adaptive_scale()
    # They seem to work
    test_adaptive_scale()

    #exec(helpers.execstr_dict(test_locals, 'test_locals'))
    #if '--cmd' in sys.argv:
    #exec(helpers.execstr_dict(adaptive_locals, 'adaptive_locals'))
    #in_depth_locals = adaptive_locals['in_depth_locals']
    #exec(helpers.execstr_dict(in_depth_locals, 'in_depth_locals'))
    #exec(df2.present(override1=True))
    exec(df2.present())
