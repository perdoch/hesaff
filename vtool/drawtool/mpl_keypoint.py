from __future__ import division, print_function
# Standard
from itertools import izip
# Science
import numpy as np
# Matplotlib
import matplotlib as mpl
# vtool
import vtool.keypoint as ktool
from vtool.drawtool import mpl_sift


# TOOD: move to util
def pass_props(dict1, dict2, *args):
    # Passes props from one kwargs dict to the next
    for key in args:
        if key in dict1:
            dict2[key] = dict1[key]


def draw_patches(ax, patch_list, color, alpha, lw, fcolor='none'):
    coll = mpl.collections.PatchCollection(patch_list)
    coll.set_facecolor(fcolor)
    coll.set_alpha(alpha)
    coll.set_linewidth(lw)
    coll.set_edgecolor(color)
    coll.set_transform(ax.transData)
    ax.add_collection(coll)


#----------------------------
def draw_keypoints(ax, kpts, scale_factor=1.0, offset=(0.0, 0.0), ell=True,
                   pts=False, rect=False, eig=False, ori=False,  sifts=None,
                   **kwargs):
    '''
    draws keypoints extracted by pyhesaff onto a matplotlib axis
    '''
    # ellipse and point properties
    pts_size       = kwargs.get('pts_size', None)
    ell_color      = kwargs.get('ell_color', None)
    ell_alpha      = kwargs.get('ell_alpha', 1)
    ell_linewidth  = kwargs.get('ell_linewidth', 2)
    # colors
    pts_color      = kwargs.get('pts_color', ell_color)
    rect_color     = kwargs.get('rect_color', ell_color)
    eig_color      = kwargs.get('eig_color', ell_color)
    ori_color      = kwargs.get('ori_color', ell_color)
    # linewidths
    eig_linewidth  = kwargs.get('eig_linewidth', ell_linewidth)
    rect_linewidth = kwargs.get('rect_linewidth', ell_linewidth)
    ori_linewidth  = kwargs.get('ori_linewidth', ell_linewidth)
    # Extract keypoint components
    (_xs, _ys, _as, _bs, _cs, _ds, _oris) = ktool.scaled_kpts(kpts, scale_factor, offset)
    # Build list of keypoint shape transforms from unit circles to ellipes
    aff_list = get_aff_list(_xs, _ys, _as, _bs, _cs, _ds, _oris)
    aff_list_noori = get_aff_list(_xs, _ys, _as, _bs, _cs, _ds)
    try:
        if sifts is not None:
            # SIFT descriptors
            sift_kwargs = {}
            pass_props(kwargs, sift_kwargs, 'bin_color', 'arm1_color', 'arm2_color')
            mpl_sift.draw_sifts(ax, sifts, aff_list, **sift_kwargs)
        if rect:
            # Bounding Rectangles
            rect_patches = rectangle_actors(aff_list)
            draw_patches(ax, rect_patches, rect_color, ell_alpha, rect_linewidth)
        if ell:
            # Keypoint shape
            ell_patches = ellipse_actors(aff_list_noori)
            draw_patches(ax, ell_patches, ell_color, ell_alpha, ell_linewidth)
        if eig:
            # Shape eigenvectors
            eig_patches = eigenvector_actors(aff_list_noori)
            draw_patches(ax, eig_patches, eig_color, ell_alpha, eig_linewidth)
        if ori:
            # Keypoint orientation
            ori_patches = orientation_actors(_xs, _ys, _as, _cs, _ds, _oris)
            draw_patches(ax, ori_patches, ori_color, ell_alpha, ori_linewidth, ori_color)
        if pts:
            # Keypoint locations
            _draw_pts(ax, _xs, _ys, pts_size, pts_color)
    except ValueError as ex:
        print('\n[mplkp] !!! ERROR %s: ' % str(ex))
        print('_oris.shape = %r' % (_oris.shape,))
        print('_xs.shape = %r' % (_xs.shape,))
        print('_as.shape = %r' % (_as.shape,))
        raise

#----------------------------


def _draw_pts(ax, _xs, _ys, pts_size, pts_color):
    ax.autoscale(enable=False)
    ax.scatter(_xs, _ys, c=pts_color, s=(2 * pts_size), marker='o', edgecolor='none')


def get_aff_list(_xs, _ys, _as, _bs, _cs, _ds, _oris=None):
    kpts_iter = izip(_xs, _ys, _as, _bs, _cs, _ds)
    aff_list = [mpl.transforms.Affine2D([(a, b, x),
                                         (c, d, y),
                                         (0, 0, 1)])
                for (x, y, a, b, c, d) in kpts_iter]
    if _oris is not None:
        rot_list = [mpl.transforms.Affine2D().rotate(ori) for ori in _oris]
        #aff_list = [aff + rot for (aff, rot) in izip(aff_list, rot_list)]
        aff_list = [rot + aff for (aff, rot) in izip(aff_list, rot_list)]
    return aff_list


def ellipse_actors(aff_list):
    # warp unit circles to keypoint shapes
    ell_actors = [mpl.patches.Circle((0, 0), 1, transform=aff)
                  for aff in aff_list]
    return ell_actors


def rectangle_actors(aff_list):
    # warp unit rectangles to keypoint shapes
    rect_actors = [mpl.patches.Rectangle((-1, -1), 2, 2, transform=aff)
                   for aff in aff_list]
    return rect_actors


def eigenvector_actors(aff_list):
    # warps arrows into eigenvector directions
    kwargs = {
        'head_width': .01,
        'length_includes_head': False,
    }
    eig1 = [mpl.patches.FancyArrow(0, 0, 0, 1, transform=aff, **kwargs)
            for aff in aff_list]
    eig2 = [mpl.patches.FancyArrow(0, 0, 1, 0, transform=aff, **kwargs)
            for aff in aff_list]
    eig_actors = eig1 + eig2
    return eig_actors


def orientation_actors(_xs, _ys, _as, _cs, _ds, _oris):
    try:
        _sins = np.sin(_oris)
        _coss = np.cos(_oris)
        # scaled orientation x and y direction relative to center
        #sf = np.sqrt(_as * _ds)
        _dxs = _coss * _as
        _dys = _sins * _ds + _coss * _cs
        head_width_list = np.sqrt(_as * _ds) / 5
        kwargs = {
            'length_includes_head': True,
            'shape': 'full',
            'overhang': 0,
            'head_starts_at_zero': False,
        }
        ori_actors = [mpl.patches.FancyArrow(x, y, dx, dy, head_width=hw, **kwargs)
                      for (x, y, dx, dy, hw) in
                      izip(_xs, _ys, _dxs, _dys, head_width_list)]
    except ValueError as ex:
        print('\n[mplkp.2] !!! ERROR %s: ' % str(ex))
        print('_oris.shape = %r' % (_oris.shape,))
        print('x, y, dx, dy = %r' % ((x, y, dx, dy),))
        print('_dxs = %r' % (_dxs,))
        print('_dys = %r' % (_dys,))
        print('_xs = %r' % (_xs,))
        print('_ys = %r' % (_ys,))
        raise

    return ori_actors
