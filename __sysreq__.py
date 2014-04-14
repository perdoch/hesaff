from __future__ import absolute_import, print_function, division
import sys
from os.path import join, exists, expanduser


def ensure_ibeis():
    import matplotlib
    matplotlib.use('Qt4Agg', warn=True, force=True)
    # Look for hotspotter in ~/code
    ibeis_dir = join(expanduser('~'), 'code', 'ibeis')
    if not exists(ibeis_dir):
        print('[jon] ibeis_dir=%r DOES NOT EXIST!' % (ibeis_dir,))
    # Append hotspotter to PYTHON_PATH (i.e. sys.path)
    if not ibeis_dir in sys.path:
        sys.path.append(ibeis_dir)

ensure_ibeis()
