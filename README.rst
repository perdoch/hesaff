|GithubActions| |Codecov| |Pypi| |Downloads| |ReadTheDocs|


Hessian Affine + SIFT keypoints in Python
=========================================

This is an implementation of Hessian-Affine detector. 

The implementation uses a Lowe's (Lowe 1999, Lowe 2004) like pyramid
to sample Gaussian scale-space and localizes local extrema of the
Detetminant of Hessian Matrix operator computed on normalized
derivatives. Then a Baumberg-Lindeberg discovery of a local affine
shape is employed (Lindeberg 1998, Baumberg 2000, Mikolajzyk 2002) to
compute affine shape of each det of Hessian extrema. Finally a local
neighbourhood is normalized to a fixed size patch and SIFT
descriptor(Lowe 1999, Lowe 2004) computed.


BUILDING
--------

There are wheels publishe on pypi using cibuildwheel.


IMPLEMENTATION
--------------

Implementation depends on OpenCV (2.3.1+). Although, the code is
original, the affine iteration and normalization was derived from the
code of Krystian Mikolajczyk.

The SIFT descriptor code was patented under a US Patent 6,711,293, which
expired on March 7th 2019, so the license is no longer required for use. 


OUTPUT
------

NOTE THIS IS NO LONGER THE CASE. WE MAY REINSTATE THIS.

The built binary rewrites output file: <input_image_name>.hesaff.sift

The output format is compatible with the binaries available from the
page "Affine Covariant Features". The geometry of an affine region is
specified by: u,v,a,b,c in a(x-u)(x-u)+2b(x-u)(y-v)+c(y-v)(y-v)=1. The
top left corner of the image is at (u,v)=(0,0). The geometry of an
affine region is followed by N descriptor values (N = 128).  

File format:

::

    N
    m
    u1 v1 a1 b1 c1 d1(1) d1(2) d1(3) ... d1(N)
          :
          :
    um vm am bm cm dm(1) dm(2) dm(3) ... dm(N)


PROPER USE
----------

If you use this code, please refer to

Perdoch, M. and Chum, O. and Matas, J.: Efficient Representation of
Local Geometry for Large Scale Object Retrieval. In proceedings of
CVPR09. June 2009.

TBD: A reference to technical report describing the details and some
retrieval results will be placed here.


NOTES
-----

Requires opencv. On ubuntu you can: ``sudo apt-get install libopencv-dev``. You can also build / use wheels. 


.. |CircleCI| image:: https://circleci.com/gh/Erotemic/pyhesaff.svg?style=svg
    :target: https://circleci.com/gh/Erotemic/pyhesaff
.. |Travis| image:: https://img.shields.io/travis/Erotemic/pyhesaff/main.svg?label=Travis%20CI
   :target: https://travis-ci.org/Erotemic/pyhesaff?branch=main
.. |Appveyor| image:: https://ci.appveyor.com/api/projects/status/github/Erotemic/pyhesaff?branch=master&svg=True
   :target: https://ci.appveyor.com/project/Erotemic/pyhesaff/branch/main
.. |Codecov| image:: https://codecov.io/github/Erotemic/pyhesaff/badge.svg?branch=main&service=github
   :target: https://codecov.io/github/Erotemic/pyhesaff?branch=main
.. |Pypi| image:: https://img.shields.io/pypi/v/pyhesaff.svg
   :target: https://pypi.python.org/pypi/pyhesaff
.. |Downloads| image:: https://img.shields.io/pypi/dm/pyhesaff.svg
   :target: https://pypistats.org/packages/pyhesaff
.. |ReadTheDocs| image:: https://readthedocs.org/projects/pyhesaff/badge/?version=latest
    :target: http://pyhesaff.readthedocs.io/en/latest/
.. |GithubActions| image:: https://github.com/Erotemic/pyhesaff/actions/workflows/tests.yml/badge.svg?branch=main
    :target: https://github.com/Erotemic/pyhesaff/actions?query=branch%3Amain
