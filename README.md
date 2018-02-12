# DESCRIPTION

This is an implementation of Hessian-Affine detector. 

The implementation uses a Lowe's (Lowe 1999, Lowe 2004) like pyramid
to sample Gaussian scale-space and localizes local extrema of the
Detetminant of Hessian Matrix operator computed on normalized
derivatives. Then a Baumberg-Lindeberg discovery of a local affine
shape is employed (Lindeberg 1998, Baumberg 2000, Mikolajzyk 2002) to
compute affine shape of each det of Hessian extrema. Finally a local
neighbourhood is normalized to a fixed size patch and SIFT
descriptor(Lowe 1999, Lowe 2004) computed.


# IMPLEMENTATION

## Installing OpenCV
```
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 2154e62 # Checkout for 2.3.1 version commit point
mkdir build
cd build
# This old OpenCV may not support your recent GPU. Just disable it.
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_CUDA=OFF ..
make -j8
sudo make install
```

## Build

```
cd hesaff
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
make
```

## Run

```
# Assume that libopencv_core.so.2.3 in in /usr/local/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
# Usage: hesaff image_name.ppm
hesaff oxford_001.ppm
```

Implementation depends on OpenCV (2.3.1+). Although, the code is
original, the affine iteration and normalization was derived from the
code of Krystian Mikolajczyk.

The SIFT descriptor code is protected under a US Patent 6,711,293. A
license MUST be obtained from the University of British Columbia for
use of SIFT code, files siftdesc.cpp/siftdesc.h, in commercial
applications (see LICENSE.SIFT for details)


# OUTPUT

The built binary rewrites output file: <input_image_name>.hesaff.sift

The output format is compatible with the binaries available from the
page "Affine Covariant Features". The geometry of an affine region is
specified by: u,v,a,b,c in a(x-u)(x-u)+2b(x-u)(y-v)+c(y-v)(y-v)=1. The
top left corner of the image is at (u,v)=(0,0). The geometry of an
affine region is followed by N descriptor values (N = 128).  

File format:

N
m
u1 v1 a1 b1 c1 d1(1) d1(2) d1(3) ... d1(N)
      :
      :
um vm am bm cm dm(1) dm(2) dm(3) ... dm(N)


# PROPER USE

If you use this code, please refer to

Perdoch, M. and Chum, O. and Matas, J.: Efficient Representation of
Local Geometry for Large Scale Object Retrieval. In proceedings of
CVPR09. June 2009.

TBD: A reference to technical report describing the details and some
retrieval results will be placed here.
