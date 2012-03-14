all: *.cpp
	g++ -O3 -Wall `pkg-config --cflags --libs opencv` -lrt -o hesaff pyramid.cpp affine.cpp siftdesc.cpp helpers.cpp hesaff.cpp
