# Set defaults for --build-arg
#FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04
#FROM jjanzic/docker-python3-opencv
#FROM python:3.6

# Notes:
#docker pull quay.io/skvark/manylinux1_x86_64
# docker build -t build_hesaff . && docker run -it build_hesaff bash
# docker run -it build_hesaff bash

FROM quay.io/skvark/manylinux1_x86_64

ARG MB_PYTHON_VERSION=3.6
ARG ENABLE_CONTRIB=1
ARG ENABLE_HEADLESS=0

ENV PYTHON_VERSION=3.6
ENV MULTIBUILD_DIR=/root/code/multibuild
ENV HOME=/root

WORKDIR /root
RUN git clone https://github.com/matthew-brett/multibuild.git $MULTIBUILD_DIR
RUN find $MULTIBUILD_DIR -iname "*.sh" -type f -exec sed -i 's/gh-clone/gh_clone/g' {} +

COPY config.sh /root/config.sh
COPY config.sh /root/.bashrc

RUN source /root/config.sh && $PYTHON_EXE -m pip install --upgrade pip
#RUN source /root/config.sh && $PYTHON_EXE -m pip install --upgrade venv
RUN source /root/config.sh && setup_venv

RUN source /root/config.sh && python -m pip install cmake ninja -U && python -m pip install scikit-build numpy


#RUN source /root/config.sh && source $MULTIBUILD_DIR/docker_build_wrap.sh 



#RUN
#source config.sh && source multibuild/docker_build_wrap.sh 


#RUN source multibuild/docker_build_wrap.sh 

#RUN cat multibuild/common_utils.sh
#&& source multibuild/common_utils.sh


## https://github.com/skvark/opencv-python/blob/master/setup.py
#cmake -G "Unix Makefiles"  
#        "-DPYTHON%d_EXECUTABLE=%s" % (sys.version_info[0], sys.executable),
#        "-DBUILD_opencv_python%d=ON" % sys.version_info[0],
#        "-DOPENCV_SKIP_PYTHON_LOADER=ON",
#        "-DOPENCV_PYTHON%d_INSTALL_PATH=python" % sys.version_info[0],
#        "-DINSTALL_CREATE_DISTRIB=ON",
        
#        # See opencv/CMakeLists.txt for options and defaults
#        "-DBUILD_opencv_apps=OFF",
#        "-DBUILD_SHARED_LIBS=OFF",
#        "-DBUILD_TESTS=OFF",
#        "-DBUILD_PERF_TESTS=OFF",
#        "-DBUILD_DOCS=OFF"
#    ] + (["-DOPENCV_EXTRA_MODULES_PATH=" + os.path.abspath("opencv_contrib/modules")] if build_contrib else [])


#RUN python3 -m pip install pip -U
#RUN python3 -m pip install cmake ninja -U
#RUN python3 -m pip install scikit-build numpy

#### OpenCV Part

#WORKDIR /
ENV OPENCV_VERSION="4.1.0"
RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
&& unzip ${OPENCV_VERSION}.zip \
&& mkdir /opencv-${OPENCV_VERSION}/cmake_binary 

RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
&& cd /opencv-${OPENCV_VERSION}/cmake_binary \
#&& cmake -DBUILD_TIFF=ON \
#  -DBUILD_opencv_java=OFF \
#  -DWITH_CUDA=OFF \
#  -DWITH_OPENGL=ON \
#  -DWITH_OPENCL=ON \
#  -DWITH_IPP=ON \
#  -DWITH_TBB=ON \
#  -DWITH_EIGEN=ON \
#  -DWITH_V4L=ON \
#  -DBUILD_TESTS=OFF \
#  -DBUILD_PERF_TESTS=OFF \
#  -DCMAKE_BUILD_TYPE=RELEASE \
#  -DCMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
#  -DPYTHON_EXECUTABLE=$(which python3) \
#  -DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
#  -DPYTHON_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
#  .. \
#&& make install

#RUN ln -s \
#  /usr/local/python/cv2/python-3.6/cv2.cpython-36m-x86_64-linux-gnu.so \
#  /usr/local/lib/python3.6/site-packages/cv2.so

####


#WORKDIR /root
#RUN mkdir -p $HOME/code/hesaff

#COPY pyhesaff /root/code/hesaff/pyhesaff
#COPY setup.py /root/code/hesaff/setup.py
#COPY CMake /root/code/hesaff/CMake
#COPY CMakeLists.txt /root/code/hesaff/CMakeLists.txt
#COPY src /root/code/hesaff/src
#COPY pyproject.toml /root/code/hesaff/pyproject.toml
#COPY requirements.txt /root/code/hesaff/requirements.txt
#COPY run_developer_setup.sh /root/code/hesaff/run_developer_setup.sh

#WORKDIR /root/code/hesaff
##RUN ./run_developer_setup.sh
##RUN python3 setup.py clean
#RUN python3 -m pip install -r requirements.txt

#RUN CMAKE_FIND_LIBRARY_SUFFIXES=".a;.so" python3 setup.py build_ext --inplace
    
##RUN python3 setup.py develop

#RUN pip install xdoctest
#COPY run_doctests.sh /root/code/hesaff/run_doctests.sh
## COPY run_tests.sh /root/code/hesaff/run_tests.py

#RUN ls $HOME/code/hesaff

## docker build --tag build_hesaff -f Dockerfile .
## docker run -it build_hesaff bash