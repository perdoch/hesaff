 FROM quay.io/skvark/manylinux1_x86_64

 # SETUP ENV
 ARG MB_PYTHON_VERSION=3.6
 ARG ENABLE_CONTRIB=0
 ARG ENABLE_HEADLESS=1
 ENV PYTHON_VERSION=3.6
 ENV PYTHON_ROOT=/opt/python/cp36-cp36m/
 ENV PYTHONPATH=/opt/python/cp36-cp36m/lib/python3.6/site-packages/
 ENV PATH=/opt/python/cp36-cp36m/bin:$PATH
 ENV PYTHON_EXE=/opt/python/cp36-cp36m/bin/python
 ENV HOME=/root
 ENV PLAT=x86_64
 ENV UNICODE_WIDTH=32

 # Update python environment
 RUN echo "$PYTHON_EXE"
 RUN $PYTHON_EXE -m pip install --upgrade pip &&             $PYTHON_EXE -m pip install cmake ninja scikit-build wheel numpy

 # This is very different for different operating systems
 # https://github.com/skvark/opencv-python/blob/master/setup.py
 COPY docker/staging/opencv /root/code/opencv
 RUN mkdir -p /root/code/opencv/build &&             cd /root/code/opencv/build &&             cmake -G "Unix Makefiles"                    -DINSTALL_CREATE_DISTRIB=ON                    -DOPENCV_SKIP_PYTHON_LOADER=ON                    -DBUILD_opencv_apps=OFF                    -DBUILD_SHARED_LIBS=OFF                    -DBUILD_TESTS=OFF                    -DBUILD_PERF_TESTS=OFF                    -DBUILD_DOCS=OFF                    -DWITH_QT=OFF                    -DWITH_IPP=OFF                    -DWITH_V4L=ON                    -DBUILD_JPEG=OFF                    -DENABLE_PRECOMPILED_HEADERS=OFF                 /root/code/opencv

# Note: there is no need to compile the above with python
# -DPYTHON3_EXECUTABLE=$PYTHON_EXE        # -DBUILD_opencv_python3=ON        # -DOPENCV_PYTHON3_INSTALL_PATH=python 
 RUN cd /root/code/opencv/build && make -j9 && make install