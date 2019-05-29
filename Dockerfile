# Set defaults for --build-arg
#FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04
#FROM jjanzic/docker-python3-opencv
FROM spmallick/opencv-docker:opencv
ENV TERM linux

# Fixes display issues when we write out utf-8 text
ENV PYTHONIOENCODING UTF-8
ENV LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        nano vim curl git bzip2 openssh-client make \
        gcc g++ gfortran build-essential \
        tree htop tmux astyle \
        libglib2.0-0 libsm6 libice6 libsm6 libxt6 libxrender1 libfontconfig1 libcups2 libxext6 \
        && \
    rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]  
#RUN ln -sf /bin/bash /bin/sh

RUN python3 -m pip install pip -U
RUN python3 -m pip install cmake ninja -U
RUN python3 -m pip install scikit-build numpy

WORKDIR /root
RUN mkdir -p $HOME/code/hesaff

COPY pyhesaff /root/code/hesaff/pyhesaff
COPY setup.py /root/code/hesaff/setup.py
COPY CMake /root/code/hesaff/CMake
COPY CMakeLists.txt /root/code/hesaff/CMakeLists.txt
COPY src /root/code/hesaff/src
COPY pyproject.toml /root/code/hesaff/pyproject.toml
COPY requirements.txt /root/code/hesaff/requirements.txt
COPY run_developer_setup.sh /root/code/hesaff/run_developer_setup.sh

WORKDIR /root/code/hesaff
#RUN ./run_developer_setup.sh
#RUN python3 setup.py clean
RUN python3 -m pip install -r requirements.txt

RUN python3 setup.py build_ext --inplace

RUN python3 setup.py develop

RUN pip install xdoctest
COPY run_doctests.sh /root/code/hesaff/run_doctests.sh
# COPY run_tests.sh /root/code/hesaff/run_tests.py

RUN ls $HOME/code/hesaff

# docker build --tag build_hesaff -f Dockerfile .
# docker run -it build_hesaff bash
