FROM quay.io/skvark/manylinux1_x86_64

ARG MB_PYTHON_VERSION=3.6
ARG ENABLE_CONTRIB=1
ARG ENABLE_HEADLESS=0

ENV PYTHON_VERSION=3.6
ENV MULTIBUILD_DIR=/root/code/multibuild
ENV HOME=/root

WORKDIR /root
COPY docker/staging/multibuild $MULTIBUILD_DIR
# Hack to fix issue
RUN find $MULTIBUILD_DIR -iname "*.sh" -type f -exec sed -i 's/gh-clone/gh_clone/g' {} +

COPY utils.sh /root/utils.sh
COPY bashrc.sh /root/.bashrc

# RUN source /root/.bashrc && $PYTHON_EXE -m pip install --upgrade pip
# RUN source /root/.bashrc && setup_venv

# RUN source /root/.bashrc && python -m pip install cmake ninja -U && python -m pip install scikit-build numpy
# RUN source /root/.bashrc.sh && build_openssl
# RUN source /root/.bashrc.sh && build_curl