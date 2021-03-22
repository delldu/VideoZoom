#!/usr/bin/env bash

# You may need to modify the following paths before compiling.

CUDA_HOME=/usr/local/cuda-10.2 \
CUDNN_INCLUDE_DIR=/usr/local/cuda-10.2/include \
CUDNN_LIB_DIR=/usr/local/cuda-10.2/lib64 \

rm -rf build/ DCNv2.egg-info/ *.so
export PATH=$CUDA_HOME/bin:$PATH

python setup.py build develop 2>&1 | tee /tmp/error.log
