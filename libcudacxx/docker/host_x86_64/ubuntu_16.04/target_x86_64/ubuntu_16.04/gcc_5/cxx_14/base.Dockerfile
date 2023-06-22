# Dockerfile for libcudacxx_base:host_x86_64_ubuntu_16.04__target_x86_64_ubuntu_16.04__gcc_5_cxx_14

FROM ubuntu:16.04

MAINTAINER Bryce Adelstein Lelbach <blelbach@nvidia.com>

ARG LIBCUDACXX_SKIP_BASE_TESTS_BUILD
ARG LIBCUDACXX_COMPUTE_ARCHS
ARG COMPILER_CXX_DIALECT

ENV COMPILER_CXX_DIALECT=$COMPILER_CXX_DIALECT

###############################################################################
# BUILD: The following is invoked when the image is built.

SHELL ["/usr/bin/env", "bash", "-c"]

RUN apt-get -y update\
 && apt-get -y install g++-5 clang-5.0 python python-pip\
 && pip install lit==0.11.0\
 && mkdir -p /sw/gpgpu/libcudacxx/build\
 && mkdir -p /sw/gpgpu/libcudacxx/libcxx/build

# Update CMake to 3.18
ADD https://github.com/Kitware/CMake/releases/download/v3.18.5/cmake-3.18.5-Linux-x86_64.sh /tmp/cmake.sh
RUN sh /tmp/cmake.sh --skip-license --prefix=/usr

# For debugging.
#RUN apt-get -y install gdb strace vim

# We use ADD here because it invalidates the cache for subsequent steps, which
# is what we want, as we need to rebuild if the sources have changed.

# Copy NVCC and the CUDA runtime from the source tree.
ADD bin /sw/gpgpu/bin


# Copy libcu++ sources from the source tree.
ADD libcudacxx /sw/gpgpu/libcudacxx

# List out everything in /sw before the build.
RUN echo "Contents of /sw:" && cd /sw/ && find

# Build libc++ and configure libc++ tests.
RUN set -o pipefail; cd /sw/gpgpu/libcudacxx/libcxx/build\
 && cmake ../..\
 -DLIBCUDACXX_ENABLE_STATIC_LIBRARY=OFF\
 -DLIBCUDACXX_ENABLE_LIBCUDACXX_TESTS=OFF\
 -DLIBCUDACXX_ENABLE_LIBCXX_TESTS=ON\
 -DLIBCXX_TEST_STANDARD_VER=c++$COMPILER_CXX_DIALECT\
 -DCMAKE_CXX_COMPILER=g++-5\
 -DCMAKE_CUDA_COMPILER=/sw/gpgpu/bin/x86_64_Linux_release/nvcc\
 && make -j\
 2>&1 | tee /sw/gpgpu/libcudacxx/build/cmake_libcxx.log

# Configure libcu++ tests.
RUN set -o pipefail; cd /sw/gpgpu/libcudacxx/build\
 && cmake ..\
 -DCMAKE_CXX_COMPILER=g++-5\
 -DCMAKE_CUDA_COMPILER=/sw/gpgpu/bin/x86_64_Linux_release/nvcc\
 -DLIBCUDACXX_ENABLE_LIBCUDACXX_TESTS=ON\
 -DLIBCUDACXX_ENABLE_LIBCXX_TESTS=OFF\
 -DLIBCUDACXX_TEST_STANDARD_VER=c++$COMPILER_CXX_DIALECT\
 2>&1 | tee /sw/gpgpu/libcudacxx/build/cmake_libcudacxx.log

# Package the logs up, because `docker container cp` doesn't support wildcards,
# and I want to limit the number of places we have to make changes when we ship
# new architectures.
RUN cd /sw/gpgpu/libcudacxx/build && tar -cvf logs.tar *.log

WORKDIR /sw/gpgpu/libcudacxx
