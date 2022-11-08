# Copyright (c) 2018-2020 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Released under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.

ARG ROOT_IMAGE

FROM ${ROOT_IMAGE} AS devenv

ARG COMPILERS="gcc clang"

ARG CMAKE_VER=3.23.1
ARG CMAKE_URL=https://github.com/Kitware/CMake/releases/download/v${CMAKE_VER}/cmake-${CMAKE_VER}-Linux-x86_64.sh

ARG UBUNTU_TOOL_DEB_REPO=https://ppa.launchpadcontent.net/ubuntu-toolchain-r/ppa/ubuntu
ARG UBUNTU_TOOL_FINGER=60C317803A41BA51845E371A1E9377A2BA9EF27F

ARG LLVM_INSTALLER=https://apt.llvm.org/llvm.sh
# `-y` answers yes to any interactive prompts.
# `-qq` because apt is noisy
ARG APT_GET="apt-get -y -qq"

ENV TZ=US/Pacific
ENV DEBIAN_FRONTEND=noninteractive
# apt-key complains about non-interactive usage.
ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

SHELL ["/usr/bin/env", "bash", "-c"]

ADD ${LLVM_INSTALLER} /tmp/llvm.sh
ADD ${CMAKE_URL} /tmp/cmake.sh

# Install baseline development tools
RUN function comment() { :; }; \
    comment "Sources for gcc"; \
    source /etc/os-release; \
    echo "deb ${UBUNTU_TOOL_DEB_REPO} ${VERSION_CODENAME} main" >> /etc/apt/sources.list.d/ubuntu-archive.list; \
    echo "deb-src ${UBUNTU_TOOL_DEB_REPO} ${VERSION_CODENAME} main" >> /etc/apt/sources.list.d/ubuntu-archive.list; \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys ${UBUNTU_TOOL_FINGER}; \
    ${APT_GET} update; \
    comment "Install basic build tools"; \
    ${APT_GET} --no-install-recommends install apt-utils curl wget git zip unzip tar \
        sudo make software-properties-common ninja-build ccache pkg-config \
        python3 python3-wheel python3-pip; \
    comment "Install GCC and Clang"; \
    # Unattended installation hack
    echo "\n" | bash /tmp/llvm.sh all; \
    ${APT_GET} install gcc g++ ${COMPILERS}; \
    comment "Install CMake"; \
    sh /tmp/cmake.sh --skip-license --prefix=/usr; \
    comment "Install Python utils"; \
    update-alternatives --quiet --install /usr/bin/python python $(which python3) 3; \
    update-alternatives --quiet --set python $(which python3); \
    python3 -m pip install setuptools wheel; \
    python3 -m pip install lit; \
    rm -rf /var/lib/apt/lists/*

# Assemble libcudacxx specific bits
FROM devenv AS libcudacxx-configured

# Default path according to CUDA Docker image, overridable if the image requires it
ARG CUDACXX_PATH=/usr/local/cuda/bin/nvcc

ARG HOST_CXX=gcc
ARG CXX_DIALECT=11

# Attempt to load env from cccl/cuda
ARG TEST_WITH_NVRTC=OFF

# Docker on Windows can't follow symlinks???
ADD .                                        /libcudacxx
ADD ./include/cuda/std/detail/libcxx/include /libcudacxx/libcxx/include
ADD ./.upstream-tests/test                   /libcudacxx/test
ADD ./.upstream-tests/utils                   /libcudacxx/utils

# Install compiler and configure project
RUN cmake -S /libcudacxx -B /build \
        -DLIBCUDACXX_ENABLE_STATIC_LIBRARY=OFF \
        -DLIBCUDACXX_ENABLE_LIBCUDACXX_TESTS=ON \
        -DLIBCUDACXX_ENABLE_LIBCXX_TESTS=ON \
        -DLIBCUDACXX_TEST_COMPILER_FLAGS="-allow-unsupported-compiler" \
        -DLIBCUDACXX_TEST_WITH_NVRTC=${TEST_WITH_NVRTC} \
        -DLIBCUDACXX_TEST_STANDARD_VER=c++${CXX_DIALECT} \
        -DLIBCXX_ENABLE_FILESYSTEM=OFF \
        -DCMAKE_CXX_COMPILER=${HOST_CXX} \
        -DCMAKE_CUDA_COMPILER=${CUDACXX_PATH} \
        -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler"

RUN make -j -C /build/libcxx

ENV LIBCUDACXX_SITE_CONFIG=/build/test/lit.site.cfg
ENV LIBCXX_SITE_CONFIG=/build/libcxx/test/lit.site.cfg
