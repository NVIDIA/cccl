# Copyright (c) 2018-2020 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Released under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.

# SDK_TYPE needs to be a base image that contains CUDA, either:
# - nvidia/cuda
# - nvcr.io/nvidia/nvhpc
ARG SDK_TYPE=nvidia/cuda
ARG SDK_VER=11.0-devel
ARG OS_TYPE=ubuntu
ARG OS_VER=18.04
# | CXX_TYPE | CXX_VER      |
# | gcc      | 5 6 7 8 9 10 |
# | clang    | 7 8 9 10     |
# | icc      | latest       |
# | nvcxx    | 20.7         |
ARG CXX_TYPE=gcc
ARG CXX_VER=5
FROM ${SDK_TYPE}:${SDK_VER}-${OS_TYPE}${OS_VER}

# Pull arguments from build arguments.
ARG SDK_TYPE
ARG SDK_VER
ARG OS_TYPE
ARG OS_VER
ARG CXX_TYPE
ARG CXX_VER

# Ubuntu 18.04 doesn't have GCC 9 and 10 in its repos.
ARG UBUNTU_TOOL_DEB_REPO=http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu
ARG UBUNTU_TOOL_FINGER=60C317803A41BA51845E371A1E9377A2BA9EF27F

ARG ICC_DEB_REPO=https://apt.repos.intel.com/oneapi
ARG ICC_KEY=https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB

# CentOS 7 doesn't have a new enough version of CMake in its repos.
ARG CMAKE_VER=3.18.4
ARG CMAKE_URL=https://github.com/Kitware/CMake/releases/download/v${CMAKE_VER}/cmake-${CMAKE_VER}-Linux-x86_64.sh

ARG TINI_VER=0.18.0
ARG TINI_URL=https://github.com/krallin/tini/releases/download/v${TINI_VER}/tini

ENV TZ=US/Pacific
ENV DEBIAN_FRONTEND=noninteractive
# apt-key complains about non-interactive usage.
ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

SHELL ["/usr/bin/env", "bash", "-c"]

RUN if [[ "${OS_TYPE}" == "ubuntu"* ]]; then \
      export ALTERNATIVES=update-alternatives; \
      apt-get -y update; \
      apt-get -y --no-install-recommends install apt-utils; \
      apt-get -y --no-install-recommends install curl; \
      if   [[ "${CXX_TYPE}" == "gcc" && "${CXX_VER}" > 8 ]]; then \
        source /etc/os-release; \
        echo "deb ${UBUNTU_TOOL_DEB_REPO} ${UBUNTU_CODENAME} main" >> /etc/apt/sources.list; \
        apt-key adv --keyserver keyserver.ubuntu.com --recv-keys ${UBUNTU_TOOL_FINGER}; \
      elif [[ "${CXX_TYPE}" == "icc" ]]; then \
        echo "deb ${ICC_DEB_REPO} all main" >> /etc/apt/sources.list; \
        curl --silent --show-error -L ${ICC_KEY} -o - | apt-key add -; \
      fi; \
      apt-get -y update; \
      apt-get -y --no-install-recommends install python3-pip python3-setuptools python3-wheel; \
      ${ALTERNATIVES} --install /usr/bin/python python $(which python3) 3; \
      ${ALTERNATIVES} --install /usr/bin/pip    pip    $(which pip3)    3; \
      ${ALTERNATIVES} --set python $(which python3); \
      ${ALTERNATIVES} --set pip    $(which pip3); \
      apt-get -y --no-install-recommends install make; \
      apt-get -y --no-install-recommends install sudo; \
      apt-get -y --no-install-recommends install gdb; \
      apt-get -y --no-install-recommends install strace; \
      apt-get -y --no-install-recommends install vim; \
      apt-get -y --no-install-recommends install llvm-dev; \
      if   [[ "${CXX_TYPE}" == "gcc" ]]; then \
        apt-get -y --no-install-recommends install g++-${CXX_VER}; \
        export CC=$(which gcc-${CXX_VER}); \
        export CXX=$(which g++-${CXX_VER}); \
      elif [[ "${CXX_TYPE}" == "clang" ]]; then \
        apt-get -y --no-install-recommends install clang-${CXX_VER}; \
        export CC=$(which clang-${CXX_VER}); \
        export CXX=$(which clang++-${CXX_VER}); \
      elif [[ "${CXX_TYPE}" == "icc" ]]; then \
        apt-get -y --no-install-recommends install intel-oneapi-icc g++; \
        source /opt/intel/oneapi/setvars.sh; \
        echo "source /opt/intel/oneapi/setvars.sh" >> /etc/cccl.bashrc; \
        export CC=$(which icc); \
        export CXX=$(which icpc); \
      elif [[ "${CXX_TYPE}" == "nvcxx" ]]; then \
        apt-get -y --no-install-recommends install g++; \
        export CC=$(which nvc); \
        export CXX=$(which nvc++); \
      fi; \
      apt-get clean; \
      rm -rf /var/lib/apt/lists/*; \
    elif [[ "${OS_TYPE}" == "centos" ]]; then \
      export ALTERNATIVES=alternatives; \
      yum -y --enablerepo=extras install epel-release; \
      yum -y updateinfo; \
      yum -y install which; \
      yum -y install python python-pip; \
      yum -y install make; \
      yum -y install llvm-devel; \
      yum -y install sudo; \
      yum -y install gdb; \
      yum -y install strace; \
      yum -y install vim; \
      if [[ "${CXX_TYPE}" == "nvcxx" ]]; then \
        yum -y install gcc-c++ libstdc++-static; \
        rm /usr/bin/cc; \
        rm /usr/bin/c++; \
        export CC=$(which nvc); \
        export CXX=$(which nvc++); \
      else \
        echo -e "\n\n>>>> ERROR: ${CXX_TYPE} is not supported on ${OS_TYPE}.\n\n"; \
        exit 1; \
      fi; \
    fi; \
    ${ALTERNATIVES} --install /usr/bin/cc  cc  ${CC}  99; \
    ${ALTERNATIVES} --install /usr/bin/c++ c++ ${CXX} 99; \
    ${ALTERNATIVES} --set cc  ${CC}; \
    ${ALTERNATIVES} --set c++ ${CXX}; \
    if   [[ "${SDK_TYPE}" == "nvidia/cuda" ]]; then \
      export CUDACXX=$(which nvcc); \
    elif [[ "${SDK_TYPE}" == "nvcr.io/nvidia/nvhpc" ]]; then \
      export CUDACXX=$(which nvc++); \
    fi; \
    echo "export CC=${CC}" >> /etc/cccl.bashrc; \
    echo "export CXX=${CXX}" >> /etc/cccl.bashrc; \
    echo "export CUDACXX=${CUDACXX}" >> /etc/cccl.bashrc; \
    echo "source /etc/cccl.bashrc" >> /etc/bash.bashrc; \
    echo "ALL ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers; \
    pip install lit; \
    curl --silent --show-error -L ${CMAKE_URL} -o cmake.bash; \
    bash cmake.bash -- --skip-license --prefix=/usr; \
    curl --silent --show-error -L ${TINI_URL} -o /usr/bin/tini; \
    chmod +x /usr/bin/tini;

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]
