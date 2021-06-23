# Copyright (c) 2018-2020 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Released under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.

# SDK_TYPE needs to be a base image that contains CUDA.
# | SDK_TYPE | SDK_VER             |
# | cuda     | 11.3.1-devel        |
# | nvhpc    | 21.5-devel-cuda11.3 |
ARG SDK_TYPE=cuda
ARG SDK_VER=11.3.1-devel
# | OS_TYPE  | OS_VER |
# | ubuntu   | 20.04  |
# | centos   | 7      |
ARG OS_TYPE=ubuntu
ARG OS_VER=20.04
# | CXX_TYPE | CXX_VER      |
# | gcc      | 5 6 7 8 9 10 |
# | clang    | 7 8 9 10     |
# | icc      | latest       |
# | nvcxx    | 21.5         |
ARG CXX_TYPE=gcc
ARG CXX_VER=7
FROM nvcr.io/nvidia/${SDK_TYPE}:${SDK_VER}-${OS_TYPE}${OS_VER}

# Pull arguments from build arguments.
ARG SDK_TYPE
ARG SDK_VER
ARG OS_TYPE
ARG OS_VER
ARG CXX_TYPE
ARG CXX_VER

# Ubuntu 20.04 doesn't have GCC 5 and GCC 6, so get it from an older release.
ARG UBUNTU_ARCHIVE_DEB_REPO="http://archive.ubuntu.com/ubuntu bionic main universe"

ARG ICC_DEB_REPO="https://apt.repos.intel.com/oneapi all main"
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
      apt-get -y --no-install-recommends install apt-utils \
        2> >(grep -v 'debconf: delaying package configuration, since apt-utils is not installed' >&2); \
      apt-get -y --no-install-recommends install curl; \
      apt-get -y --no-install-recommends install gnupg; \
      if   [[ "${CXX_TYPE}" == "gcc" && "${CXX_VER}" -le 6 ]]; then \
        echo "deb ${UBUNTU_ARCHIVE_DEB_REPO}" >> /etc/apt/sources.list.d/ubuntu-archive.list; \
      elif [[ "${CXX_TYPE}" == "icc" ]]; then \
        echo "deb ${ICC_DEB_REPO}" > /etc/apt/sources.list.d/icc.list; \
        curl --silent --show-error -L ${ICC_KEY} -o - | apt-key add -; \
      fi; \
      apt-get -y update; \
      apt-get -y --no-install-recommends install python3-pip python3-setuptools python3-wheel; \
      ${ALTERNATIVES} --install /usr/bin/python python $(which python3) 3; \
      ${ALTERNATIVES} --set python $(which python3); \
      apt-get -y --no-install-recommends install sudo; \
      apt-get -y --no-install-recommends install make ninja-build; \
      apt-get -y --no-install-recommends install llvm-dev; \
      apt-get -y --no-install-recommends install libtbb-dev; \
      apt-get -y --no-install-recommends install libomp-dev; \
      apt-get -y --no-install-recommends install gdb; \
      apt-get -y --no-install-recommends install strace; \
      apt-get -y --no-install-recommends install less; \
      apt-get -y --no-install-recommends install git; \
      apt-get -y --no-install-recommends install vim emacs-nox; \
      if   [[ "${CXX_TYPE}" == "gcc" ]]; then \
        apt-get -y --no-install-recommends install g++-${CXX_VER}; \
        export CC=$(which gcc-${CXX_VER}); \
        export CXX=$(which g++-${CXX_VER}); \
      elif [[ "${CXX_TYPE}" == "clang" ]]; then \
        apt-get -y --no-install-recommends install clang-${CXX_VER}; \
        export CC=$(which clang-${CXX_VER}); \
        export CXX=$(which clang++-${CXX_VER}); \
      elif [[ "${CXX_TYPE}" == "icc" ]]; then \
        apt-get -y --no-install-recommends install intel-oneapi-compiler-dpcpp-cpp-and-cpp-classic g++; \
        source /opt/intel/oneapi/setvars.sh; \
        echo "bash_args=(\"\$@\")"                            >> /etc/cccl.bashrc; \
        echo "source /opt/intel/oneapi/setvars.sh"            >> /etc/cccl.bashrc; \
        echo "export CMAKE_LIBRARY_PATH=\"\${LIBRARY_PATH}\"" >> /etc/cccl.bashrc; \
        echo "set -- \"\${bash_args[@]}\""                    >> /etc/cccl.bashrc; \
        export CC=$(which icc); \
        export CXX=$(which icpc); \
      elif [[ "${CXX_TYPE}" == "nvcxx" ]]; then \
        export CC=$(which nvc); \
        export CXX=$(which nvc++); \
      fi; \
      apt-get clean; \
      rm -rf /var/lib/apt/lists/*; \
      echo "source /etc/cccl.bashrc" >> /etc/bash.bashrc; \
    elif [[ "${OS_TYPE}" == "centos" ]]; then \
      export ALTERNATIVES=alternatives; \
      yum -y --enablerepo=extras install epel-release; \
      yum -y updateinfo; \
      yum -y install centos-release-scl; \
      yum -y install which; \
      yum -y install python python-pip; \
      yum -y install sudo; \
      yum -y install make ninja-build; \
      yum -y install llvm-devel; \
      yum -y install tbb-devel; \
      yum -y install gdb; \
      yum -y install strace; \
      yum -y install less; \
      yum -y install git; \
      yum -y install vim emacs-nox; \
      if   [[ "${CXX_TYPE}" == "gcc" ]]; then \
        yum -y install devtoolset-${CXX_VER}-gcc*; \
        source scl_source enable devtoolset-${CXX_VER}; \
        echo "source scl_source enable devtoolset-${CXX_VER}" >> /etc/cccl.bashrc; \
        source /etc/cccl.bashrc; \
        export CC=$(which gcc); \
        export CXX=$(which g++); \
      elif [[ "${CXX_TYPE}" == "nvcxx" ]]; then \
        yum -y install devtoolset-7-gcc*; \
        echo "source scl_source enable devtoolset-7" >> /etc/cccl.bashrc; \
        source /etc/cccl.bashrc; \
        export CC=$(which nvc); \
        export CXX=$(which nvc++); \
      else \
        echo -e "\n\n>>>> ERROR: ${CXX_TYPE} is not supported on ${OS_TYPE}.\n\n"; \
        exit 1; \
      fi; \
      rm -f /usr/bin/cc; \
      rm -f /usr/bin/c++; \
      echo "source /etc/cccl.bashrc" >> /etc/bashrc; \
    fi; \
    ${ALTERNATIVES} --install /usr/bin/cc  cc  ${CC}  99; \
    ${ALTERNATIVES} --install /usr/bin/c++ c++ ${CXX} 99; \
    ${ALTERNATIVES} --set cc  ${CC}; \
    ${ALTERNATIVES} --set c++ ${CXX}; \
    if [[ "${CXX_TYPE}" == "nvcxx" ]]; then \
      export CUDACXX=$(which nvc++); \
    else \
      export CUDACXX=$(which nvcc); \
    fi; \
    echo "export CC=${CC}"             >> /etc/cccl.bashrc; \
    echo "export CXX=${CXX}"           >> /etc/cccl.bashrc; \
    echo "export CUDACXX=${CUDACXX}"   >> /etc/cccl.bashrc; \
    echo "export SDK_TYPE=${SDK_TYPE}" >> /etc/cccl.bashrc; \
    echo "export SDK_VER=${SDK_VER}"   >> /etc/cccl.bashrc; \
    echo "export OS_TYPE=${OS_TYPE}"   >> /etc/cccl.bashrc; \
    echo "export OS_VER=${OS_VER}"     >> /etc/cccl.bashrc; \
    echo "export CXX_TYPE=${CXX_TYPE}" >> /etc/cccl.bashrc; \
    echo "export CXX_VER=${CXX_VER}"   >> /etc/cccl.bashrc; \
    echo "ALL ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers; \
    pip install lit; \
    curl --silent --show-error -L ${CMAKE_URL} -o cmake.bash; \
    bash cmake.bash -- --skip-license --prefix=/usr; \
    rm cmake.bash; \
    curl --silent --show-error -L ${TINI_URL} -o /usr/bin/tini; \
    chmod +x /usr/bin/tini;

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]
