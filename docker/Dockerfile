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
# | CXX_TYPE | CXX_VER         |
# | gcc      | 5 6 7 8 9 10 11 |
# | clang    | 7 8 9 10 11     |
# | icc      | latest          |
# | nvcxx    | 21.5            |
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

# Ubuntu 20.04 doesn't have GCC 11 in its repos, so get it from the toolchain PPA.
ARG UBUNTU_TOOL_DEB_REPO=http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu
ARG UBUNTU_TOOL_FINGER=60C317803A41BA51845E371A1E9377A2BA9EF27F

# Ubuntu 20.04 doesn't have GCC 5 and GCC 6, so get it from an older release.
ARG UBUNTU_ARCHIVE_DEB_REPO="http://archive.ubuntu.com/ubuntu bionic main universe"

ARG ICC_DEB_REPO="https://apt.repos.intel.com/oneapi all main"
ARG ICC_KEY=https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB

# CentOS 7 doesn't have a new enough version of CMake in its repos.
ARG CMAKE_VER=3.18.4
ARG CMAKE_URL=https://github.com/Kitware/CMake/releases/download/v${CMAKE_VER}/cmake-${CMAKE_VER}-Linux-x86_64.sh

ARG TINI_VER=0.18.0
ARG TINI_URL=https://github.com/krallin/tini/releases/download/v${TINI_VER}/tini

# `--silent --show-error` disables non-error output.
# `--fail` causes `curl` to return an error code on HTTP errors.
ARG CURL="curl --silent --show-error --fail"

# `-y` answers yes to any interactive prompts.
# `--no-install-recommends` avoids unnecessary packages, keeping the image smaller.
ARG APT_GET="apt-get -y --no-install-recommends"

# `-y` answers yes to any interactive prompts.
ARG YUM="yum -y"

ENV TZ=US/Pacific
ENV DEBIAN_FRONTEND=noninteractive
# apt-key complains about non-interactive usage.
ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

SHELL ["/usr/bin/env", "bash", "-c"]

RUN function comment() { :; }; \
    if [[ "${OS_TYPE}" == "ubuntu"* ]]; then \
      export ALTERNATIVES=update-alternatives; \
      ${APT_GET} update; \
      ${APT_GET} install apt-utils \
        2> >(grep -v 'debconf: delaying package configuration, since apt-utils is not installed' >&2); \
      ${APT_GET} install curl; \
      ${APT_GET} install gnupg; \
      if   [[ "${CXX_TYPE}" == "gcc" && "${CXX_VER}" -le 6 ]]; then \
        echo "deb ${UBUNTU_ARCHIVE_DEB_REPO}" >> /etc/apt/sources.list.d/ubuntu-archive.list; \
      elif [[ "${CXX_TYPE}" == "gcc" && "${CXX_VER}" -ge 6 ]]; then \
        source /etc/os-release; \
        echo "deb ${UBUNTU_TOOL_DEB_REPO} ${UBUNTU_CODENAME} main" >> /etc/apt/sources.list; \
        apt-key adv --keyserver keyserver.ubuntu.com --recv-keys ${UBUNTU_TOOL_FINGER} 2>&1; \
      elif [[ "${CXX_TYPE}" == "icc" ]]; then \
        echo "deb ${ICC_DEB_REPO}" > /etc/apt/sources.list.d/icc.list; \
        ${CURL} -L ${ICC_KEY} -o - | apt-key add -; \
      fi; \
      ${APT_GET} update; \
      comment "Build Tools"; \
      ${APT_GET} install python3-pip python3-setuptools python3-wheel; \
      ${ALTERNATIVES} --install /usr/bin/python python $(which python3) 3; \
      ${ALTERNATIVES} --set python $(which python3); \
      ${APT_GET} install zip unzip tar; \
      ${APT_GET} install sudo; \
      ${APT_GET} install openssh-client; \
      ${APT_GET} install llvm-dev; \
      ${APT_GET} install libtbb-dev; \
      ${APT_GET} install libomp-dev; \
      ${APT_GET} install make ninja-build; \
      ${APT_GET} install pkg-config; \
      comment "Developer Tools"; \
      ${APT_GET} install gdb; \
      ${APT_GET} install strace; \
      ${APT_GET} install less; \
      ${APT_GET} install git; \
      ${APT_GET} install vim emacs-nox; \
      comment "Documentation Tools."; \
      ${APT_GET} install bundler ruby-dev; \
      ${APT_GET} install doxygen; \
      comment "Compilers."; \
      ${APT_GET} install g++-9; \
      if   [[ "${CXX_TYPE}" == "gcc" ]]; then \
        ${APT_GET} install g++-${CXX_VER} gcc-${CXX_VER}; \
        export CC=$(which gcc-${CXX_VER}); \
        export CXX=$(which g++-${CXX_VER}); \
      elif [[ "${CXX_TYPE}" == "clang" ]]; then \
        ${APT_GET} install clang-${CXX_VER}; \
        export CC=$(which clang-${CXX_VER}); \
        export CXX=$(which clang++-${CXX_VER}); \
      elif [[ "${CXX_TYPE}" == "icc" ]]; then \
        ${APT_GET} install intel-oneapi-compiler-dpcpp-cpp-and-cpp-classic g++; \
        source /opt/intel/oneapi/setvars.sh; \
        export CC=$(which icc); \
        export CXX=$(which icpc); \
      elif [[ "${CXX_TYPE}" == "nvcxx" ]]; then \
        export CC=$(which nvc); \
        export CXX=$(which nvc++); \
      fi; \
      ${APT_GET} clean; \
      rm -rf /var/lib/apt/lists/*; \
      echo "source /etc/cccl.bashrc" >> /etc/bash.bashrc; \
    elif [[ "${OS_TYPE}" == "centos" ]]; then \
      export ALTERNATIVES=alternatives; \
      ${YUM} --enablerepo=extras install epel-release; \
      ${YUM} updateinfo; \
      ${YUM} install centos-release-scl; \
      comment "Build Tools"; \
      ${YUM} install python python-pip; \
      ${YUM} install zip unzip tar; \
      ${YUM} install sudo; \
      ${YUM} install openssh-clients; \
      ${YUM} install pkgconfig; \
      ${YUM} install make ninja-build; \
      ${YUM} install llvm-devel; \
      ${YUM} install tbb-devel; \
      comment "Developer Tools"; \
      ${YUM} install which; \
      ${YUM} install gdb; \
      ${YUM} install strace; \
      ${YUM} install less; \
      ${YUM} install git; \
      ${YUM} install vim emacs-nox; \
      comment "Compilers."; \
      if   [[ "${CXX_TYPE}" == "gcc" ]]; then \
        ${YUM} install devtoolset-${CXX_VER}-gcc*; \
        source scl_source enable devtoolset-${CXX_VER}; \
        echo "source scl_source enable devtoolset-${CXX_VER}" >> /etc/cccl.bashrc; \
        source /etc/cccl.bashrc; \
        export CC=$(which gcc); \
        export CXX=$(which g++); \
      elif [[ "${CXX_TYPE}" == "nvcxx" ]]; then \
        ${YUM} install devtoolset-7-gcc*; \
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
    echo "export PATH=${PATH}"            >> /etc/cccl.bashrc; \
    echo "export LD_LIBRARY_PATH=${PATH}" >> /etc/cccl.bashrc; \
    echo "export CC=${CC}"                >> /etc/cccl.bashrc; \
    echo "export CXX=${CXX}"              >> /etc/cccl.bashrc; \
    echo "export CUDACXX=${CUDACXX}"      >> /etc/cccl.bashrc; \
    echo "export SDK_TYPE=${SDK_TYPE}"    >> /etc/cccl.bashrc; \
    echo "export SDK_VER=${SDK_VER}"      >> /etc/cccl.bashrc; \
    echo "export OS_TYPE=${OS_TYPE}"      >> /etc/cccl.bashrc; \
    echo "export OS_VER=${OS_VER}"        >> /etc/cccl.bashrc; \
    echo "export CXX_TYPE=${CXX_TYPE}"    >> /etc/cccl.bashrc; \
    echo "export CXX_VER=${CXX_VER}"      >> /etc/cccl.bashrc; \
    if [[ "${CXX_TYPE}" == "icc" ]]; then \
      comment "ICC's setup script clobbers bash arguments, so we must"; \
      comment "save them before calling it and restore them afterwards."; \
      echo "bash_args=(\"\$@\")"                            >> /etc/cccl.bashrc; \
      echo "source /opt/intel/oneapi/setvars.sh --force"    >> /etc/cccl.bashrc; \
      echo "export CMAKE_LIBRARY_PATH=\"\${LIBRARY_PATH}\"" >> /etc/cccl.bashrc; \
      echo "set -- \"\${bash_args[@]}\""                    >> /etc/cccl.bashrc; \
    fi; \
    echo "ALL ALL=(ALL) NOPASSWD:ALL"     >> /etc/sudoers; \
    pip install lit; \
    ${CURL} -L ${CMAKE_URL} -o cmake.bash; \
    bash cmake.bash -- --skip-license --prefix=/usr; \
    rm cmake.bash; \
    comment "Build Doxybook2."; \
    if [[ "${OS_TYPE}" == "ubuntu"* ]]; then \
      git clone https://github.com/microsoft/vcpkg /tmp/vcpkg 2>&1; \
      /tmp/vcpkg/bootstrap-vcpkg.sh -disableMetrics -useSystemBinaries; \
      git clone https://github.com/pantor/inja.git /tmp/inja 2>&1; \
      mkdir /tmp/inja/build; \
      cmake -B /tmp/inja/build \
            -DCMAKE_CXX_COMPILER=g++-9 \
            -DCMAKE_BUILD_TYPE=MinSizeRel \
            /tmp/inja; \
      cmake --build /tmp/inja/build --config MinSizeRel; \
      git clone https://github.com/matusnovak/doxybook2.git /tmp/doxybook2 2>&1; \
      comment "--no-binarycaching avoids zipping up and storing stuff we're "; \
      comment "just going to delete later."; \
      /tmp/vcpkg/vcpkg --no-binarycaching install argagg catch2 fmt dirent fmt nlohmann-json rang; \
      mkdir /tmp/doxybook2/build; \
      cmake -B /tmp/doxybook2/build \
            -DCMAKE_CXX_COMPILER=g++-9 \
            -DCMAKE_BUILD_TYPE=MinSizeRel \
            -DCMAKE_PREFIX_PATH=/tmp/inja/build \
            -DCMAKE_TOOLCHAIN_FILE=/tmp/vcpkg/scripts/buildsystems/vcpkg.cmake \
            /tmp/doxybook2; \
      cmake --build /tmp/doxybook2/build --config MinSizeRel; \
      cmake --install /tmp/doxybook2/build --config MinSizeRel; \
      rm -rf /tmp/vcpkg; \
      rm -rf ~/.vcpkg; \
      rm -rf /tmp/inja; \
      rm -rf /tmp/doxybook2; \
    fi; \
    ${CURL} -L ${TINI_URL} -o /usr/bin/tini; \
    chmod +x /usr/bin/tini; \
    ${CXX} --version

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]
