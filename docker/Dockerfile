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
# | icc      |              |
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

ARG TINI_VER=v0.18.0
ARG TINI_URL=https://github.com/krallin/tini/releases/download/${TINI_VER}/tini
ARG CMAKE_VER=3.18.4
ARG CMAKE_URL=https://github.com/Kitware/CMake/releases/download/v${CMAKE_VER}/cmake-${CMAKE_VER}-Linux-x86_64.sh

ENV TZ=US/Pacific
ENV DEBIAN_FRONTEND=noninteractive
# apt-key complains about non-interactive usage.
ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

SHELL ["/bin/bash", "-c"]

# Add Debian repositories and fetch package data for Ubuntu.
RUN if [[ "${OS_TYPE}" == "ubuntu"* ]]; then \
      apt-get -y update; \
      apt-get -y --no-install-recommends install apt-utils; \
      apt-get -y --no-install-recommends install wget ca-certificates; \
      if   [[ "${CXX_TYPE}" == "gcc" && "${CXX_VER}" > 8 ]]; then \
        source /etc/os-release; \
        echo "deb http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu ${UBUNTU_CODENAME} main" >> /etc/apt/sources.list; \
        apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 60C317803A41BA51845E371A1E9377A2BA9EF27F; \
      elif [[ "${CXX_TYPE}" == "icc" ]]; then \
        echo "deb https://apt.repos.intel.com/oneapi all main" >> /etc/apt/sources.list; \
        wget -q https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB -O- | apt-key add -; \
      fi; \
      apt-get -y update; \
    else \
      echo -e "\n\n>>>> SKIPPING: $${OS_VER} is not \"ubuntu\"\n\n"; \
    fi

# Install cmake
RUN wget ${CMAKE_URL} -O cmake.sh && \
      chmod +x cmake.sh && \
      ./cmake.sh --skip-license && \
      rm cmake.sh

# Install Debian packages for Ubuntu.
RUN if [[ "${OS_TYPE}" == "ubuntu"* ]]; then \
      apt-get -y --no-install-recommends install python3-pip python3-setuptools python3-wheel; \
      if   [[ "${CXX_TYPE}" == "gcc" ]]; then \
        apt-get -y --no-install-recommends install g++-${CXX_VER}; \
        update-alternatives --install /usr/bin/cc  cc  $(which gcc-${CXX_VER}) ${CXX_VER}; \
        update-alternatives --install /usr/bin/c++ c++ $(which g++-${CXX_VER}) ${CXX_VER}; \
        update-alternatives --set cc  $(which gcc-${CXX_VER}); \
        update-alternatives --set c++ $(which g++-${CXX_VER}); \
        update-alternatives --install /usr/bin/gcc gcc $(which gcc-${CXX_VER}) ${CXX_VER}; \
        update-alternatives --install /usr/bin/g++ g++ $(which g++-${CXX_VER}) ${CXX_VER}; \
        update-alternatives --set gcc $(which gcc-${CXX_VER}); \
        update-alternatives --set g++ $(which g++-${CXX_VER}); \
      elif [[ "${CXX_TYPE}" == "clang" ]]; then \
        apt-get -y --no-install-recommends install clang-${CXX_VER}; \
        update-alternatives --install /usr/bin/cc  cc  $(which clang-${CXX_VER})   ${CXX_VER}; \
        update-alternatives --install /usr/bin/c++ c++ $(which clang++-${CXX_VER}) ${CXX_VER}; \
        update-alternatives --set cc  $(which clang-${CXX_VER}); \
        update-alternatives --set c++ $(which clang++-${CXX_VER}); \
        update-alternatives --install /usr/bin/clang   clang   $(which clang-${CXX_VER})   ${CXX_VER}; \
        update-alternatives --install /usr/bin/clang++ clang++ $(which clang++-${CXX_VER}) ${CXX_VER}; \
        update-alternatives --set clang   $(which clang-${CXX_VER}); \
        update-alternatives --set clang++ $(which clang++-${CXX_VER}); \
      elif [[ "${CXX_TYPE}" == "icc" ]]; then \
        apt-get -y --no-install-recommends install intel-oneapi-icc g++; \
        source /opt/intel/oneapi/setvars.sh; \
        update-alternatives --install /usr/bin/cc  cc  $(which icc)  99; \
        update-alternatives --install /usr/bin/c++ c++ $(which icpc) 99; \
        update-alternatives --set cc  $(which icc); \
        update-alternatives --set c++ $(which icpc); \
        update-alternatives --install /usr/bin/icc  icc  $(which icc)  99; \
        update-alternatives --install /usr/bin/icpc icpc $(which icpc) 99; \
        update-alternatives --set icc  $(which icc); \
        update-alternatives --set icpc $(which icpc); \
      fi; \
      if   [[ "${SDK_TYPE}" == "nvidia/cuda" ]]; then \
        update-alternatives --install /usr/bin/cu++ nvcc $(which nvcc) 99; \
      elif [[ "${SDK_TYPE}" == "nvcr.io/nvidia/nvhpc" ]]; then \
        update-alternatives --install /usr/bin/cu++ nvc++ $(which nvc++) 99; \
      fi; \
      apt-get clean; \
      rm -rf /var/lib/apt/lists/*; \
    else \
      echo -e "\n\n>>>> SKIPPING: $${OS_VER} is not \"ubuntu\"\n\n"; \
    fi

# Install lit.
RUN python3 -m pip install lit

# Install tini for init.
RUN wget --quiet ${TINI_URL} -O /usr/bin/tini && chmod +x /usr/bin/tini

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]
