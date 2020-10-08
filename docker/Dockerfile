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

ENV TZ=US/Pacific
ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

# Update and add packages for Ubuntu.
RUN if [[ "${OS_TYPE}" == "ubuntu"* ]]; then \
      source /etc/os-release; \
      echo "deb http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu ${UBUNTU_CODENAME} main" >> /etc/apt/sources.list; \
      APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1 apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 60C317803A41BA51845E371A1E9377A2BA9EF27F; \
      apt-get -y update; \
      apt-get -y --no-install-recommends install apt-utils; \
      apt-get -y --no-install-recommends install wget ca-certificates; \
      apt-get -y --no-install-recommends install python3-pip python3-setuptools python3-wheel; \
      apt-get -y --no-install-recommends install cmake; \
      if   [[ "${CXX_TYPE}" == "gcc" ]]; then \
        apt-get -y --no-install-recommends install g++-${CXX_VER}; \
        update-alternatives --install /usr/bin/cc gcc /usr/bin/gcc-${CXX_VER} ${CXX_VER}; \
        update-alternatives --install /usr/bin/c++ g++ /usr/bin/g++-${CXX_VER} ${CXX_VER}; \
        update-alternatives --set cc /usr/bin/gcc-${CXX_VER}; \
        update-alternatives --set c++ /usr/bin/g++-${CXX_VER}; \
        update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-${CXX_VER} ${CXX_VER}; \
        update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-${CXX_VER} ${CXX_VER}; \
        update-alternatives --set gcc /usr/bin/gcc-${CXX_VER}; \
        update-alternatives --set g++ /usr/bin/g++-${CXX_VER}; \
      elif [[ "${CXX_TYPE}" == "clang" ]]; then \
        apt-get -y --no-install-recommends install clang-${CXX_VER}; \
        update-alternatives --install /usr/bin/cc clang /usr/bin/clang-${CXX_VER} ${CXX_VER}; \
        update-alternatives --install /usr/bin/c++ clang++ /usr/bin/clang++-${CXX_VER} ${CXX_VER}; \
        update-alternatives --set cc /usr/bin/clang-${CXX_VER}; \
        update-alternatives --set c++ /usr/bin/clang++-${CXX_VER}; \
        update-alternatives --install /usr/bin/clang clang /usr/bin/clang-${CXX_VER} ${CXX_VER}; \
        update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-${CXX_VER} ${CXX_VER}; \
        update-alternatives --set clang /usr/bin/clang-${CXX_VER}; \
        update-alternatives --set clang++ /usr/bin/clang++-${CXX_VER}; \
      fi; \
      apt-get clean; \
      rm -rf /var/lib/apt/lists/*; \
      python3 -m pip install lit; \
    else \
      echo -e "\n\n>>>> SKIPPING: OS_VER is not 'ubuntu*'\n\n"; \
    fi

# Install tini for init.
RUN wget --quiet ${TINI_URL} -O /usr/bin/tini && chmod +x /usr/bin/tini

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]
