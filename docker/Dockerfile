# FROM_IMAGE needs to be a base image that contains CUDA, either:
# - nvidia/cuda
# - nvcr.io/nvidia/nvhpc
ARG NVIDIA_IMAGE=nvidia/cuda
ARG NVIDIA_VER=11.0
ARG IMAGE_TYPE=devel
ARG OS_VER=ubuntu18.04
FROM ${NVIDIA_IMAGE}:${NVIDIA_VER}-${IMAGE_TYPE}-${OS_VER}

# Pull arguments from build arguments.
ARG NVIDIA_IMAGE
ARG NVIDIA_VER
ARG OS_VER

ARG TINI_VER=v0.18.0
ARG TINI_URL=https://github.com/krallin/tini/releases/download/${TINI_VER}/tini

ENV TZ=US/Pacific
ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

# Update and add packages for Ubuntu.
RUN if [[ "${OS_VER}" == "ubuntu"* ]]; then \
      source /etc/os-release; \
      echo "deb http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu ${UBUNTU_CODENAME} main" >> /etc/apt/sources.list; \
      APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1 apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 60C317803A41BA51845E371A1E9377A2BA9EF27F; \
      apt-get -y update; \
      apt-get -y --no-install-recommends install apt-utils; \
      apt-get -y --no-install-recommends install wget ca-certificates; \
      apt-get -y --no-install-recommends install g++-5 g++-6 g++-7 g++-8 g++-9 g++-10; \
      apt-get -y --no-install-recommends install clang-7 clang-8 clang-9 clang-10; \
      apt-get -y --no-install-recommends install python3-pip python3-setuptools python3-wheel; \
      apt-get -y --no-install-recommends install cmake; \
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
