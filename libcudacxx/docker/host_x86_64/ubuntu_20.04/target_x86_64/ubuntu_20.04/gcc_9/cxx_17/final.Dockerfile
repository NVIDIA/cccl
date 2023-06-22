# Dockerfile for libcudacxx:host_x86_64_ubuntu_20.04__target_x86_64_ubuntu_20.04__gcc_9_cxx_17

FROM libcudacxx_base:host_x86_64_ubuntu_20.04__target_x86_64_ubuntu_20.04__gcc_9_cxx_17

MAINTAINER Bryce Adelstein Lelbach <blelbach@nvidia.com>

###############################################################################
# BUILD: The following is invoked when the CUDA driver is attached.

# We use ADD here because it invalidates the cache for subsequent steps, which
# is what we want, as we need to rebuild if the sources have changed.

ADD nvidia-modprobe /usr/bin/
# Restore setuid.
RUN chmod +s /usr/bin/nvidia-modprobe

ADD libcuda.so* /usr/lib/x86_64-linux-gnu/
ADD libnvidia-ptxjitcompiler.so* /usr/lib/x86_64-linux-gnu/

###############################################################################
# CMD: The following is invoked when the image is run.

RUN useradd -ms /bin/bash libcudacxx && chown -R libcudacxx:libcudacxx /sw
USER libcudacxx

WORKDIR /sw/gpgpu/libcudacxx

CMD /sw/gpgpu/libcudacxx/utils/nvidia/linux/perform_tests.bash

