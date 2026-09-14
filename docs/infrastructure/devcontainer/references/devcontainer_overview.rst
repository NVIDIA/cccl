.. _infra-devcontainer-overview:

Devcontainer overview
=====================

CCCL devcontainers package a CUDA toolkit and host compiler into a Docker image
that matches a CI environment. Reproducing a Linux CI result locally requires
no guesswork about toolchain versions: pick the combination, launch the
container, and the compiler, CTK, and supporting tools are identical to what CI
used.

Two uses: local development and CI
----------------------------------

**Local development.** Open the repository in VSCode and select a devcontainer
from the picker, or launch one directly with ``.devcontainer/launch.sh``.
:ref:`infra-devcontainer-launch-sh-reference` covers ``launch.sh`` flags and the available
combinations. ``launch.sh --docker`` runs the container without VSCode and drops
into a shell or runs a script. Without ``--docker``, it opens the container in
VSCode.

**CI.** Linux jobs run the same ``rapidsai/devcontainers`` images used for local
development. Windows jobs run the corresponding images from the internal
``ghcr.io/nvidia/cccl-windows-containers`` package.

Image sources
-------------

The image definitions and build workflow are maintained in the
`rapidsai/devcontainers <https://github.com/rapidsai/devcontainers>`_ repository.
Each image bundles a CUDA toolkit, a host compiler, and development tooling.
Linux and local-development images are published to Docker Hub. CCCL invokes
the reusable build workflow to publish Windows CI images to its internal GitHub
Container Registry package.

Linux image tags follow the pattern
``rapidsai/devcontainers:<version>-cpp-<compiler><version>-cuda<ctk>[ext]``.
The ``-cuda<ctk>`` segment is present for every combination except nvhpc, which
bundles its own CUDA toolkit; nvhpc images omit it.
Windows image tags follow the pattern
``ghcr.io/nvidia/cccl-windows-containers:<version>-cuda<ctk>-cl<version>``.
The ``<version>`` tag is defined in ``ci/matrix.yaml`` under
``devcontainer_version``. A generated ``.devcontainer/<combo>/devcontainer.json``
contains the resolved Linux tag for any given combination.

This upstream dependency gates toolchain changes. Adding a CUDA toolkit version
or a host compiler requires the matching image to exist in rapidsai/devcontainers
first. The sequence is:

#. Update rapidsai/devcontainers to build and publish images for the new
   combination.
#. Update CCCL's ``ci/matrix.yaml`` to reference the new version.
#. Regenerate CCCL's devcontainer configs to match all matrix workflow requirements.

:ref:`infra-devcontainer-adding-toolchain` has additional details on this process.
