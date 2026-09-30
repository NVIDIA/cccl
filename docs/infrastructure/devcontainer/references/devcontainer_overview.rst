.. _infra-devcontainer-overview:

Devcontainer overview
=====================

CCCL devcontainers package a CUDA toolkit and host compiler into a Docker image
that matches a CI environment exactly. The same image that builds and tests a
project in GitHub Actions runs on a developer's machine. Reproducing a CI
result locally requires no guesswork about toolchain versions: pick the
combination, launch the container, and the compiler, CTK, and supporting tools
are identical to what CI used.

Two uses: local development and CI
----------------------------------

**Local development.** Open the repository in VSCode and select a devcontainer
from the picker, or launch one directly with ``.devcontainer/launch.sh``.
:ref:`infra-devcontainer-launch-sh-reference` covers ``launch.sh`` flags and the available
combinations. ``launch.sh --docker`` runs the container without VSCode and drops
into a shell or runs a script. Without ``--docker``, it opens the container in
VSCode.

**CI.** GitHub Actions runs the identical images. A CI job for a given CTK and
compiler combination builds and tests inside the same ``rapidsai/devcontainers``
image a developer would launch locally.

Image source: rapidsai/devcontainers
------------------------------------

The base images are built and published by the
`rapidsai/devcontainers <https://github.com/rapidsai/devcontainers>`_ repository,
not by CCCL. Each image bundles a CUDA toolkit, a host compiler, and
development tooling. CCCL references these images by tag; it does not build them.

Image tags follow the pattern
``rapidsai/devcontainers:<version>-cpp-<compiler><version>-cuda<ctk>[ext]``.
The ``-cuda<ctk>`` segment is present for every combination except nvhpc, which
bundles its own CUDA toolkit; nvhpc images omit it.
The ``<version>`` tag is defined in ``ci/matrix.yaml`` under
``devcontainer_version``. A generated ``.devcontainer/<combo>/devcontainer.json``
contains the resolved tag for any given combination.

This upstream dependency gates toolchain changes. Adding a CUDA toolkit version
or a host compiler requires the matching image to exist in rapidsai/devcontainers
first. The sequence is:

#. Update rapidsai/devcontainers to build and publish images for the new
   combination.
#. Update CCCL's ``ci/matrix.yaml`` to reference the new version.
#. Regenerate CCCL's devcontainer configs to match all matrix workflow requirements.

:ref:`infra-devcontainer-adding-toolchain` has additional details on this process.
