.. _infra-devcontainer-adding-toolchain:

Adding a new devcontainer toolchain
===================================

A toolchain is one CTK version paired with one host compiler. CCCL generates a
devcontainer config for every combination listed in the ``devcontainers:`` section of
``ci/matrix.yaml``. The generated configs live under ``.devcontainer/<name>/devcontainer.json``,
one directory per combination, all produced by ``.devcontainer/make_devcontainers.sh [--clean]``.

Run the steps in order: edit the matrix, regenerate the configs, then verify before merge. The
base image for the combination must already exist in the
`rapidsai/devcontainers <https://github.com/rapidsai/devcontainers>`_ project before the matrix
edit.

Check that the base image exists
--------------------------------

Every CCCL devcontainer is built on a published ``rapidsai/devcontainers`` image. Image tags
follow this pattern::

    rapidsai/devcontainers:<devcontainer_version>-cpp-<compiler><version>-cuda<ctk>[ext]

The ``-cuda<ctk>`` segment is present for every combination except nvhpc, which bundles its
own CUDA toolkit and omits it.

The ``<devcontainer_version>`` value is the ``devcontainer_version:`` field in ``ci/matrix.yaml``.

The images are maintained in the https://github.com/rapidsai/devcontainers/ repo, in the top-level
matrix file. If new images are required for the coverage, submit a PR against `main`.

Add the combination to ci/matrix.yaml
-------------------------------------

The source-of-truth when generating devcontainer toolchains is the ``matrix.yaml`` file. All jobs
from all workflows are parsed, the toolchains extracted, and the ``.devcontainer/...`` directories built.

At the bottom of the workflows section of ``matrix.yaml`` is a ``devcontainers:`` section.
This is intended to be a living mirror of the available images in the `rapidsai/devcontainers` repo,
and is useful for quickly checking supported CTK / host compilers while editing the matrix.
Occasionally we'll need a devcontainer that isn't referenced in any workflow, and this section is the place to add it.
Make sure that your new toolchain is listed and documented here.

Regenerate the devcontainer configs
-----------------------------------

From the repository root, regenerate every ``.devcontainer/<name>/devcontainer.json`` from the updated matrix:

.. code-block:: bash

    .devcontainer/make_devcontainers.sh --clean

The script reads all matrix workflow entries, expands aliases, and writes one directory per combination
using the naming pattern ``cuda<version>[ext]-<compiler><version>``.
It also updates the root ``.devcontainer/devcontainer.json`` default to the newest GCC + newest
CUDA combination.
Pass ``--clean`` to remove directories for combinations no longer in the matrix (recommended).

Never hand-edit a generated ``.devcontainer/<name>/devcontainer.json``. Edits are overwritten on
the next run. To change settings that apply to every combination, edit the root
``.devcontainer/devcontainer.json`` template, then rerun the generator to propagate the change.

Verify before merge
-------------------

Locally test launching the devcontainer using the appropriate ``.devcontainer/launch.sh`` invocation.
See :ref:`infra-devcontainer-launching` for details on launching and using the devcontainer.

The ``verify-devcontainers`` CI workflow reruns ``make_devcontainers.sh --verbose --clean``
and fails if the result differs from the committed files.
