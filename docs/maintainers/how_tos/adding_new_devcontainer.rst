.. _infra-devcontainer-adding-toolchain:

Adding a new devcontainer toolchain
===================================

A toolchain is one CTK version paired with one host compiler. The ``ci`` branch
is the source of truth for the matrix, generator, and generated
devcontainer files. Work in sibling checkouts named ``cccl`` for source and
``cccl-ci`` for ``@ci``. CCCL generates a config for every combination listed
in the ``devcontainers:`` section of ``cccl-ci/ci/matrix.yaml``. The canonical
configs live under ``cccl-ci/.devcontainer/<name>/devcontainer.json``, one
directory per combination, and are produced by
``cccl-ci/.devcontainer/make_devcontainers.sh [--clean]``.

Run the steps in order on a branch based on ``ci``: edit the matrix, regenerate
the configs, then verify before merge. The base image for the combination must already exist in the
`rapidsai/devcontainers <https://github.com/rapidsai/devcontainers>`_ project before the matrix
edit.

Check that the base image exists
--------------------------------

Every CCCL devcontainer is built on a published ``rapidsai/devcontainers`` image. Image tags
follow this pattern::

    rapidsai/devcontainers:<devcontainer_version>-cpp-<compiler><version>-cuda<ctk>[ext]

The ``-cuda<ctk>`` segment is present for every combination except nvhpc, which bundles its
own CUDA toolkit and omits it.

The ``<devcontainer_version>`` value is the ``devcontainer_version:`` field in
``ci/matrix.yaml`` on the ``ci`` branch.

The images are maintained in the https://github.com/rapidsai/devcontainers/ repo, in the top-level
matrix file. If new images are required for the coverage, submit a PR against `main`.

Add the combination to @ci:ci/matrix.yaml
-----------------------------------------

The source of truth when generating devcontainer toolchains is
``ci/matrix.yaml`` in the ``cccl-ci`` checkout. All jobs from all workflows are
parsed, the toolchains extracted, and the canonical ``.devcontainer/...``
directories built.

At the bottom of the workflows section of ``matrix.yaml`` is a ``devcontainers:`` section.
This is intended to be a living mirror of the available images in the `rapidsai/devcontainers` repo,
and is useful for quickly checking supported CTK / host compilers while editing the matrix.
Occasionally we'll need a devcontainer that isn't referenced in any workflow, and this section is the place to add it.
Make sure that your new toolchain is listed and documented here.

Regenerate the devcontainer configs
-----------------------------------

From the ``cccl-ci`` root, regenerate every
``.devcontainer/<name>/devcontainer.json`` from the updated matrix:

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

Do not edit the promoted ``cccl/.devcontainer`` copy. After a change lands on
``ci``, the promotion workflow copies the canonical directory to ``main``, adds
``.devcontainer/.ci-source`` containing the exact ``ci`` commit SHA, and opens
or updates the promotion pull request. The marker lets Codespaces and local
devcontainers obtain the matching CI checkout at ``/home/coder/cccl-ci``.

Verify before merge
-------------------

Locally test launching the devcontainer using the appropriate ``.devcontainer/launch.sh`` invocation.
See :ref:`infra-devcontainer-launching` for details on launching and using the devcontainer.

The reusable ``verify-devcontainers`` workflow reruns
``make_devcontainers.sh --verbose --clean`` against the canonical ``@ci``
files and fails if the result differs. Source PR CI separately rejects any
``.devcontainer`` tree that differs from the ``@ci`` commit recorded in its
``.ci-source`` marker.
