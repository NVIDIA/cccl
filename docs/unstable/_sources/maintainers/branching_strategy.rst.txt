Git Methodology
=========================

This page defines the canonical branch model used for CCCL development and
release maintenance.

Canonical branches
------------------

- ``main``

  - The default development branch.
  - Updates should be made via pull requests following our :doc:`contributing guidelines </cccl/contributing>`.

- ``branch/X.Y.x``

  - Branches from ``main`` meant for stabilizing and publishing tagged releases.
  - Created via release automation.
  - Changes should be made via the :doc:`backport process <backport_process>`.

- ``ctk/X.Y.x``

  - Branches from ``branch/X.Y.x`` meant for representing exact contents of CUDA Toolkit Releases
  - Created by NVIDIA. May contain internally developed features that are additive with the release it was based on.
  - This is a read-only branch and is created when the CTK or relevant documentation becomes available.

Tagging conventions
-------------------
- ``vX.Y.Z``: finalized release tags on release branches
- ``vX.Y.Z-ctkN.M.K``: finalized release tags exactly equivalent to the CCCL contents in NVIDIA CUDA Toolkit release `N.M.K`.
- ``vX.Y.Z-rcN``: release-candidate tags for pre-release validation.
- ``vX.Y.Z.dev``: the first commit of development for ``X.Y.Z`` (the commit that increments the library version)
