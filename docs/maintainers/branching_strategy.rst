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

Tagging conventions
-------------------
- ``vX.Y.Z`` finalized release tags on release branches
- ``vX.Y.Z-rcN``: release-candidate tags for pre-release validation.
- ``vX.Y.Z.dev``: the first commit of development for ``X.Y.Z`` (the commit that increments the library version)
