Git Branches
==================

This page documents which branches are used for active development and releases.

Branches
--------
- `main` - the default development branch
  - Updates to `main` are made via pull requests following our [contributing guidelines](../contributing.rst).
- `branch/X.Y.x` - release branches created from `main` when finalizing a new release
  - Updates to release branches should be made via our [backport process](backport_process.rst).
  - Release tags, `vX.Y.Z`, are created from these branches to mark finalized releases.

Other branches may be used for various purposes, but the above are the only ones used for well-defined maintenance processes.
