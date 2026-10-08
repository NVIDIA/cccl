.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-configuration:

Configuration
=============

Use these settings to choose headers, control compiler caches, or inspect
generated code. Build-time settings apply when packaging ``cuda-coop``.

Runtime environment variables
-----------------------------

For the Boolean switches below, *truthy* means any value other than the
empty string, ``0``, ``false``, ``no``, or ``off``. For example, ``1``,
``true``, ``yes``, and ``on`` are all truthy. Values are case-insensitive,
and leading and trailing whitespace is ignored. An unset variable is false.

``CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION``
   A truthy value disables automatic backend activation during
   :mod:`cuda.coop` import. Explicit ``coop.register(...)`` and
   backend imports still work.

``CUDA_COOP_CCCL_ROOT``
   Selects a CCCL source checkout or a ``cuda-coop`` header bundle. An invalid
   configured root is an error; resolution does not fall back to another CCCL
   source.

``CUDA_COOP_ENABLE_CACHE``
   A truthy value enables the Numba-CUDA-MLIR persistent compiler cache.
   Read when the Numba backend cache module is imported.

``XDG_CACHE_HOME``
   For the Numba backend on Linux and other POSIX systems, sets the cache
   base directory; entries are stored in ``<value>/cccl``. Unset, empty, or
   relative values fall back to ``~/.cache/cccl``. Read when the backend
   cache module is imported.

``LOCALAPPDATA``
   For the Numba backend on Windows, sets the cache base directory; entries
   are stored in ``<value>\cccl``. Unset, empty, or relative values fall back to
   ``~\AppData\Local\cccl``. Read when the backend cache module is imported.

``CUDA_COOP_CUTLASS_PROVIDER_CACHE_DIR``
   Selects the CUTLASS provider artifact cache directory. The default is a
   user-specific directory under the system temporary directory. CUTLASS
   always writes provider artifacts for the linker; ``CUDA_COOP_ENABLE_CACHE``
   does not disable this cache. The cache must be a real directory owned by
   the current user where ownership checks are available. The backend sets
   its permissions to ``0700``, including for a configured directory.
   See the :doc:`CUTLASS Developer Guide <cutlass_developer_guide>` for
   artifact lifetime and cache validation.

``CUDA_COOP_SOURCE_DUMP_DIR``
   Writes generated CUDA source to this directory for compiler diagnostics.
   Files use ``cuda_coop_<backend>_<hash>.cu`` names so different backends can
   share a directory. Set it before compiling; both backends also write
   the source when their provider compilation cache is hit. Unset or empty
   disables dumping.

``CUDA_PATH``
   Supplies ``<value>/include`` as a CUDA header candidate if
   ``cuda-pathfinder`` does not resolve one.

``CUDA_HOME``
   Supplies ``<value>/include`` after ``CUDA_PATH`` under the same fallback
   rule.

``CUDA_ROOT``
   Supplies ``<value>/include`` after ``CUDA_HOME`` under the same fallback
   rule.

On Linux and other POSIX systems, ``/usr/local/cuda/include`` is tried last.
Windows uses ``cuda-pathfinder`` or the configured toolkit roots above; it
does not try the Unix fallback. If no valid CUDA include directory is found,
compilation reports a header-resolution error.

Build-time CMake variables
----------------------------

``CUDA_COOP_INSTALL_HEADER_BUNDLE``
   Defaults to ``ON``. Installs the private CCCL header and CMake-package
   bundle into the wheel.

``CUDA_COOP_ALLOW_DIRTY_HEADER_BUNDLE``
   Defaults to ``OFF``. Allows a Git-worktree bundle when selected inputs are
   changed or ``git status`` cannot verify them, and records its source
   revision as ``unknown``.

``CUDA_COOP_CCCL_SOURCE_REVISION``
   Defaults to empty. Supplies the revision token recorded instead of deriving
   it from Git. A dirty or unverifiable Git worktree still records ``unknown``.

Compilation and headers
-----------------------

``cuda-coop`` compiles providers against its configured CCCL root, the active
source checkout during in-tree development, or its installed header bundle,
in that order. It never substitutes the CUDA Toolkit's copy of CUB. CUDA
headers and compiler/linker libraries must resolve to a compatible toolkit.
Shared planner decisions describe the primitive; each backend adapts its
compiler's values and lifecycle to those decisions.

Follow a kernel through the implementation in the
:doc:`Numba-CUDA-MLIR Developer Guide <developer_overview>` or
:doc:`CUTLASS Developer Guide <cutlass_developer_guide>`.
