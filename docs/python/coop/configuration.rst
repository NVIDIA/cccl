.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-configuration:

Configuration
=============

Installation
------------

Install the extra matching the CUDA major version used to compile the kernel:

.. code-block:: console

   python -m pip install "cuda-coop[numba-cuda-mlir-cu13]"
   # Use numba-cuda-mlir-cu12 with CUDA 12.

The base ``cuda-coop`` distribution contains the portable API, type
declarations, and a coherent bundle of CUB, Thrust, and libcu++ headers.
Installed-wheel compilation uses that bundle by default. Development from a
CCCL source checkout uses the matching checkout headers, and
``CUDA_COOP_CCCL_ROOT`` can select another source checkout or ``cuda-coop``
header bundle. Importing :mod:`cuda.coop` does not require Numba-CUDA-MLIR or
an accessible GPU.

The Numba backend is intentionally limited to
``numba-cuda-mlir>=0.5.0,<0.6``. Its private compiler API module
provides access to overload templates, IR, datamodels, and the registries
needed to roll back a failed activation. It does not adapt between runtime
versions. Other runtime series are rejected before compiler registries change.


Runtime environment variables
-----------------------------

For the Boolean switches below, *truthy* means any value other than the
empty string, ``0``, ``false``, ``no``, or ``off``. For example, ``1``,
``true``, ``yes``, and ``on`` are all truthy. Values are case-insensitive,
and leading and trailing whitespace is ignored. An unset variable is false.

``CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION``
   A truthy value disables automatic backend activation during
   :mod:`cuda.coop` import. Explicit qualified-backend import still works.

``CUDA_COOP_CCCL_ROOT``
   Selects a CCCL source checkout or a ``cuda-coop`` header bundle. An invalid
   configured root is an error; resolution does not fall back to another CCCL
   source.

``CUDA_COOP_ENABLE_CACHE``
   A truthy value enables the persistent compiler cache. The value is read
   when the backend cache module is imported.

``XDG_CACHE_HOME``
   On Linux and other POSIX systems, sets the cache base directory; entries
   are stored in ``<value>/cccl``. Unset, empty, or relative values fall back
   to ``~/.cache/cccl``. Read when the backend cache module is imported.

``LOCALAPPDATA``
   On Windows, sets the cache base directory; entries are stored in
   ``<value>\cccl``. Unset, empty, or relative values fall back to
   ``~\AppData\Local\cccl``. Read when the backend cache module is imported.

``CUDA_COOP_SOURCE_DUMP_DIR``
   Writes generated CUDA source to this directory for compiler diagnostics.
   Files use ``cuda_coop_<backend>_<hash>.cu`` names so different backends can
   share a directory. Set it before compiling; the Numba backend also writes
   the source when its provider compilation cache is hit. Unset or empty
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
--------------------------

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
source checkout during in-tree development, or its installed header bundle, in
that order. It never substitutes the CUDA Toolkit's copy of CUB. CUDA headers,
NVRTC, ``nvrtc-builtins``, and nvJitLink must resolve to a compatible toolkit
root. The resulting compiler artifacts and caches include the launch
dimensions, dtype and item extent, storage ABI, compute capability, compiler
options, ordered header identity, and toolkit-library identity.

See :doc:`../coop_api` for the public API reference.
