.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-configuration:

Configuration
=============

Installation
------------

Install ``cuda-coop`` without adding Python package dependencies:

.. code-block:: console

   python -m pip install cuda-coop

The wheel includes the common API, every shipped DSL integration (including
``cuda.coop.numba_mlir`` and ``cuda.coop.cutlass``), type declarations, and a
matching bundle of CUB, Thrust, and libcu++ headers. The base install
declares no Python package dependencies. You can import ``cuda.coop`` without
a compiler or GPU; using an integration requires its backend dependencies to
be installed.

For Numba-CUDA-MLIR, install the extra matching your CUDA major version:

.. code-block:: console

   python -m pip install "cuda-coop[numba-cuda-mlir-cu13]"
   # Use numba-cuda-mlir-cu12 with CUDA 12.

Both commands install the same ``cuda-coop`` wheel with the same DSL
integrations. The extra only adds the dependency requirements declared in
``pyproject.toml`` so pip installs the supported Numba-CUDA-MLIR stack for
the selected CUDA major version. The current integration requires
``numba-cuda-mlir>=0.5.0,<0.6``.
For CUTLASS / CuTe DSL, install the base wheel alongside a runtime meeting
the :ref:`CUTLASS requirements <coop-cutlass-requirements>`. A public CUTLASS
extra and minimum version await qualification of an official artifact.

Installing an extra does not register a backend in a running Python process;
see :ref:`installation versus registration <coop-faq-installed-extra>`.

Installed-wheel compilation uses the bundled CCCL headers. Development from a
CCCL source checkout uses its matching headers. ``CUDA_COOP_CCCL_ROOT`` can
select another source checkout or ``cuda-coop`` header bundle.


With Numba-CUDA-MLIR 0.5.x, activating the ``cuda.coop`` backend disables
the compiler's ``cache=True`` disk cache for all kernels in that process.
Compiled kernels still have an in-memory cache. The provider cache controlled
by ``CUDA_COOP_ENABLE_CACHE`` below is separate.

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
   user-specific directory under the system temporary directory. See the
   :doc:`CUTLASS Developer Guide <cutlass_developer_guide>` for artifact
   lifetime and cache validation.

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
source checkout during in-tree development, or its installed header bundle,
in that order. It never substitutes the CUDA Toolkit's copy of CUB. CUDA
headers and compiler/linker libraries must resolve to a compatible toolkit.
Shared planner decisions describe the primitive; each backend adapts its
compiler's values and lifecycle to those decisions.

Follow a kernel through the implementation in the
:doc:`Numba-CUDA-MLIR Developer Guide <developer_overview>` or
:doc:`CUTLASS Developer Guide <cutlass_developer_guide>`.
