.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-configuration:

Configuration
=============

Use these settings to choose headers, control compiler caches, or inspect
generated code. Build-time settings apply when packaging ``cuda-coop-cutlass``.

Runtime environment variables
-----------------------------

``CUDA_COOP_CUTLASS_CCCL_ROOT``
   Selects a CCCL source checkout or a ``cuda-coop-cutlass`` header bundle. An invalid
   configured root is an error; resolution does not fall back to another CCCL
   source.

``CUDA_COOP_CUTLASS_PROVIDER_CACHE_DIR``
   Selects the CUTLASS provider artifact cache directory. The default is a
   user-specific directory under the system temporary directory. CUTLASS
   writes provider artifacts for the linker. The cache must be a real directory owned by
   the current user where ownership checks are available. The backend sets
   its permissions to ``0700``, including for a configured directory.
   The setting is read on each provider compilation request, including cache
   hits. Changing it selects a separate cache; existing compiled kernels keep
   their artifacts. Set it before ``cute.compile``; a CuTe cache hit that skips
   provider compilation will not populate a new provider directory.
   See the :doc:`Developer Guide <developer_guide>` for artifact lifetime and
   cache validation.

``CUDA_COOP_CUTLASS_SOURCE_DUMP_DIR``
   Writes generated CUDA source to this directory for compiler diagnostics.
   Files use ``cuda_coop_cutlass_<hash>.cu`` names. Set it before compiling; it also writes
   the source when the provider compilation cache is hit. Unset or empty
   disables dumping.

Walkthrough and test settings
-----------------------------

``CUDA_COOP_CUTLASS_EXAMPLE_PRINT_IR``
   Prints finalized MLIR in the :doc:`cutlass_integration_example` when set
   to a nonempty value.

``CUDA_COOP_CUTLASS_EXAMPLE_ARTIFACT_DIR``
   Selects an existing parent directory for that example's temporary LTO-IR
   directory. It defaults to the system temporary directory. The example
   reads this setting at startup and removes its temporary directory on exit;
   it does not use the package's persistent provider cache.

``CUDA_COOP_CUTLASS_RUN_RACECHECK``
   Set to ``1`` to run the optional Run Length Decode sanitizer test.

CUDA Toolkit settings
---------------------

These variables belong to the CUDA toolchain and retain its standard names.

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

``CUDA_COOP_CUTLASS_INSTALL_HEADER_BUNDLE``
   Defaults to ``ON``. Installs the private CCCL header and CMake-package
   bundle into the wheel.

``CUDA_COOP_CUTLASS_ALLOW_DIRTY_HEADER_BUNDLE``
   Defaults to ``OFF``. Allows a Git-worktree bundle when selected inputs are
   changed or ``git status`` cannot verify them, and records its source
   revision as ``unknown``.

``CUDA_COOP_CUTLASS_CCCL_SOURCE_REVISION``
   Defaults to empty. Supplies the revision token recorded instead of deriving
   it from Git. A dirty or unverifiable Git worktree still records ``unknown``.

Compilation and headers
-----------------------

``cuda-coop-cutlass`` compiles providers against its configured CCCL root, the active
source checkout during in-tree development, or its installed header bundle,
in that order. It never substitutes the CUDA Toolkit's copy of CUB. CUDA
headers and compiler/linker libraries must resolve to a compatible toolkit.
Plans describe the primitive; the lowering adapts CuTe values and lifecycle
to those decisions.

Keep headers and the toolkit installation unchanged while compilation is in
progress. Recreate providers and recompile kernels after changing those
inputs. Follow the compilation path in the :doc:`Developer Guide
<developer_guide>`.
