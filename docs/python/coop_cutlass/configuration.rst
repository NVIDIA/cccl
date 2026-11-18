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

``CUDA_COOP_CUTLASS_PROVIDER_PCH``
   Defaults to ``auto``. On a provider cache miss, NVRTC can reuse precompiled
   headers for another specialization with a compatible header prefix in the
   same process. Set this to ``off`` to compile each miss without PCH.

``CUDA_COOP_CUTLASS_AOT_PACK_PATH``, ``CUDA_COOP_CUTLASS_AOT_MODE``
   Select an :ref:`ahead-of-time pack <coop-cutlass-aot-packs>` by absolute
   path. The mode can be ``auto`` (default), ``required``, or ``off``.
   ``cuda-coop-aot run`` sets both for the workload and its child processes.
   Each process reads the pack at first use. Give replacement packs a new
   path. An explicit ``aot.use`` overrides these settings for its context.

``CUDA_COOP_CUTLASS_SOURCE_DUMP_DIR``
   Writes generated CUDA source to this directory for compiler diagnostics.
   Files use ``cuda_coop_cutlass_<hash>.cu`` names. Set it before compiling; it also writes
   the source when the provider compilation cache is hit. Unset or empty
   disables dumping.

.. _coop-cutlass-aot-packs:

Ahead-of-time provider packs
----------------------------

To move provider compilation off the deployment path, run a representative
workload once and capture its providers. Start with a new pack directory name.
The capture must reach CuTe tracing for each specialization you want to ship.
For example, run these commands from a checkout with the package installed:

.. code-block:: bash

   example=python/cuda_coop_cutlass/examples/block_storage.py
   CUTE_DSL_NO_CACHE=1 cuda-coop-aot capture --output providers -- python "$example"
   cuda-coop-aot inspect providers
   cuda-coop-aot run --pack providers --mode required -- python "$example"

You can copy the entire directory to another location before running the last
command. It contains a manifest, provider LTO-IR, and the scratch sizes and
alignments from the same compilation. Select a pack from a trusted producer.
An exact hit uses the compiled CUB implementation from that pack. The GPU
target, host operating system, generated source, layout queries, provider
ABI and code-generation options must match. Pack selection checks that the
nvJitLink library visible to CUDA Python has the same major version as the
producing NVRTC and at least its minor version. CuTe's final device linker
must also support that LTO-IR version.

In Python, wrap the call that traces the kernel:

.. code-block:: python

   from cuda.coop.cutlass import aot

   with aot.capture("providers"):
       compile_my_kernel()

   with aot.use("providers", mode="required"):
       compile_my_kernel()

An exact hit writes verified bytes to the local provider cache for the final
link. CuTe still compiles and links the enclosing kernel. In ``required``
mode a missing or incompatible provider raises ``aot.PackMissError``. In
``auto`` mode it uses the ordinary provider cache and JIT compiler. Use
``off`` to suspend pack selection inside an enclosing ``aot.use`` context.
A corrupt pack raises ``aot.PackIntegrityError`` when selected.

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
