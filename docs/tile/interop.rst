.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Tile and SIMT interoperation
============================

Both language interfaces use one CCCL-owned ``cuda_coop_tile.tilelib``.
The library contains the operation declarations and the native linkage
provider. Python lowering emits a call through
``cuda.tile_preview.PreviewForeignCall``. The C++ header emits the
corresponding foreign call, and ``nvcc`` passes the TileLib to TileIRAS
while compiling the kernel.

The provider offers CUB implementations with explicit element types,
layouts, thread participation, and resource requirements. The compiler
chooses a compatible implementation and connects its results to the
surrounding Tile computation. The :doc:`developer guide </tile/development>`
follows this process through the source.

Selecting a TileLib
-------------------

The Python distribution places its TileLib at
``cuda/coop/tile/_native/cuda_coop_tile.tilelib``. The Python interface and
C++ configuration helpers resolve that package-relative artifact by
default. The native CMake installation locates the TileLib relative to its own
installation prefix.

Set ``CUDA_COOP_TILE_LIBRARY_PATH`` to select an exact alternative file.
The resolver reports an invalid explicit path instead of silently choosing
another library. Use this override when testing a rebuilt provider against
an existing frontend installation.

An explicit file still needs to match the declarations and compiler ABI
used by the consumer. Changing only the path does not make a provider
from a different source revision compatible.

Artifact identity and compilation caches
----------------------------------------

Python preview calls carry a library identity along with the path. The
distribution's ``provider-manifest.json`` records its distribution, version,
TileLib filename, and hash. The resolver validates packaged identity so a
changed or mismatched artifact is diagnosed before a cached compilation
can be mistaken for the intended build.

An unpackaged development TileLib needs
``CUDA_COOP_TILE_LIBRARY_VERSION``. Use a version token tied to its source
and build inputs, and change it after rebuilding different provider code
or dependencies. Reusing the identity can retain a compiled kernel from
an earlier library.

This compiler cache is separate from the native provider's in-process
caches for resource probes and compiled implementation bundles. Provider
keys include the validated specialization, target, explicit epoch, and ordered
include paths. These keys do not detect changed header contents or compiler
binaries at the same paths. Restart the provider or change its cache epoch
when testing such a change, and separately invalidate the frontend cache. The
:doc:`developer guide </tile/development>` explains the distinction when
measuring compilation or investigating stale artifacts.

CMake consumers also need a file dependency because the TileLib is read
while compiling CUDA source. Call ``cuda_coop_tile_link(target)`` after
adding the target's CUDA sources. Each consuming target needs the helper;
a host linker dependency alone does not cause CUDA recompilation when the
TileLib changes.

Combining providers in one kernel
---------------------------------

Multiple Tile libraries may contribute calls to a kernel. Each provider
owns its declarations and artifact identity. The calls must use a
compatible compiler/interop ABI and admit a common physical execution.
A composition can fail even when each library compiles successfully in
an independent kernel.

For example, an explicit Exchange width constrains the participating
threads. A call from another library must be compatible with that width.
Resource requirements can also change when several calls appear together.

When composing a ``cuda.coop.tile`` kernel with another provider's Tile
library, record the provider artifacts and compare the composed kernel's
outputs with an independent reference. Validate the complete composition,
including its numerical behavior and resource requirements.

Provider traces
---------------

Enable traces before compiling the reproducer:

.. code-block:: bash

   export CUDA_COOP_TILE_PROVIDER_TRACE=1
   export CUDA_COOP_TILE_PROVIDER_TRACE_FILE=/path/to/provider-trace.log
   python reproducer.py

The trace records provider discovery, resolved implementations, resource
estimates, and compilation/cache events. CUDA 13.5 filters and scores
candidates during ``provide_impls``; it does not invoke ``filter_and_rank``.
Discovery knows each candidate's width, but not the enclosing kernel's final
worker count. Instantiation validates that resolved count before compilation.
If a compiler cache returns a finished kernel, that run may not invoke the
provider. Use the frontend's cache controls when the investigation needs a
fresh provider compilation, and keep that choice with the reproducer.

For a failure report, preserve the kernel source and inputs, shapes and
dtypes, worker-width settings, exact package/compiler identities, TileLib
hash, and complete diagnostic. Identify the phase that failed: Python
lowering or C++ overload resolution, implementation discovery, ranking,
instantiation, NVRTC compilation, or GPU execution. A numerical result
does not establish that an unexercised provider callback behaved correctly.

Selection qualification checks early catalog scores and the implementation
chosen for compilation; see :doc:`/tile/development` for focused validation.
