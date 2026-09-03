.. _cccl-python-coop:

``cuda.coop``: Cooperative Block Load and Store
================================================

``cuda.coop`` provides cooperative CUDA primitives for Python kernel DSLs.
The initial release integrates with Numba-CUDA-MLIR and supports Block Load
and Block Store. Its portable descriptors and planning records are designed so
that later groups and primitive families can be added without changing the
public dispatch model.

Installation
------------

Install the extra matching the CUDA major version used to compile the kernel:

.. code-block:: console

   python -m pip install "cuda-coop[numba-cuda-mlir-cu13]"
   # Use numba-cuda-mlir-cu12 with CUDA 12.

The base ``cuda-coop`` distribution contains the portable API, type
declarations, and a coherent bundle of CUB, Thrust, libcu++, and CUDAX headers.
Importing :mod:`cuda.coop` does not require Numba-CUDA-MLIR or an accessible
GPU.

Backend activation
------------------

When using the portable namespace with Numba-CUDA-MLIR, import the compiler
runtime first:

.. code-block:: python

   from numba_cuda_mlir import cuda

   from cuda import coop

Because Numba-CUDA-MLIR is already imported, importing :mod:`cuda.coop`
automatically activates its compiler hooks. A standalone :mod:`cuda.coop`
import does not discover or load optional compiler runtimes or CUDA bindings.

If :mod:`cuda.coop` was imported first, activate the backend explicitly before
compiling a kernel:

.. code-block:: python

   from cuda import coop
   import cuda.coop.numba_mlir  # Activate support for portable coop calls.

Alternatively, import :mod:`cuda.coop.numba_mlir` as ``coop`` to use the
qualified namespace and its backend-specific controls.

Kernel API
----------

The portable root and qualified backend expose matching entry points:

.. code-block:: python

   import numpy as np
   from numba_cuda_mlir import cuda

   from cuda import coop

   # Inside a Numba-CUDA-MLIR kernel:
   block = coop.this_block()
   items = coop.ThreadData(2, dtype=np.int32)
   loaded = coop.load(
       block,
       source,
       items,
       valid_items=count,
       oob_default=0,
       offset=source_offset,
   )
   coop.store(
       block,
       destination,
       loaded,
       valid_items=count,
       offset=destination_offset,
   )

Use the qualified namespace when backend-specific types or controls are
required:

.. code-block:: python

   import cuda.coop.numba_mlir as coop

Both spellings are compiler markers. Calls must occur in a compatible compiler
context; they are not host-side data movement operations.

Groups and thread data
----------------------

:func:`cuda.coop.this_block` describes the current CUDA thread block. The
portable group vocabulary also includes thread, warp, cluster, grid, and
mapped-group descriptors, but this release lowers Load and Store only for
``this_block()``. Other targets produce a structured unsupported plan before
provider compilation.

``ThreadData(items_per_thread, dtype=None)`` describes the fixed-size register
payload owned by each participating thread. An untyped Load output infers its
dtype from the source. Load returns the identical output object supplied by the
caller; it does not allocate or substitute another container.

Supported payload types are signed and unsigned 8-, 16-, 32-, and 64-bit
integers plus 32- and 64-bit floating-point values. Boolean, 16-bit floating
point, complex, and mismatched payload types are rejected before NVRTC
compilation.

Load and Store semantics
------------------------

The signatures are:

.. code-block:: python

   load(
       group, source, output, /, *,
       algorithm="direct",
       valid_items=None,
       oob_default=None,
       offset=None,
       temp_storage=None,
   ) -> output

   store(
       group, destination, value, /, *,
       algorithm="direct",
       valid_items=None,
       offset=None,
       temp_storage=None,
   ) -> None

``valid_items`` counts the valid prefix of the entire block tile, not the
number of valid items per thread. With Load, invalid output slots remain
unchanged unless ``oob_default`` is supplied. A default is valid only when
``valid_items`` is present. With Store, elements outside the valid prefix are
not written.

``offset`` is an element offset into the source or destination. It is
independent of ``valid_items`` and is not measured in bytes. Static offsets
must be nonnegative; a runtime offset is a caller-enforced nonnegative
precondition. Source and destination arrays must be one-dimensional and
contiguous. Store accepts both scalar values and multi-item ``ThreadData``
payloads.

The public algorithm vocabulary includes ``direct``, ``striped``,
``vectorize``, ``transpose``, ``warp_transpose``, and
``warp_transpose_timesliced``. The Numba-CUDA-MLIR capability layer currently
executes only ``direct``; selecting another algorithm returns a stable
unsupported plan before provider compilation.

Temporary storage
-----------------

Use ``TempStorage`` to control scratch ownership explicitly:

.. code-block:: python

   scratch = coop.TempStorage(
       size_in_bytes=None,
       alignment=None,
       auto_sync=None,
       sharing="shared",
   )

Omitted storage is planned from the provider requirements. Caller-provided
storage is checked for sufficient size and alignment before code generation.
Shared storage may reuse a slice and defaults to automatic block
synchronization. Exclusive storage receives a distinct slice and cannot
request automatic synchronization.

The compiler uses static shared memory within the default limit and requests
the exact dynamic-memory requirement above it. A requirement beyond the
device's opt-in limit is rejected before launch.

Compilation and headers
-----------------------

``cuda-coop`` compiles providers only against its bundled CCCL headers; it
never substitutes the CUDA Toolkit's copy of CUB. CUDA headers, NVRTC,
``nvrtc-builtins``, and nvJitLink must resolve to a compatible toolkit root.
The resulting compiler artifacts and caches include the launch dimensions,
dtype and item extent, storage ABI, compute capability, compiler options,
ordered header identity, and toolkit-library identity.

See :doc:`coop_api` for the public API reference.
