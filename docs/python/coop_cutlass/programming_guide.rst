.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-cutlass:
.. _cuda.coop.cutlass.programming_guide:
.. _cuda-coop-cutlass-cute-dsl-integration:

Programming Guide
=================

This guide starts with a complete program and explains its parts. The later
sections cover families of primitives. :ref:`Reduction <coop-cutlass-reduce>`
and :ref:`Scan <coop-cutlass-scan>` in the API reference cover reductions and
scans. :doc:`concepts` introduces the ideas the
families share, and the :doc:`API reference <api>` has every signature.

.. _coop-cutlass-load-store:

A first program
---------------

This program computes the running sum of 512 integers with one block of 128
threads. Each thread loads four items, the block computes the running sum
across all of them, and each thread stores its four results:

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/prefix_sum.py
   :language: python

Run it from the repository root:

.. code-block:: bash

   python python/cuda_coop_cutlass/examples/prefix_sum.py

The kernel has four lines. Each one shows something that
:doc:`Programming concepts <concepts>` explains:

* ``coop.this_block()`` is a :ref:`thread group <coop-concepts-groups>`, the
  128 threads that make each call together.
* ``coop.load`` returns each thread's four items as a
  :ref:`ThreadData <coop-concepts-payloads>`, and ``coop.store`` writes one
  to memory.
* ``coop.inclusive_sum(block, values)`` is a
  :ref:`single-phase call <coop-concepts-single-phase>`. It takes the block
  size from the launch, and the element type and item count from ``values``.
* ``coop.inclusive_sum`` also needs scratch space in shared memory, and the
  compiler provides it. A kernel that wants to size or share that space
  passes a :ref:`TempStorage <coop-concepts-storage>`.

On the host side:

* ``make_ptr(cutlass.Int32, ...)`` gives each pointer its element type. That
  is how ``load`` and ``store`` know the items are ``Int32``.
* ``items_per_thread`` is declared ``cutlass.Constexpr`` in the kernel and
  in the launcher, so its value is known when the kernel is compiled.
* ``kernel.launch(grid=1, block=128)`` sets the block size.

The first call to ``launch`` compiles the kernel.
:ref:`Checking and tuning a kernel <coop-cutlass-checking>` shows how to
compile once and launch many times.

.. _coop-cutlass-storage:

Block scratch and reuse
-----------------------

Primitives that need scratch space in shared memory get it from the compiler,
or from a ``TempStorage`` that you pass. :ref:`TempStorage <coop-temp-storage>` in
the API reference covers layouts, barriers, ``reserve`` and ``release``.

The following example transforms eight tiles in a loop with one
``TempStorage``. Each trip around the loop reuses the same bytes. Its
``run_example`` function also accepts ``sharing="exclusive"``, and
``manual_sync=True`` replaces the automatic barriers with ``storage.sync()``
calls.
`View the storage example on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/block_storage.py>`_:

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/block_storage.py
   :language: python
   :start-after: docs: start cutlass-block-storage
   :end-before: docs: end cutlass-block-storage

Typed workspaces for native CuTe code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`TempStorage.reserve <cuda.coop.cutlass.TempStorage.reserve>` returns
an ordinary one-dimensional CuTe shared tensor from a ``TempStorage`` region,
so native CuTe code and cooperative primitives can take turns with the same
bytes. Its API entry has the rules.

A reserved tensor with a supported Load/Store dtype can also serve as the
shared source or Store destination in the
:ref:`staging example <coop-shared-load-store>`. Keep live
staging data separate from scratch that an operation may overwrite. Direct
and striped Load/Store need no scratch. A transpose Load or Store needs
scratch of its own while it reads or writes that live tensor.

The following example passes a reserved tensor to native CuTe code, reuses its
bytes for a cooperative reduction, and checks that a separate native shared
array retains its values. `View the workspace example on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/shared_workspace.py>`_:

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/shared_workspace.py
   :language: python
   :start-after: docs: start cutlass-shared-workspace
   :end-before: docs: end cutlass-shared-workspace

.. _coop-cutlass-device-operators:

Compiled scalar device operators
--------------------------------

``compile_operator(function, dtype, arch=...)`` prepares a CuTe ``@cute.jit``
binary function for ``reduce`` and ``scan``. Call it on the host before
tracing the enclosing kernel. It returns an immutable ``DeviceOperator``
specialized for one scalar dtype and GPU target. Both inputs and the result
must have that dtype, and the function must declare that dtype as its return
annotation. The operator must be pure and associative; it must not synchronize
threads, access mutable shared state, or invoke a cooperative primitive.

For example, the associative operation ``a + b + 1`` can be written as:

.. code-block:: python

   import cutlass
   from cutlass import cute
   import cuda.coop.cutlass as coop

   @cute.jit
   def combine(a: cutlass.Int32, b: cutlass.Int32) -> cutlass.Int32:
       return a + b + 1

   combine_op = coop.compile_operator(combine, cutlass.Int32, arch="sm_80")

Inside a kernel compiled for the same target, use
``coop.reduce(block, value, binary_op=combine_op)`` or
``coop.inclusive_scan(block, value, scan_op=combine_op)``. Exclusive Scan
requires an explicit initial value, for example
``coop.exclusive_scan(block, value, scan_op=combine_op, initial_value=-1)``.
The dtype must match the input values exactly. Block Reduce's
``raking_commutative_only`` algorithm rejects custom operators because the
compiler does not prove their commutativity.

An operator is a scalar function and cannot call a primitive. Call
primitives from the kernel, or from a ``@cute.jit`` function that the kernel
calls.

This requires CUTLASS DSL 4.9 device-function export support through
``cute.compile_to`` and ``cutlass.compiler.ArtifactType.LtoIr``, including
typed declaration metadata. The callback's LTO-IR, the generated CUB wrapper,
and the enclosing kernel are linked together. Existing group, scratch, and
synchronization rules still apply. Runtime captures, Scan prefix callbacks,
and operators returning tuples or structures are unsupported.

.. _coop-cutlass-histogram:

Counting samples into bins
--------------------------

``histogram`` takes integer bin indices and returns fresh per-thread
counters. Use a complete one-dimensional block and initialize every sample
to an index in ``[0, bins)``. Samples may be uint8, int32, uint32, int64, or
uint64; counters independently use int32, uint32, int64, or uint64.

Each thread receives ``bins_per_thread`` counters in striped order: thread
``t`` owns bins ``t + i * block_size``. The output capacity must cover every
bin, and slots beyond ``bins`` contain zero. Use striped Store to write the
counters in bin order. In this example, 64 threads each receive two counters
to cover 65 bins. The first 65 output positions hold counts, and the other
63 positions are zero:

`View test_histogram.py on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_histogram.py>`_

.. literalinclude:: ../../../python/cuda_coop_cutlass/tests/runtime/test_histogram.py
   :language: python
   :start-after: docs: start cutlass-histogram
   :end-before: docs: end cutlass-histogram
   :dedent: 4

Both ``algorithm="atomic"`` and ``algorithm="sort"`` preserve the input.
Each call initializes fresh counters, including when calls share an
explicit ``TempStorage`` descriptor. Add the returned counters yourself to
accumulate several tiles, choosing a wide enough counter dtype. There is no
``valid_items`` control: padding contributes counts just like other input.

Calls also accept CuTe register tensors, immutable vectors, and
CuTe counter dtype selectors. The result is always fresh ``ThreadData``.
See the :doc:`Histogram explorer <visualizations/histogram>` for the
shared ownership and accumulation rules.

.. _coop-cutlass-checking:

Checking and tuning a kernel
----------------------------

Check values and ownership against a CPU reference before timing a kernel.
Include partial tiles, nonzero offsets, multiple warp groups, and repeated
scratch reuse when those cases occur in the application. For operations with
undefined tails or nonleader results, compare only the defined outputs.

Compile before timing and synchronize the measured work. ``cute.compile``
returns a callable you can retain for repeated launches; the
:ref:`debugger walkthrough <cuda.coop.cutlass.debugger_walkthrough>` shows
compilation followed by two executions. The first compilation includes
provider generation, NVRTC, and device linking.

To inspect generated C++, set ``CUDA_COOP_CUTLASS_SOURCE_DUMP_DIR`` before compilation.
Use the final linked cubin to assess inlining, barriers, shared memory, and
register use. A provider's source or intermediate PTX does not establish what
remains in the kernel. Use Compute Sanitizer race checking when changing
scratch reuse or synchronization.

More items per thread can increase register use, and more scratch can reduce
the number of blocks resident on a multiprocessor. Check the compiled kernel's
resource usage as well as its run time.

.. _coop-cutlass-run-length:

Run Length Decode
-----------------

``run_length_decode`` expands blocked run values and lengths into a fresh
per-thread window. ``run_length_decode_into`` writes the full decoded stream
to a contiguous one-dimensional CuTe global-memory tensor and returns a
``Uint32`` total to every block member. Both operations require a complete
one-dimensional block and preserve their inputs.

Positive run lengths must precede trailing zero padding. Negative lengths,
misplaced padding, and totals exceeding uint32 trap before decoding. A window
starting beyond the decoded stream contains zeros. Bulk decoding checks the
destination offset and remaining capacity before writing; empty input leaves
the destination unchanged. The internal prepared table is retained across
all bulk windows.

Both functions also accept CuTe register payloads. Decoded totals use
``Uint32``.

Use ``run_length_decode_into`` when the complete decoded result belongs in
memory. It prepares the run table once, then decodes and stores successive
windows while keeping that table alive. ``decoded_items_per_thread`` sets
the size of each register window; the output can be much larger. The call
handles the final partial window and returns the full decoded size.

The example below contrasts this with an explicit loop over
``run_length_decode``. That form lets the kernel process each window's
register values before advancing, but each call repeats table preparation.
Reusing ``TempStorage`` provides storage for those calls; it does not retain
a prepared decoder between them.

With ``items_per_thread=1``, 32 input runs expand to 94 items: 31 runs have
length three and the last has length one. One decoded item per thread gives
32-item windows at decoded offsets 0, 32, and 64. The first two windows are
full; the last has 30 valid items. Both variants write all 94 items beginning
at destination offset 5 and preserve the prefix and suffix sentinels. The
host checks both outputs against ``numpy.repeat``. With four runs per
thread, the same example expands 382 items through 12 windows.

`View the complete example and CPU checks on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_run_length_documentation.py>`_

.. literalinclude:: ../../../python/cuda_coop_cutlass/tests/runtime/test_run_length_documentation.py
   :language: python
   :start-after: # cutlass-run-length-example-begin
   :end-before: # cutlass-run-length-example-end
   :dedent: 4

.. _coop-shared-load-store:

Shared-memory staging
---------------------

Load accepts compact global- or shared-memory sources, including native CuTe
shared tensors and typed tensors from ``TempStorage.reserve``. Store accepts
the corresponding memory destinations. Load infers its dtype from a shared
tensor just as it does from a global-memory source.

Use native CuTe copies when staging a global tile in shared memory. Partition
the source and destination with the same thread/value mapping, issue the
copy, and synchronize before another thread consumes the shared tile.
Then ``coop.load(block, shared, items_per_thread=N)`` returns a fresh
payload. Select its algorithm according to the ownership required by the
next operation; a striped reload changes which thread owns each item.

A predicated native copy leaves masked-out destinations unchanged. Initialize
invalid slots when the application needs a default value. Without a default,
do not read values outside the valid prefix. Source and destination must have
compatible element types, sufficient capacity, and nonoverlapping storage.
The copy atom and view alignment must satisfy the selected native operation.
Asynchronous copies also require their own completion protocol before the
consumer's synchronization.

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/shared_staging.py
   :language: python
   :start-after: docs: start cutlass-native-shared-staging
   :end-before: docs: end cutlass-native-shared-staging

Store-to-shared is also supported: Load into ``ThreadData``, Store into a
native shared tensor or reserved workspace, then synchronize before another
thread reads those writes. Application shared storage must remain disjoint
from scratch that a concurrent primitive can overwrite.

.. _coop-cutlass-producer-interop:

Producer-owned register loads
-----------------------------

``ThreadData.load(source)`` adapts a producer that explicitly provides the
``__cuda_coop_thread_data_load__`` tracing hook. The producer chooses the
copy atom and thread partition, ensures that its source is ready, and owns
its lifetime. A producer can use that hook for a tensor-memory-to-register
copy when its application supplies the required mapping and synchronization.

Each call invokes the hook exactly once during tracing and propagates the
producer's exception. A single-use producer must reject subsequent calls
itself. Capability detection requires a declared hook; a dynamic
``__getattr__`` fallback does not establish that capability. The hook must
return ``ThreadData`` or a statically sized per-thread register payload
accepted by ``from_payload``. Optional ``items_per_thread`` and ``dtype``
constrain the adapted result; a supplied item count is validated before the
hook runs. Invalid result diagnostics identify the source and result types.

``ThreadDataLoadSource`` describes the producer hook.
``ThreadDataTensorMetadata`` describes static ``shape`` and ``dtype``
properties. ``ThreadDataSource`` combines the two protocols. These protocols
make the boundary explicit; they do not select a device copy or wait for
asynchronous work on the producer's behalf.

``values.to_tensor_ssa(like=source)`` supplies omitted shape and dtype from
that metadata. Explicit arguments take precedence. Shapes must be positive,
fully static, and contain exactly the payload's item count. Conversion
preserves flat per-thread item order, and the default shape is
``(items_per_thread,)``. It does not recover an input fragment's original
shape or reconstruct distributed MMA ownership from that shape.

A bare TMEM tensor does not carry the copy atom, partition, readiness, and
lifetime contract needed for this adaptation. Supply an explicit producer
capability and qualify its copy and synchronization on suitable hardware.
Register conversions do not promise zero-copy behavior or the absence of
local-memory spills; inspect the final kernel for those properties.
