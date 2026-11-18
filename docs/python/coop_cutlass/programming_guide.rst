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

.. _coop-cutlass-radix:

Radix Sort and Rank
-------------------

``radix_sort_keys(block, keys, ...)`` returns sorted keys, and
``radix_sort_pairs(block, keys, values, ...)`` returns sorted keys and their
associated values. ``radix_rank_keys(block, keys, ...)`` instead returns each
input item's position in the order of a selected digit. It preserves the
input arrangement; use the returned ranks when assigning destinations.
All three primitives preserve their inputs. Equal selected digits retain
flattened blocked input order in both ascending and descending modes.

These primitives require a complete physical block, including multidimensional
blocks. Every block thread participates with identical controls and per-thread
extents. A block tile contains at most 65,535 items. Warp and mapped groups
are unsupported. Inputs use blocked layout, and array results are fresh
``ThreadData`` payloads with the same item count. Read-only inputs are
accepted. Sort preserves the key and value dtypes; Rank returns signed
``cutlass.Int32`` values. The :doc:`Radix visualization
<visualizations/radix>` illustrates the relation between digits, ranks,
and sorted positions.

The API accepts payloads of ``int32``, ``uint32``, ``int64``, or
``uint64`` keys. Pair values may use any of the ten element types of
:ref:`ThreadData <coop-thread-data>`; key and value extents must match. The API additionally
accepts scalars and CuTe register tensors. Scalar pairs require two scalars;
array pairs require two arrays. Scalar inputs produce scalar results.
Register-tensor inputs produce ``ThreadData`` results.

Sort intervals and output layout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sort uses the half-open transformed-bit interval ``[begin_bit, end_bit)``.
The default begin is zero, and the default end is the key width, including
when begin is nonzero. Bounds must satisfy
``0 <= begin_bit < end_bit <= key_width``. They may be runtime signed integers
up to 64 bits or unsigned integers up to 32 bits; invalid runtime bounds trap
before narrowing. ``descending`` is a compile-time Boolean.

Signed integer keys invert their sign bit before selecting digits. CUTLASS
Sort also accepts ``float32`` and ``float64`` keys. Floating keys invert all
bits when negative and only the sign bit when nonnegative. Negative and
positive zero compare equivalently; NaNs follow transformed-bit ordering.
Returned keys retain their original bit representations.

Results use blocked layout by default. CUTLASS Sort's compile-time
``blocked_to_striped=True`` maps item ``i`` at thread ``t`` to sorted index
``i * block_threads + t``. This applies to both outputs of a pair sort.
Match the Store algorithm to this register layout when writing a contiguous
sorted tile. Sort accepts a :ref:`TempStorage <coop-temp-storage>`. It does
not accept a valid count,
comparison callback, or algorithm selector.

Rank digits and bin prefixes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rank keys are always 32- or 64-bit integers, including through the
API. Signed keys invert the sign bit before digit extraction. Its bounds
and ``radix_bits`` must be compile-time integers, selecting one through eight
bits within the key width. Omitted ``end_bit`` means ``begin_bit + radix_bits``,
or ``begin_bit + 4`` when both controls are omitted. An explicit end and width
must describe the same interval. Defaults do not clamp to the key width.
Rank manages scratch and reuse synchronization automatically and does not
accept ``temp_storage``.

CUTLASS Rank adds ``exclusive_digit_prefix``, a writable signed Int32
``ThreadData`` output distinct from the keys. Its dtype may be inferred.
For digit width ``R`` and block size ``T``, each thread supplies
``P = max(1, ceil(2**R / T))`` slots. Slot ``i`` of thread ``t`` owns bin
``t * P + i`` in both directions. Each prefix counts keys with smaller digits
in ascending mode or greater digits in descending mode. Slots beyond the
number of bins are undefined and must not be read. These bin counts are
separate from the returned per-item ranks.

This example sorts full signed keys, orders pairs by their transformed high
digit, and computes that digit's inverse ranks. The example also uses
striped Sort output and descending bin prefixes. CPU checks verify stable
pair order, the signed-key transformation, ranks, and input preservation.
`View the Radix Sort and Rank example on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/radix.py>`_:

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/radix.py
   :language: python
   :start-after: docs: start cutlass-radix
   :end-before: docs: end cutlass-radix

.. _coop-cutlass-topk:

TopK selection
--------------

``topk_min_keys`` and ``topk_max_keys`` select a tile's smallest or largest
keys. ``topk_min_pairs`` and ``topk_max_pairs`` also carry each selected value
with its key. Controls follow the :func:`TopK contract
<cuda.coop.cutlass.topk_min_keys>`: provide ``k``, optionally limit the input with
``valid_items``, and optionally supply ``temp_storage``.

TopK requires a complete one-dimensional block and fixed per-thread payloads
in blocked order. All threads participate with uniform counts and extents.
For tile capacity ``N = block_threads * items_per_thread``, both counts lie in
``[0, N]``; omitted ``valid_items`` means ``N``. Runtime counts may be signed
integers up to 64 bits or unsigned integers up to 32 bits. Invalid runtime
counts trap before narrowing.

Only the first ``min(k, valid_items)`` blocked output positions are defined.
Results are unsorted, and selection and ordering among equal keys are
unspecified. If either count is zero, no result positions are defined.
Restrict Store to the defined prefix; do not read or store the remaining
positions.

Keys and values may use signed or unsigned 8-, 16-, 32-, or 64-bit integers,
or 32- or 64-bit floats. Pair extents must match; their dtypes may differ.
Readable inputs need not support mutation. TopK preserves both inputs and
returns fresh ``ThreadData`` with the original dtypes and extents. Signed
floating-point zeros compare equally while retaining their original bits;
NaNs have no guaranteed numeric ordering.

The API additionally accepts CuTe register tensors and immutable
register values, including mixed register-tensor and ``ThreadData`` pairs.
Its results remain ``ThreadData``, and its controls match the API.
Scalar inputs are unsupported. TopK accepts a
:ref:`TempStorage <coop-temp-storage>`.

The :doc:`TopK visualization <visualizations/topk>` shows the selected
prefix. This example selects the smallest keys and largest pairs from a
partial tile, reusing scratch and preserving both original payloads. Its CPU
checks compare selected multisets and pair identities without assuming
output order or stable ties. The example also demonstrates register
payload conversion.
`View the TopK example on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/topk.py>`_:

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/topk.py
   :language: python
   :start-after: docs: start cutlass-topk
   :end-before: docs: end cutlass-topk

.. _coop-cutlass-neighbors:

Comparing neighboring values
----------------------------

``adjacent_difference`` subtracts the previous or next item from each item
in a block's flattened, blocked sequence. ``discontinuity`` returns int32
head flags, tail flags, or both. Both operations preserve their input and
support the ten numeric payload dtypes. Every thread in the complete block
participates, including in multidimensional blocks.

Pass ``tile_predecessor_item`` or ``tile_successor_item`` to compare across
tile boundaries. Adjacent Difference also accepts a block-uniform
``valid_items`` count; its invalid suffix retains the original input.
Right partial tiles cannot take a successor, and Discontinuity requires a
full tile. Without an external neighbor, Adjacent Difference preserves the
boundary input, while Discontinuity marks the boundary as a head or tail.
These defaults let a tile act as a separate sequence; provide a neighbor
when it continues a sequence in another tile. See
:doc:`Adjacent Difference <visualizations/adjacent-difference>` and
:doc:`Discontinuity <visualizations/discontinuity>` for the shared rules.

Both functions accept explicit block ``TempStorage``. The example below
reuses one descriptor between the operations; automatic trailing
synchronization makes that reuse safe. The functions also accept
CuTe register tensors and immutable register vectors. They snapshot the
input and return fresh ``ThreadData``; flags contain CuTe ``Int32`` values.
Custom arithmetic and flag callbacks are not supported.

`View test_neighbors.py on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_neighbors.py>`_

.. literalinclude:: ../../../python/cuda_coop_cutlass/tests/runtime/test_neighbors.py
   :language: python
   :start-after: docs: start cutlass-neighbors
   :end-before: docs: end cutlass-neighbors
   :dedent: 4

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
