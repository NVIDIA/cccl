.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-faqs:

FAQs
====

.. _coop-faq-namespaces:
.. _coop-faq-qualified-only:

Which package and import should I use?
--------------------------------------

Install ``cuda-coop-cutlass`` and use ``import cuda.coop.cutlass as coop``.
The import activates its CuTe integration. ``cuda.coop`` is a namespace
parent; public operations live in ``cuda.coop.cutlass``. No separate
registration call or backend selector is needed.

.. _coop-faq-thread-data-local-array:
.. _coop-faq-local-array-payload:

Why use ``ThreadData`` for per-thread values?
---------------------------------------------

``ThreadData`` carries a compile-time item count, element type, and optional
alignment through cooperative operations. A returning Load infers its dtype
from the source. Direct construction supports generated values and auxiliary
outputs, while register-tensor and immutable-vector adapters connect native
CuTe values to this payload model.

Declare ``items_per_thread: cutlass.Constexpr`` on the kernel and launcher
and forward it at launch. The count corresponds to the CUB
``ITEMS_PER_THREAD`` specialization parameter. A value computed from device
data cannot supply a static payload extent. Initialize every slot that an
operation reads and use ``cutlass.range_constexpr`` for payload indexing.

See :ref:`register payloads <coop-cutlass-register-payloads>` for copy,
identity, lifetime, and reconstruction rules. Conversion does not promise
zero-copy behavior, absence of spills, or reconstruction of distributed MMA
ownership from a flat per-thread shape.

.. _coop-faq-returning-load:

What does Load return?
----------------------

``coop.load(group, source, items_per_thread=N)`` always returns a fresh
``ThreadData``, including when ``N=1``. Separate calls have independent
result ownership. Use ``ThreadData`` constructors for generated values;
Load has no caller-provided output parameter.

``alignment`` requests a minimum alignment when returned payload storage is
materialized. It does not establish alignment of the source or destination
pointer. ``offset`` counts source elements, and ``valid_items`` counts tile
elements across the group. Without ``oob_default``, invalid payload slots
are unspecified and must not be read.

.. _coop-faq-shared-load:

How do I stage a tile in shared memory?
---------------------------------------

Use native CuTe copies with deliberately matched source and destination
partitions. Synchronize before another thread consumes the shared tile,
then use returning Load to form per-thread payloads. Store can also write
``ThreadData`` to a shared tensor. Both native shared allocations and typed
``TempStorage.reserve`` views are supported. See
:ref:`shared-memory staging <coop-shared-load-store>`.

A masked-out native copy leaves the destination unchanged. Initialize those
slots explicitly if the application needs a default. Asynchronous native
copies require their own completion protocol; a consumer barrier alone does
not replace that protocol.

.. _coop-faq-thread-data-dtype:

When does ``ThreadData`` need an explicit element type?
-------------------------------------------------------

Load infers its dtype from a supported CuTe source. Direct construction can
infer it from homogeneous initialized values. Use typed scalar assignments
when a computation needs a particular width or signedness. Explicit dtype
construction remains supported, and a consuming primitive may require a
specific type or a narrower set of types than the general payload supports.

Raw MLIR integer values can carry a width without signedness. Preserve their
intended element type in the producer or wrap them in the corresponding
CUTLASS scalar type. Explicit metadata must agree with that width. Giving a
dtype does not initialize missing slots or reconcile conflicting values.
See :ref:`advanced interop <coop-cutlass-dtype-inference>`.

.. _coop-faq-producer:

Who chooses a producer's memory copy and synchronization?
---------------------------------------------------------

A producer hook used by ``ThreadData.load(source)`` chooses the copy or load,
ensures readiness, and owns the source lifetime. The adapter invokes the hook
once and propagates its errors. A bare TMEM tensor is not automatically
adapted, and a flat payload shape cannot recover distributed MMA ownership.
See :ref:`producer interoperability <coop-cutlass-producer-interop>`.

.. _coop-faq-release:

Does releasing storage wait for its users?
------------------------------------------

``TempStorage.release()`` inserts no barrier or wait. Complete and synchronize
outstanding uses before releasing the descriptor. Release consumes it and
its aliases on the executing path, including within branches and loops, and
returns the whole allocated region including padding and exclusive slices.
Using the descriptor or stale aliases after release is an error.

.. _coop-faq-exclusive-storage:
.. _why-use-sharing-exclusive-instead-of-omitting-storage:
.. _why-use-tempstorage-sharing-exclusive:

Why use ``TempStorage`` with ``sharing="exclusive"``?
-----------------------------------------------------

Passing ``coop.TempStorage(sharing="exclusive")`` as ``temp_storage`` gives
distinct call sites separate scratch slices. Omitting ``temp_storage`` lets
the compiler choose the scratch layout and insert reuse barriers. It may
reuse scratch across compatible calls; omission does not guarantee a
separate slice for each call site.

The explicit ``TempStorage`` descriptor retains its capacity, alignment, and
synchronization controls. Two calls with separate slices need no barrier
solely to reuse each other's scratch. This can save that synchronization when
``auto_sync=False``, at the cost of more shared memory. Barriers required by
the algorithm or by application data dependencies still apply.

Repeated execution of one call site, including a loop, reuses its slice.
Synchronize before that reuse or set ``auto_sync=True`` to request trailing
reuse barriers. ``sharing`` controls layout independently of ``auto_sync``.
The backend accounts for cooperative scratch with either explicit or omitted
storage; ``exclusive`` is a choice about which calls may share its bytes.

.. _coop-faq-temp-storage:

Why or when would I provide my own ``TempStorage``?
---------------------------------------------------

Usually, you can omit it. The compiler allocates the scratch required by
an operation and inserts a barrier to make reuse safe. Direct, striped, and
vectorized Load/Store need no shared scratch.

An explicit descriptor lets you choose which supported block operations
share an allocation. For example, a transpose Load and Store can
share scratch while the values stay in each thread's payload:

.. code-block:: python

   # Inside a kernel; source, destination, and items_per_thread are arguments.
   block = coop.this_block()
   scratch = coop.TempStorage(auto_sync=True)
   items = coop.load(
       block, source, items_per_thread=items_per_thread,
       algorithm="transpose", temp_storage=scratch,
   )
   coop.store(
       block, destination, items, algorithm="transpose", temp_storage=scratch
   )

The compiler sizes and aligns the shared region for both calls. It inserts
a barrier after each use. Construct the descriptor inside the kernel, and
keep application values in ``ThreadData`` or your own arrays.

A descriptor also lets you request capacity or alignment, or choose separate
slices with ``sharing="exclusive"``. Explicit descriptors default to
``auto_sync=False``, so the kernel must provide reuse barriers, including across
loop iterations. The example requests ``auto_sync=True`` to insert those
barriers automatically. See :ref:`exclusive scratch slices
<coop-faq-exclusive-storage>` for the tradeoff between memory and reuse
synchronization.

Explicit block descriptors are supported by transpose-family Load/Store,
Reduce, Scan, Merge Sort, Radix Sort, TopK, Adjacent Difference,
Discontinuity, Histogram, and both Run Length Decode forms. Storage-free
Load/Store accept a descriptor without using scratch. Warp operations manage
their own resources and reject explicit descriptors. See
:ref:`the storage model <coop-common-storage>` and
:ref:`storage requirements <coop-cutlass-storage>` for each family's limits.

.. _coop-faq-topk-order:

Does TopK return sorted results?
--------------------------------

No. It selects the smallest or largest keys and places them in a blocked
output prefix without promising their order. Only the first
``min(k, valid_items)`` positions are defined. When keys tie at the selection
boundary, any of the tied keys may fill the remaining positions. Pair variants
keep each selected key attached to its value. Use a sorting primitive when
you need ordered output. See the :ref:`TopK examples <coop-cutlass-topk>`.

.. _coop-faq-global-sort:

Does sorting each block sort the whole array?
---------------------------------------------

Each call sorts only the selected group's tile. Several blocks therefore
produce independently sorted tiles. A globally sorted array requires an
algorithm that combines those tiles. Warp and logical-warp Merge Sort
likewise sort each participating group's tile independently. The
:doc:`Merge Sort <visualizations/merge-sort>` and
:doc:`radix sorting <visualizations/radix>` visualizations illustrate these
operations.

.. _coop-faq-neighbor-operations:

When should I use Adjacent Difference or Discontinuity?
-------------------------------------------------------

Use :func:`cuda.coop.cutlass.adjacent_difference` to compute a value from each item
and its neighbor, such as the delta between successive samples. Use
:func:`cuda.coop.cutlass.discontinuity` to produce flags that mark boundaries, such
as the first and last item of each equal-value run. It returns ``int32``
flags; ``mode="heads_and_tails"`` returns both payloads.

Tile boundaries matter for both operations. Supply a predecessor or
successor when comparisons must continue across tiles. A multiblock
kernel that reads global neighbors must preserve those source values
until all readers finish. The
:doc:`Adjacent Difference <visualizations/adjacent-difference>` and
:doc:`Discontinuity <visualizations/discontinuity>` examples use separate
input and output arrays.

.. _coop-faq-histogram-padding:

Can I zero-pad a partial Histogram tile?
----------------------------------------

Every input sample contributes to a bin, including a padded zero. A
zero-padded load therefore adds extra counts to bin zero. Histogram has
no ``valid_items`` parameter. Process complete tiles or handle the tail
separately with a kernel that counts only valid samples.

Output padding is different: returned counter slots whose bin index is
at least ``bins`` contain zero. Store the first ``bins`` counters using
the striped layout. See the
:doc:`Histogram explorer <visualizations/histogram>`.

.. _coop-faq-histogram-accumulation:

Does Histogram retain counters between calls?
---------------------------------------------

Each call returns fresh counts and preserves its samples. For repeated
tiles within a kernel, keep an accumulator payload and add the returned
counts to it. Each thread retains the same striped bin ownership when
the configuration stays fixed. The :doc:`tested accumulation example
<visualizations/histogram>` shows this pattern.

CUB's ``BlockHistogram::Composite`` accumulates into a caller's counter
buffer. That persistent state belongs to the buffer; it does not require
a persistent C++ Histogram object. The Python API exposes the fresh-count
operation, so neither a parent object nor retained counters in
``TempStorage`` are needed.

.. _coop-faq-rld-lifecycle:

Why do windowed and bulk Run Length Decode use different calls?
---------------------------------------------------------------

:func:`cuda.coop.cutlass.run_length_decode` returns a fixed-size payload for the
window beginning at ``decoded_window_offset``. Use it when the kernel
needs to work with that window's values. It prepares the run table on
each call, even when calls share a ``TempStorage`` descriptor.

:func:`cuda.coop.cutlass.run_length_decode_into` writes the entire expanded
sequence into a destination array. It prepares the table once, loops
over decoding windows internally, and returns the total decoded size to
every thread. The destination must have enough remaining capacity;
use separate source and destination storage.

The prepared run table persists only within the call. A window offset
selects a position in the expanded sequence; the decoder does not advance
a hidden cursor. A shared scratch descriptor controls allocation reuse,
not decoder state. See :ref:`run positions and windows
<coop-glossary-decoding>` and the
:doc:`RLD explorer <visualizations/run-length-decode>`.

.. _coop-faq-rld-padding:

How do I pad run inputs and recognize the end of decoded output?
----------------------------------------------------------------

Use a positive prefix of run lengths followed by zeros. An all-zero tile
represents an empty sequence; an interior zero followed by a positive
length is invalid. The values associated with padding runs are ignored.

A windowed decode fills positions beyond the expanded sequence with
zero. Zero may also be a real run value, so use the total decoded size
to determine which positions are valid. The bulk operation returns
the total to every block member. For a window, derive its valid prefix
from the input run lengths. Bulk decoding leaves the unused destination
suffix unchanged.

.. _coop-faq-batched-reduce:

How does Batched Warp Reduction differ from ordinary Reduce?
------------------------------------------------------------

Ordinary ``reduce(group, values)`` combines all participating threads'
payload items into one aggregate. Both block and Warp reductions accept
one scalar or a fixed-size payload per thread and define the result only
at group rank zero.
``reduce_batched(warp, values)`` reduces each local slot independently across
the warp. Three slots per lane mean three independent results, one for each
slot.

The results are distributed among lanes in blocked or striped order;
they are not broadcast to every lane. Each returned payload has
``ceil(batches / warp_width)`` slots, and slots without a corresponding
batch are unspecified. The :doc:`feature-sum example
<visualizations/reduce-batched>` guards its stores by batch index.

.. _coop-faq-compiler-setup:

Why does import work but kernel compilation fail?
-------------------------------------------------

Import validates the CuTe interfaces. Compilation additionally requires
matching CCCL headers, NVRTC, linker support, and a compatible toolkit.
Check the operation's launch shape, dtype, and participation requirements.
A successful host import alone does not establish GPU runtime qualification.

For provider compilation errors, verify the toolkit and CCCL header origins,
then use ``CUDA_COOP_SOURCE_DUMP_DIR`` to inspect generated source. The
:doc:`Developer Guide <developer_guide>` explains compiler and linker
inspection, including why the final linked cubin is needed to assess calls,
barriers, spills, and resource use.
