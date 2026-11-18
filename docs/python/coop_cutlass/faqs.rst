.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-faqs:

FAQs
====

.. _coop-faq-namespaces:
.. _coop-faq-namespace:

Which package and import should I use?
--------------------------------------

Install ``cuda-coop-cutlass`` and import its public API:

.. code-block:: python

   import cuda.coop.cutlass as coop

The import activates its CuTe integration. ``cuda.coop`` is a namespace
parent; public operations live in ``cuda.coop.cutlass``. No separate
registration call or backend selector is needed.

.. _coop-faq-thread-data-local-array:
.. _coop-faq-local-array-payload:

Why use ``ThreadData`` for per-thread values?
---------------------------------------------

A primitive is compiled for a known number of items per thread and a known
element type. ``ThreadData`` carries both, so a call such as
``coop.inclusive_sum(block, values)`` needs no other arguments. The item
count is a template argument of the CUB code behind the call, which is why
it is a Python integer and cannot come from device data.

:ref:`ThreadData <coop-thread-data>` in the API reference has the rules and
the conversions to and from CuTe register tensors. A conversion copies the
items in flat index order. Which thread owns which element of an MMA
fragment is for the kernel to keep track of.

.. _coop-faq-returning-load:

What does Load return?
----------------------

A new ``ThreadData`` in each thread, with ``items_per_thread`` items of the
source's element type. That holds for ``items_per_thread=1`` too, and each
call returns its own payload:

.. code-block:: python

   # Inside a CuTe kernel; source is a global or shared memory operand.
   block = coop.this_block()
   items = coop.load(block, source, items_per_thread=2)
   one_item = coop.load(block, source, items_per_thread=1)
   # Both results are ThreadData; each call returns a separate payload.

``offset`` counts source elements, and ``valid_items`` counts elements of
the group's tile. Items past ``valid_items`` have no defined value unless the
call passes ``oob_default``. ``alignment`` applies to register storage made
from the returned payload, and says nothing about the source pointer.

.. _coop-faq-shared-load:

How do I stage a tile in shared memory?
---------------------------------------

Use native CuTe copies with deliberately matched source and destination
partitions. Synchronize before another thread consumes the shared tile,
then use ``load`` to form per-thread payloads. Store can also write
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

When its items are raw MLIR integers, which have a width and no sign. Pass
``dtype`` with that width, or wrap each item in a CUTLASS type such as
``cutlass.Uint32``.

Otherwise leave it out. ``load`` takes the type from the memory, and a
primitive takes it from the items of a payload the kernel fills, which must
all have the same type.

``to_tensor_ssa`` and ``to_register_tensor`` need the type as well. A
payload from ``load``, a primitive or a conversion carries it. For one the
kernel built with ``ThreadData(n)``, ``from_values`` or ``from_fn`` and no
``dtype``, pass ``dtype`` to the call, or ``like`` to ``to_tensor_ssa``.

Passing ``dtype`` to ``ThreadData(...)`` assigns no missing items. Python
numbers among the items take that type if it can hold them. An integer out
of range, or a float in an integer payload, is a compile error. An item
that already has a CuTe type is not converted and must have exactly that
type. ``ThreadData.from_fn`` with ``dtype`` converts.
:ref:`ThreadData <coop-thread-data>` in the API reference lists the
supported types.

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

No. ``release`` adds no barrier. Every thread must be past the last
primitive that used the region before any thread writes through the returned
tensor. See
:meth:`TempStorage.release <cuda.coop.cutlass.TempStorage.release>`.

.. _coop-faq-exclusive-storage:
.. _why-use-sharing-exclusive-instead-of-omitting-storage:
.. _why-use-tempstorage-sharing-exclusive:

Why use ``TempStorage`` with ``sharing="exclusive"``?
-----------------------------------------------------

It uses more shared memory and needs fewer barriers. With
``sharing="exclusive"`` each call site has its own slice, so two different
calls never reuse each other's scratch and need no barrier between them when
``auto_sync=False``. See :ref:`TempStorage <coop-temp-storage>` for the two
layouts side by side.

Leaving ``temp_storage`` out always adds a barrier after each call. The
compiler picks the layout, which today is one region per call.

.. _coop-faq-temp-storage:

Why or when would I provide my own ``TempStorage``?
---------------------------------------------------

Usually you can leave it out. The compiler provides the scratch each call
needs and adds a barrier after it. Pass your own to choose which calls share
bytes, to place the barriers yourself, to set the region's size or
alignment, or to keep your own data in the region. This kernel shares one
region across three calls:

.. literalinclude:: ../../../python/cuda_coop_cutlass/tests/runtime/test_concepts_documentation.py
   :language: python
   :start-after: # cutlass-concept-temp-storage-begin
   :end-before: # cutlass-concept-temp-storage-end

:ref:`TempStorage <coop-temp-storage>` in the API reference lists the
primitives that accept one and shows the layouts and the barriers.
:ref:`Exclusive slices <coop-faq-exclusive-storage>` use more shared memory
and need fewer barriers.

.. _coop-faq-topk-order:

Does TopK return sorted results?
--------------------------------

No. The selected keys fill the first positions of the result in no defined
order. To get them in order, pass the result of ``topk_min_keys`` or
``topk_max_keys`` to :func:`~cuda.coop.cutlass.merge_sort_keys` with
``valid_items`` set to the number of keys selected and an ``oob_default``
that sorts after them. The first kernel in
:ref:`Top-k selection <coop-cutlass-topk>` in the API reference does this
with ``topk_min_keys``. After ``topk_min_pairs`` or ``topk_max_pairs``, pass
both payloads to :func:`~cuda.coop.cutlass.merge_sort_pairs` in the same
way, and each value moves with its key.

.. _coop-faq-global-sort:

Does sorting each block sort the whole array?
---------------------------------------------

No. A merge sort or a radix sort orders the :term:`tile` of the group that
makes the call. A kernel that sorts every block leaves each block's tile in
order and the array as a whole unsorted. The package has no primitive that
merges sorted tiles. A merge sort also runs on a warp or a group of lanes,
and a radix sort runs on a block only.

:ref:`Comparison sorting <coop-cutlass-merge-sort>` in the API reference
says where the sorted keys end up across the threads. The
:doc:`Merge Sort <visualizations/merge-sort>` and
:doc:`radix sorting <visualizations/radix>` pages draw both sorts.

.. _coop-faq-neighbor-operations:

When should I use Adjacent Difference or Discontinuity?
-------------------------------------------------------

Use :func:`cuda.coop.cutlass.adjacent_difference` for a value computed from
each item and its neighbor, such as the change between successive samples.
Use :func:`cuda.coop.cutlass.discontinuity` for flags that mark where each
run of equal values starts and ends.
:ref:`Neighbor comparisons <coop-cutlass-neighbors>` in the API reference
sets the two side by side.

To continue a sequence across tiles, pass the last item of the previous tile
as ``tile_predecessor_item``, or the first item of the next tile as
``tile_successor_item``. Read that item from the input array and write
results to a separate array, so that the input keeps its values for another
tile or block to read.

.. _coop-faq-histogram-padding:

Can I zero-pad a partial Histogram tile?
----------------------------------------

No. ``histogram`` counts every item of every thread, so each padded zero
adds one to bin 0. Pad with the index of an extra bin, and leave that bin
out when you store the counts.

In this kernel ``load`` gives every item at or past ``count`` the value 70.
``histogram`` counts those items in bin 70, the last of its 71 bins, and
``store`` writes the counts of bins 0 to 69:

.. literalinclude:: ../../../python/cuda_coop_cutlass/tests/runtime/test_histogram_documentation.py
   :language: python
   :start-after: # cutlass-histogram-partial-tile-example-begin
   :end-before: # cutlass-histogram-partial-tile-example-end
   :dedent: 4

Without ``oob_default`` the items at or past ``count`` would hold undefined
values, and ``histogram`` would use them as bin indices.
:ref:`Histogram <coop-cutlass-histogram>` in the API reference has the
rules for each argument and for the result.

.. _coop-faq-histogram-accumulation:

Does Histogram retain counters between calls?
---------------------------------------------

No. Each call counts from zero, even when it shares a ``TempStorage`` with
earlier calls. To count several tiles, keep a payload of totals and add
each call's counters to it. Item ``i`` of a thread's counters stands for
the same bin in every call, so each thread adds item by item:

.. literalinclude:: ../../../python/cuda_coop_cutlass/tests/runtime/test_histogram_documentation.py
   :language: python
   :start-after: # cutlass-histogram-tiles-example-begin
   :end-before: # cutlass-histogram-tiles-example-end
   :dedent: 4

Here ``totals`` and the counters are ``Int32``. For totals past the range
of ``Int32``, call ``histogram`` with ``counter_dtype=cutlass.Int64`` and
give ``totals`` the same element type. CUB's ``Composite`` method, which
adds to counters the caller keeps, has no counterpart in the package.
:ref:`Histogram <coop-cutlass-histogram>` in the API reference has the
rules for each argument and for the result.

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
every thread. That avoids repeating length validation, the prefix scan,
and shared-table staging for every output window. The destination must
have enough remaining capacity; use separate source and destination storage.

For example, 94 decoded values with a 32-item window require three decode
iterations. One bulk call handles all three, writing only the 30 valid items
in the last window. The :ref:`complete example <coop-cutlass-run-length>`
compares this with an explicit loop over the windowed operation and checks
both results against ``numpy.repeat``.

Use the windowed operation when your kernel needs to process register values
between windows. The bulk operation stores each window directly; it has no
per-window callback or prepared decoder that can be carried between calls.

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

``reduce(group, values)`` combines every item of every thread into one
result, which the thread with rank 0 in the group holds.
``reduce_batched(warp, values)`` keeps the items apart: item ``b`` of every
lane goes into result ``b``, so three items per lane give three results.

Each result is in one lane, and a lane can end up with none, so a kernel
tests the batch number before it reads an item of the returned payload.
:ref:`Batched reduction <coop-cutlass-reduce-batched>` in the API reference
has the two output layouts and a tested kernel, and the
:doc:`explorer <visualizations/reduce-batched>` draws both layouts.

.. _coop-faq-collective-control-flow:

Can thread-local work surround a collective?
--------------------------------------------

Yes. A leader thread can prepare data before a collective and consume its
result afterwards. Every member of the group must reach the collective in the
same iteration, including members whose inputs are excluded by ``valid_items``.
A condition around the collective itself must be uniform across the group.
A block collective inside ``if thread == 0`` violates this requirement.

Use a runtime ``if`` statement for conditional collective calls. A runtime
conditional expression such as ``coop.sum(block, x) if condition else y``
is rejected when it needs temporary storage. Synchronize application data
before another thread reads it; the collective's scratch-reuse barrier does
not replace synchronization required by earlier application writes.

.. _coop-faq-scan-loop-aggregate:

How do I carry a scan aggregate across loop iterations?
-------------------------------------------------------

Create the ``aggregate_output`` payload inside the loop and read its item
there. A payload created outside the loop and only passed as
``aggregate_output`` is rejected when the kernel is compiled.

In this kernel ``tile_total`` is created on each trip, and
``total = total + tile_total[0]`` carries the running total to the next
trip:

.. literalinclude:: ../../../python/cuda_coop_cutlass/tests/runtime/test_scan_documentation.py
   :language: python
   :start-after: # cutlass-scan-running-total-example-begin
   :end-before: # cutlass-scan-running-total-example-end
   :dedent: 4

To keep the aggregate in a payload created before the loop, give that
payload a value of the aggregate's element type first, as in
``outer = coop.ThreadData.from_values(cutlass.Int32(0))`` for the ``Int32``
values here, and assign the item in the loop:
``outer[0] = tile_total[0]``. The same holds inside a runtime ``if``, and
for ``exclusive_digit_prefix`` of ``radix_rank_keys``.

The loop count or condition must be the same in every thread of the group.
Every thread receives the aggregate, whereas a ``reduce`` result is defined
at rank 0 only. The aggregate leaves out ``initial_value``.

.. _coop-faq-payload-conversion:

How do I change a payload's element type?
-----------------------------------------

Use ``ThreadData.from_fn(len(values), lambda i: values[i], dtype=cutlass.Float32)``
to create a converted payload. Supported NumPy scalar classes and ``np.dtype``
selectors name the same element types as their CUTLASS equivalents. ``int``
and ``float`` select ``Int32`` and ``Float32`` for runtime CuTe values.

.. code-block:: python

   # Inside a CuTe kernel; values is an initialized integer ThreadData.
   floating_values = coop.ThreadData.from_fn(
       len(values), lambda i: values[i], dtype=cutlass.Float32
   )

``ThreadData(..., dtype=..., values=...)`` and ``from_values`` declare the
type of existing items. Typed items must agree with that declaration;
changing metadata does not convert them. Export to a register tensor or
``TensorSSA`` checks the source declaration before applying an explicit
output dtype. Widening unsigned input preserves its unsigned value.
An explicit narrowing conversion can discard information, so choose the
destination width deliberately.

Carry a fixed item count, dtype, and alignment through runtime branches and
loops. Changing any of these properties at a runtime join is rejected.
Initialize loop-carried payloads with the alignment of the values that will
replace them. A CuTe "different structure" diagnostic can indicate a dtype or
alignment change. Normalize branch results before the join, or use a
compile-time condition when each alternative needs a different representation.

.. _coop-faq-storage-control-flow:

Can I choose temporary storage inside a branch or loop?
--------------------------------------------------------

Yes, as long as every path uses the same object. Construct one
``TempStorage`` inside the kernel before the ``if`` or loop. The Notes under
:class:`TempStorage <cuda.coop.cutlass.TempStorage>` list what the compiler
rejects.

.. _coop-faq-floating-prefix:

How do floating min/max and scan seeds handle NaN and infinity?
---------------------------------------------------------------

On ``Float32`` and ``Float64`` values, ``"min"`` and ``"max"`` follow
``fmin`` and ``fmax`` in reductions and in scans: a number wins over NaN.
A reduction returns NaN only when every value that goes into it is NaN.
``numpy.minimum`` and ``numpy.maximum`` select the same two operators.

An exclusive scan with ``"min"`` or ``"max"`` starts from its
``initial_value``. A plain Python float must be finite there, so pass a
typed infinity: ``numpy.float32(numpy.inf)`` for ``"min"`` and
``numpy.float32(-numpy.inf)`` for ``"max"`` on ``Float32`` values. A typed
NaN is accepted as well. :ref:`Reduction <coop-cutlass-reduce>` and
:ref:`Scan <coop-cutlass-scan>` in the API reference have the rules for
operators and initial values.

.. _coop-faq-compiler-setup:

Why does import work but kernel compilation fail?
-------------------------------------------------

Import validates the CuTe interfaces. Compilation additionally requires
matching CCCL headers, NVRTC, linker support, and a compatible toolkit.
Check the operation's launch shape, dtype, and participation requirements.
A successful host import alone does not establish GPU runtime qualification.

For provider compilation errors, verify the toolkit and CCCL header origins,
then use ``CUDA_COOP_CUTLASS_SOURCE_DUMP_DIR`` to inspect generated source. The
:doc:`Developer Guide <developer_guide>` explains compiler and linker
inspection, including why the final linked cubin is needed to assess calls,
barriers, spills, and resource use.
