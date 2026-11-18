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

Install the ``cuda-coop-cutlass`` package, as :doc:`Installation <index>`
shows, and import ``cuda.coop.cutlass``:

.. code-block:: python

   import cuda.coop.cutlass as coop

The whole public API is in this one module, and importing it is the only
setup. Keep the module's name in front of each call, as the examples do with
``coop``. The package has a function named ``sum``, and
``from cuda.coop.cutlass import sum`` replaces Python's built-in ``sum`` in
the importing module.

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

A new ``ThreadData`` in each thread of the group, with ``items_per_thread``
items of the source's element type. In CUB, ``Load`` fills an array that
the caller declares. Here every call returns a payload of its own.

``items_per_thread=1`` gives a payload of one item. A primitive that takes
one scalar per thread, such as a warp scan, takes ``values[0]``.
:func:`~cuda.coop.cutlass.load` in the API reference has the rules for
``offset``, ``valid_items``, ``oob_default`` and ``alignment``.

.. _coop-faq-shared-load:

How do I stage a tile in shared memory?
---------------------------------------

Write the tile to a shared tensor, wait at a barrier, and read it back with
``load``. The tensor comes from CuTe's ``SmemAllocator`` or from
:meth:`TempStorage.reserve <cuda.coop.cutlass.TempStorage.reserve>`. The
kernel fills it from a payload with ``store``, or straight from global
memory with native CuTe copies. The barrier, such as ``block.sync()``, is
what lets ``load`` give a thread elements that another thread wrote.

:ref:`Staging a tile in shared memory <coop-shared-load-store>` in Advanced
topics has a complete kernel that stages a tile with a native copy, and the
rules for copies with predicates and for asynchronous copies.

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
of range, or a float in an integer payload, is rejected when the kernel is
compiled. An item that already has a CuTe type is not converted and must
have exactly that type. ``ThreadData.from_fn`` with ``dtype`` converts, as
in :ref:`How do I change a payload's element type?
<coop-faq-payload-conversion>`. :ref:`ThreadData <coop-thread-data>` in the
API reference lists the supported types.

.. _coop-faq-producer:

Who chooses a producer's memory copy and synchronization?
---------------------------------------------------------

The producer does. A producer is an object that brings its own data into
registers, such as one that holds a tensor in tensor memory (TMEM).
``ThreadData.load(source)`` calls the ``__cuda_coop_thread_data_load__``
method of ``source`` once, while the kernel is compiled, and makes a
payload from the register values the method returns. The method chooses the
copy and makes sure the data is ready. ``ThreadData.load`` itself reads no
memory and waits at no barrier.

An object without the method is rejected when the kernel is compiled, a bare
tensor in tensor memory included. :ref:`Producer-owned register loads
<coop-cutlass-producer-interop>` in Advanced topics has a tested producer
and the rules for the method and for what it returns.

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

Leaving ``temp_storage`` out adds a barrier after each call that uses
scratch. The compiler picks the layout, which today is one region per call.

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

Use :func:`~cuda.coop.cutlass.adjacent_difference` for a value computed
from each item and its neighbor, such as the change between successive
samples. Use :func:`~cuda.coop.cutlass.discontinuity` for flags that mark
where each run of equal values starts and ends.
:ref:`Neighbor comparisons <coop-cutlass-neighbors>` in the API reference
sets the two side by side.

To continue a sequence across tiles, pass the last item of the previous tile
as ``tile_predecessor_item``, or the first item of the next tile as
``tile_successor_item``. Read that item from the input array and write
results to a separate array, so that the input keeps its values for another
tile or block to read. The second example under
:func:`~cuda.coop.cutlass.adjacent_difference` does this in a loop over
tiles.

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

:func:`~cuda.coop.cutlass.run_length_decode` returns one window of the
decoded stream as a payload, for the kernel to use.
:func:`~cuda.coop.cutlass.run_length_decode_into` writes the whole stream to
a tensor and returns the length of the stream to every thread.

Every call checks the run lengths and finds where each run starts before it
decodes, and nothing is kept from one call to the next. A stream of 94 items
in windows of 32 takes three ``run_length_decode`` calls, each with that
work. One ``run_length_decode_into`` call does the work once and writes the
three windows itself.

:ref:`Run Length Decode <coop-cutlass-run-length>` in the API reference sets
the two functions side by side, and the example under
``run_length_decode_into`` is this comparison as a tested kernel. The
:doc:`explorer <visualizations/run-length-decode>` steps through both calls,
and :ref:`run positions and windows <coop-glossary-decoding>` in the
glossary follows one stream through a window.

.. _coop-faq-rld-padding:

How do I pad run inputs and recognize the end of decoded output?
----------------------------------------------------------------

End the block's run lengths with zeros. With the runs taken thread by thread
across the block, every positive length comes before every zero. The value
paired with a zero length never appears in the output. A block whose lengths
are all zero decodes to nothing.

:func:`~cuda.coop.cutlass.run_length_decode` puts zero in every item past
the end of the stream. Zero can also be a run value, so compare an item's
position with the length of the stream. ``run_length_decode`` does not
return that length, and the kernel works it out from the run lengths.
:func:`~cuda.coop.cutlass.run_length_decode_into` returns the length to
every thread and leaves the elements after the stream as they were.
:ref:`Run Length Decode <coop-cutlass-run-length>` in the API reference has
the rules for lengths and offsets.

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

Can I call a primitive inside an ``if`` or a loop?
--------------------------------------------------

Yes, when every thread of the group makes the call. An ``if`` around the
call must have the same condition in every thread of the group, and a loop
around it the same number of trips. Threads whose data is at or past a
``valid_items`` count make the call too. For a call on a warp or a group of
lanes, the group is that warp or those lanes, and the block's other warps
or groups can take another path. A block primitive inside
``if block.rank() == 0:`` breaks the rule. The package does not check it,
so that kernel still compiles. :ref:`Thread groups <coop-thread-groups>` in
the API reference has the rules that every primitive shares.

Work for some of the threads goes before or after the call, in an ``if`` of
its own. This fragment makes one group of 24 lanes in each warp. The outer
condition is true in every lane of the group, so the whole group calls
``sum``, and the eight lanes outside the group skip the call. The inner
``if`` lets the lane with rank 0 store the result:

.. literalinclude:: ../../../python/cuda_coop_cutlass/tests/runtime/test_concepts_documentation.py
   :language: python
   :start-after: # cutlass-concept-group-by-partial-begin
   :end-before: # cutlass-concept-group-by-partial-end
   :dedent: 4

Write the condition as an ``if`` statement. Inside a runtime
``a if c else b`` expression, a block primitive that takes ``temp_storage``
and uses scratch is rejected when the kernel is compiled. For a loop, the
kernel under :ref:`How do I carry a scan aggregate across loop iterations?
<coop-faq-scan-loop-aggregate>` calls ``load``, ``exclusive_scan`` and
``store`` on every trip.

.. _coop-faq-blocks:

How do I process more than one block?
-------------------------------------

Launch a grid of blocks and compute the ``offset`` of ``load`` and ``store``
from ``cute.arch.block_idx()``. ``load`` and ``store`` never add the block
index, so blocks that pass the same ``offset`` read and write the same
:term:`tile`. In a launch with ``grid=3, block=128``,
``cute.arch.block_idx()[0] * 128 * items_per_thread`` starts each block's
tile after the tiles of the blocks before it.
:ref:`More than one block <coop-cutlass-blocks>` in the Programming Guide
has the complete kernel.

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

Build a new payload with
:meth:`ThreadData.from_fn <cuda.coop.cutlass.ThreadData.from_fn>` and name
the new type as ``dtype``:

.. code-block:: python

   # Inside a kernel. values is a payload of Int32 items.
   as_floats = coop.ThreadData.from_fn(
       len(values), lambda i: values[i], dtype=cutlass.Float32
   )

Each item is cast as ``cutlass.Float32(values[i])`` casts it. ``dtype`` is
a CUTLASS type, the matching NumPy type or a NumPy ``dtype`` such as
``numpy.dtype("float32")``, or Python ``int`` or ``float``, which mean
``Int32`` and ``Float32``. The conversions between a payload and CuTe's
register forms take ``dtype`` as well and cast the items they copy:
``from_register_tensor`` and ``from_vector`` on the way in,
``to_tensor_ssa`` and ``to_register_tensor`` on the way out. An unsigned
item keeps its value when the new type is wider.

``dtype`` on ``ThreadData(...)`` or ``from_values`` states the type the
items have, and converts no item that already has a CuTe type.
``to_tensor_ssa`` and ``to_register_tensor`` check the items against that
type before they cast. A payload declared ``Int32`` that holds a
``Float32`` item is rejected when the kernel is compiled, whatever ``dtype``
the call names. :ref:`When does ThreadData need an explicit element type?
<coop-faq-thread-data-dtype>` has those rules.

A variable that holds a payload keeps one item count, element type and
alignment through a runtime ``if`` or loop. Inside one, assigning the
variable a payload that differs in any of the three is rejected when the
kernel is compiled, and CuTe can report the change as a container whose
structure differs. Convert before the ``if`` or loop, or to the same type
on every path. When a loop assigns the variable a new payload, give the
variable a payload with the same element type and alignment before the
loop. To choose a type by a Python value, write
``if cutlass.const_expr(flag):``, which is decided when the kernel is
compiled.

.. _coop-faq-storage-control-flow:

Can I choose temporary storage inside a branch or loop?
-------------------------------------------------------

Yes, as long as every path uses the same object. Construct one
``TempStorage`` inside the kernel before the ``if`` or loop. The Notes under
:class:`TempStorage <cuda.coop.cutlass.TempStorage>` list what the compiler
rejects.

.. _coop-faq-floating-prefix:

How do ``"min"`` and ``"max"`` treat NaN, and can a scan start from infinity?
-----------------------------------------------------------------------------

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

The import checks one thing: that the installed CuTe DSL has the
:ref:`interfaces the package needs <coop-cutlass-compiler-requirements>`.
The rest is checked when a kernel is compiled, by ``cute.compile`` or by a
call of the kernel's launcher:

* The CUDA toolkit. NVRTC must report the same version as the CUDA headers
  the package found, and nvJitLink the same major version and a minor
  version at least as new.
* The CCCL headers. The wheel includes them.
  ``CUDA_COOP_CUTLASS_CCCL_ROOT`` selects others, and a root without them
  is rejected.
* The kernel. Each primitive checks its group, its element types and the
  block size of the launch. :ref:`Thread groups <coop-thread-groups>` in
  the :doc:`API reference <api>` has the rules that every primitive shares,
  and each entry there has its own.

:ref:`Requirements <coop-cutlass-requirements>` lists the versions to
install, and :doc:`configuration` lists the variables that say where the
toolkit and the headers are. When NVRTC rejects the generated C++, set
``CUDA_COOP_CUTLASS_SOURCE_DUMP_DIR`` to a directory before compiling and
read the source the package writes there. The
:doc:`Developer Guide <developer_guide>` follows a compilation step by
step.
