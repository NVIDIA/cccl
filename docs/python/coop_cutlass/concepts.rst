.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-programming-concepts:

Programming concepts
====================

Cooperative operations let a group of threads work on data together. This
page explains how threads divide a tile, which threads must participate,
and how to use results and temporary storage. The :doc:`Programming Guide
<programming_guide>` provides complete CuTe kernels and launch examples.

.. _coop-api-namespaces:
.. _kernel-api:

Kernel API
----------

Use ``import cuda.coop.cutlass as coop``. Calls inside a CuTe kernel are
compiler markers; compilation turns them into device operations. The
``cuda.coop`` parent is a namespace package. The standalone distribution
installs its public operations only under ``cuda.coop.cutlass``.

Built-in operators and their documented aliases are supported. Custom
operators and stateful Scan callbacks are currently unsupported.

.. _coop-common-calling-conventions:

Calling conventions
-------------------

A primitive's group and input/output operands precede ``/`` in its
signature and are passed positionally. Controls after ``*`` are keyword-only:

.. code-block:: python

   items = coop.load(
       group, source, items_per_thread=items_per_thread,
       valid_items=count, offset=offset,
   )
   result = coop.reduce(group, items, binary_op="max")

Load always returns a fresh ``ThreadData`` payload. An operation that
returns a new value leaves its input payload unchanged unless its contract
says otherwise. Check :doc:`api` for each operation's accepted operands,
controls, and ownership rules.

Shared execution model
----------------------

.. _coop-common-groups:
.. _coop-thread-groups:
.. _groups-and-thread-data:

Groups and tiles
^^^^^^^^^^^^^^^^

A group identifies the threads working together. ``this_block()`` selects
the current block; ``this_warp()`` selects a physical 32-thread warp.
``this_warp().group_by(width)`` partitions it into consecutive logical
warps of 1, 2, 4, 8, 16, or 32 threads. A group processing ``K`` items per
thread has a tile of ``group_size * K`` items. Each block or warp group
processes its own tile.

The hierarchy also describes individual threads, clusters, the grid, and mapped
groups of physical warps. These descriptors support queries such as rank,
count, and membership. Their availability does not make every primitive valid
for that group: Reduce, Load/Store, and Scan use block or warp groups, and grid
primitives are unavailable. Consult the backend guide for the supported query
levels and synchronization operations.

A multidimensional block uses x-major linear thread rank:
``x + block_x * (y + block_y * z)``. Warp primitives require an enclosing
block made of complete physical warps. For warp Load/Store, the compiler
adds the group's within-block tile origin to the memory address. A caller's
``offset`` supplies the remaining element offset, such as the block's tile
origin in a larger array; do not add the within-block group origin again.

.. _coop-common-participation:
.. _coop-participation:
.. _load-and-store-semantics:

Participation and valid prefixes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every required participant must reach the same primitive invocation.
A short tile does not reduce the number of threads that must participate.
For example, ``valid_items=45`` limits a 64-item Load to its first 45 tile
positions; all threads in the group still call Load. Complete sibling
logical warps may take different paths where the operation's contract
permits it.

For Load and Store, ``valid_items`` counts tile elements, not items per
thread. It must be uniform within the group and lie between zero and the tile
capacity. Clamp a tail count before passing it. Invalid static counts are
rejected during compilation; invalid runtime counts trap on the device.
``offset`` is a nonnegative element offset, also uniform within the group.
It is independent of the valid count. Other families define their own
valid-prefix controls; for example, reduction counts contributing threads.

Load leaves invalid payload slots unspecified unless ``oob_default`` is
provided, even if those slots were initialized before Load. Store leaves
destination elements outside the valid prefix untouched. A valid prefix
controls data access; it does not make it safe to skip a required participant
or a reuse barrier.

.. _coop-common-payloads:

Per-thread payloads
^^^^^^^^^^^^^^^^^^^

Prefer ``coop.load(group, source, items_per_thread=items_per_thread)`` for each
participating thread's fixed-size ``ThreadData`` payload. Its element type
comes from ``source``. Pass the positive compile-time count as a kernel
argument declared ``items_per_thread: cutlass.Constexpr`` in both the CuTe
kernel and its launcher. A count of one still returns ``ThreadData``.

Use ``coop.ThreadData(items_per_thread)`` to construct a payload for computed
values or writable auxiliary outputs. Dtype-free construction remains
supported: initialize every consumed item with a compatible typed value.
Load's ``items_per_thread`` and ``alignment`` are keyword-only controls.

Global and shared arrays can supply Load's source and receive Store's output.
Those arrays hold a group tile; a ``ThreadData`` holds one thread's values.
Use native CuTe copies to stage a tile directly in shared memory, then
synchronize before another thread consumes it. See
:ref:`shared-memory staging <coop-shared-load-store>`.

Supported payload dtypes are signed and unsigned 8-, 16-, 32-, and 64-bit
integers and 32- and 64-bit floating-point values. An explicit alignment is
a positive power of two in bytes and sets a minimum when payload storage
is materialized. It does not assert alignment of an input or output array.
For example,
``coop.load(group, source, items_per_thread=items_per_thread, alignment=16)``
requests at least 16-byte alignment for the returned payload, as does
``coop.ThreadData(items_per_thread, alignment=16)`` for a directly constructed
payload. ``None`` lets the compiler choose; it may use stronger alignment.

The payload extent is a compile-time value available as
``items.items_per_thread``. Directly constructed slots start uninitialized.
Initialize every item an operation will read; Load without
``oob_default`` leaves invalid slots unspecified. Assign those slots after
Load before reading them, or supply a default. As in CUB, transpose Store
algorithms may rearrange the input payload in place. Copy values before
Store if they are needed later.

Typing protocols describe the supported payload interfaces; protocol
compatibility alone does not make an arbitrary Python object a kernel value.
CuTe register tensors and immutable ``TensorSSA`` values have explicit
adapters. Typed assignments and supported producers can establish the element
type. See :ref:`element-type inference <coop-faq-thread-data-dtype>` and
:ref:`register and producer conversions <coop-cutlass-register-payloads>`
for ownership, lifetime, and runtime-control-flow rules.

.. _coop-common-layouts:
.. _exchange-semantics:
.. _shuffle-semantics:
.. _scan-semantics:

Layouts and operation order
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In a :term:`blocked` layout, each thread owns consecutive tile positions.
In a :term:`striped` layout, consecutive threads own consecutive positions
at each item slot. The :ref:`layout glossary <coop-glossary-layouts>` gives
the formulas and an ownership table.

Load/Store's ``direct`` and ``vectorize`` algorithms expose blocked
payloads. ``striped`` exposes striped payloads. Transpose algorithms use
striped memory accesses and present blocked payloads to the caller.
Exchange converts between supported layouts. ``ThreadData`` does not carry
a layout tag that automatically fixes a mismatched Load/Store pair.
Shuffle shifts values within a block's tile. The
:doc:`Exchange <visualizations/exchange>` and
:doc:`Shuffle <visualizations/shuffle>` visualizations show the
supported rearrangements.

Scan traverses the group's values in blocked tile order. A striped Load
must therefore be converted before a Scan intended to follow the array's
original order. Choosing a memory-access algorithm and choosing the order
of values seen by an operation are related but distinct decisions.

.. _coop-common-results:

Result ownership
^^^^^^^^^^^^^^^^

A primitive's return type does not say which threads may use its result.
Reduce/Sum define their result only at group rank zero. Scan returns each
thread's part of the prefix sequence. Check the operation's group and algorithm
contract before consuming a result.

Sorting and selection operate on one group's tile. Sorting each block does not
sort a whole array. TopK defines an unordered selected prefix; the remaining
payload positions are not output. See the :doc:`API reference <api>`
for sorting and selection operations.

.. _coop-common-storage:
.. _coop-temp-storage:
.. _temporary-storage:

Scratch allocation and reuse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The compiler allocates temporary shared storage for operations that need
it. Direct, striped, and vectorized Load/Store are storage-free: they add
no scratch pointer, shared allocation, or reuse barrier. Other algorithms
use the concrete CUB specialization's required size and alignment.

For operations that accept an explicit descriptor, construct ``TempStorage``
inside the kernel. A descriptor lets several calls reuse one region or request
capacity and minimum alignment. An undersized request is an error.
``sharing="shared"`` overlaps the uses of one descriptor;
``sharing="exclusive"`` assigns distinct call sites separate slices. With
automatic synchronization disabled, separate slices avoid barriers needed
solely for cross-call scratch reuse, at the cost of more shared memory.
Independent descriptors do not alias. See :ref:`the exclusive-storage FAQ
<coop-faq-exclusive-storage>`.

Explicit descriptors default to ``auto_sync=False`` for both sharing modes.
The kernel must provide the required barrier before reuse, including across
iterations: a call site inside a loop reuses its slice even with exclusive
storage. Set ``auto_sync=True`` to insert a trailing reuse barrier after each
storage-using call. Compiler-managed scratch, used when no descriptor is
supplied, synchronizes automatically. A scratch reuse barrier does not replace
synchronization for the kernel's own shared data.

Warp operations that need scratch keep independent storage per physical or
logical group and use the appropriate warp mask. Each primitive documents
whether it accepts explicit storage. See :ref:`CUTLASS storage
<coop-cutlass-storage>` and :ref:`the storage FAQ <coop-faq-temp-storage>`
for examples and limits.

``reserve(num_elems, dtype, alignment=...)`` returns a typed shared view
from a descriptor. Shared policy may alias primitive scratch; exclusive policy
assigns slices. A reservation is application data, so a scratch-reuse barrier
alone does not establish that another thread can consume its contents.

``release()`` consumes the descriptor and its aliases on the executing path
and exposes the entire existing region, including padding and exclusive
slices. It inserts no barrier or wait. Finish outstanding uses and synchronize
as required before release; subsequent descriptor or stale-alias use is an
error. Conditional and repeated execution must respect the same lifetime.
