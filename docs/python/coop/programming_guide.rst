.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _cuda.coop.programming_guide:

``cuda.coop`` Programming Guide
===============================

``cuda.coop`` lets threads cooperate inside a Python GPU kernel. You can
load a tile, operate on its per-thread values, and write the result
without leaving the kernel. You choose the participating group and the data
each thread contributes.

The :doc:`visualizations <visualizations/index>` show how values move through
these operations. Each explorer includes an example kernel and lets you
step through the algorithm. The :doc:`glossary <glossary>` defines terms and
layouts; the :doc:`FAQs <faqs>` explain common API choices.

This guide assumes you have written a CUDA kernel and know how threads,
blocks, and device arrays work. The examples use Numba-CUDA-MLIR. The
:doc:`installation instructions <../coop>` describe the matching
``cuda-coop`` extra; the current backend requires
``numba-cuda-mlir>=0.5.0,<0.6``. Check the
:ref:`validation scope <coop-numba-validation>` for environment coverage.
Keep a compiled kernel in its original CUDA context; see the
:ref:`device and context-lifetime limitation <coop-numba-context-lifetime>`
before reusing a dispatcher across devices or recreated contexts.

This guide describes the experimental Numba-CUDA-MLIR 0.5.x API.
Operation support varies by group and backend. The examples below use
supported block and warp operations.


A first kernel: copying a tile
--------------------------------

The :func:`cuda.coop.load` reference contains a complete, tested copy kernel,
including allocation, launch, partial-tile handling, and an output check.
All threads participate in Load and Store, while ``valid_items`` selects the
valid prefix of the final tile. The examples below use these imports:

.. code-block:: python

   import numpy as np
   from numba_cuda_mlir import cuda, types
   from cuda import coop

.. _coop-programming-api-choice:

Choosing the common or qualified API
------------------------------------

The common API is imported with:

.. code-block:: python

   from numba_cuda_mlir import cuda
   from cuda import coop

``cuda.coop`` expresses operations through a common vocabulary of groups,
numeric values, ``ThreadData``, and ``TempStorage``. The kernel
compiler uses its registered backend to implement those calls. Start here
when these operations cover your kernel's needs. Numba-CUDA-MLIR is the first
backend; CUTLASS support is planned.

The qualified import selects the Numba-CUDA-MLIR API explicitly:

.. code-block:: python

   import cuda.coop.numba_mlir as numba_coop

Both imports can be used in the same program. This guide uses ``coop`` for
common calls and ``numba_coop`` for qualified calls so the choice is visible.
In a program that uses only the qualified API, importing it as ``coop`` is
also fine.

The qualified API accepts Numba-specific payloads and adds controls to
several operations:

.. list-table::
   :header-rows: 1
   :widths: 24 34 42

   * - Need
     - Common API
     - Numba-CUDA-MLIR-qualified API
   * - Fixed data per thread
     - ``ThreadData``; scalar inputs where the operation accepts them
     - Also accepts fixed local arrays in supported array operations;
       exposes ``local`` and ``shared`` memory namespaces
   * - Layout exchange
     - Blocked-to-striped and striped-to-blocked conversion
     - Also supports block scatter and warp-striped layouts, with optional
       warp time slicing where supported
   * - Shifting values
     - Unit ``up`` and ``down`` shifts of a block's ``ThreadData`` tile
     - Also supports scalar block ``offset`` and ``rotate`` modes
   * - Load/Store algorithms and explicit scratch
     - String algorithm selectors and ``TempStorage`` on supported block calls
     - Same shared controls; qualifying the import is unnecessary for these

The common API defines backend-independent contracts. You must still
check that the selected backend implements the requested group, dtype, and
operation. The kernels here also contain Numba launch and indexing code;
porting the complete kernel to another DSL involves those parts too.

Registering the compiler backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Register explicitly on the host before compiling when imports may occur in
any order, as in a library or notebook:

.. code-block:: python

   from cuda import coop

   coop.register("numba-cuda-mlir")

   from numba_cuda_mlir import cuda

Repeated calls are safe. ``"numba_cuda_mlir"`` is also accepted. Importing
Numba-CUDA-MLIR before ``cuda.coop`` activates the backend automatically;
importing ``cuda.coop.numba_mlir as numba_coop`` does so explicitly and gives
you the backend namespace. See :ref:`backend registration
<coop-backend-registration>` for details.

These primitive calls belong inside kernels compiled by a compatible
backend. Registration itself is a host-side operation.

.. _coop-thread-groups:

Groups: which threads cooperate
-------------------------------


A group defines the participants in an operation. ``this_block()`` uses
all threads in the current block; ``this_warp()`` uses a complete physical
warp of 32 threads. ``this_warp().group_by(8)`` partitions it into four
consecutive logical warps of eight threads. The compiler obtains the block
shape from the kernel launch; the group factories take no size arguments.

Load and Store support blocks, physical warps, and logical warp widths of
1, 2, 4, 8, 16, or 32. Warp operations require a block size divisible by 32.
The partition width is a compile-time constant. Every member of a
participating group must reach the same primitive invocation.

The common API also declares other hierarchy descriptors. Their presence
does not imply executable operations or queries on those scopes in this
backend.


.. _coop-participation:

Participation and synchronization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every required participant must reach the same primitive invocation.
For a block operation, a branch around the call must be uniform across the
block. For a logical-warp operation, it must be uniform within that logical
warp. Complete sibling logical groups can follow different paths.

In particular, putting a block primitive inside ``if index < count``
breaks participation on a partial tile. Keep the primitive outside the
per-element condition. Use guarded loads or initialized values to handle
missing input, as in the first kernel. An early return by some block threads
has the same problem if the remaining threads later execute a block
primitive.

Use ``cuda.syncthreads()`` for explicit block synchronization in
Numba kernels. Every thread in the block must reach the barrier.

An operation's scratch-reuse barrier protects its temporary storage.
Automatic synchronization inserts a trailing barrier after each call that
consumes scratch. It does not prove that arbitrary user control flow is safe;
every member must still reach the primitive and its barrier.
Arrange synchronization for your own shared-memory communication as well.
Constructing a group or a ``ThreadData`` object does not synchronize threads.


.. _coop-thread-data:

``ThreadData``: the part of a tile owned by one thread
------------------------------------------------------

``coop.ThreadData(2, dtype=np.int32)`` gives each thread two integer slots.
With 128 threads, the group owns 256 values. Each thread
indexes its own slots with ``items[0]`` and ``items[1]``. Each thread accesses only its own slots.

:class:`~cuda.coop.ThreadDataLike` names the common payload interface used
in API signatures. It describes the item count, dtype, and indexed reads and
writes. Use :func:`~cuda.coop.ThreadData` to construct a payload for the active
compiler backend. Other payload representations require support from that
backend.

The item count must be a positive compile-time integer. You can use
``items.items_per_thread`` as a loop bound:

.. code-block:: python

   for i in range(items.items_per_thread):
       items[i] = types.int32(items[i] * 2)

Initialize every slot before reading it. Constructing ``ThreadData`` does
not fill it with zeros. A full Load initializes the entire payload; a
partial Load needs either an ``oob_default`` or previously initialized slots
for the missing elements.

.. _coop-data-layouts:

Blocked and striped order
^^^^^^^^^^^^^^^^^^^^^^^^^

The glossary defines :term:`blocked` and :term:`striped` ownership and
compares their :ref:`layouts <coop-glossary-layouts>`.

A primitive needs to know how the per-thread slots correspond to the
group's tile. Here is a small layout illustration with four threads and
two items each. The numbers are positions in the tile; this illustration
does not prescribe a supported CUDA launch size.

.. list-table::
   :header-rows: 1

   * - Thread rank
     - Blocked: ``items[0], items[1]``
     - Striped: ``items[0], items[1]``
   * - 0
     - 0, 1
     - 0, 4
   * - 1
     - 2, 3
     - 1, 5
   * - 2
     - 4, 5
     - 2, 6
   * - 3
     - 6, 7
     - 3, 7

For group size ``G``, items per thread ``K``, thread rank ``t``, and local
item index ``i``, blocked order uses tile position ``t * K + i``.
Striped order uses ``t + i * G``.


The payload does not carry a runtime layout tag that corrects mismatched
operations. The algorithms you call determine the interpretation. Pairing
a striped Load with a direct Store without a conversion permutes the data.

Compare the :doc:`Load <visualizations/load>` and
:doc:`Store <visualizations/store>` visualizations to see how each algorithm
maps memory positions to per-thread slots.

Dtypes and storage
^^^^^^^^^^^^^^^^^^

The current numeric payload types are signed and unsigned integers of 8,
16, 32, or 64 bits, and 32- or 64-bit floating point. Boolean, half
precision, complex, and structured payloads are outside this contract.

You may omit ``dtype`` when surrounding operations establish it:
``items = coop.ThreadData(2)`` followed by Load infers the source dtype.
If you initialize the payload yourself, specifying a dtype usually makes
the code easier to follow. Conflicting dtype requirements are errors.

Load writes into the payload supplied by the caller. Store preserves its
input. Both return ``None``. Exchange and array Shuffle return fresh payloads,
so their input values remain available afterwards.

Numba can promote integer arithmetic. Store requires an exact match to the
destination dtype, so cast computed values when necessary, as in the
``types.int32`` assignment above. Accumulation dtype also matters: an
integer sum can overflow, and parallel floating-point sums can differ from
a sequential CPU sum because of evaluation order.

The compiler can keep payload elements in registers. Indexing patterns,
address-taking, and register pressure influence the final placement.
Increasing the item count increases the amount of live data per thread.

The optional ``alignment`` keyword requests a minimum power-of-two alignment
in bytes when the compiler materializes the payload. For example,
``coop.ThreadData(4, dtype=np.float32, alignment=16)`` requests at least
16-byte alignment. The compiler may strengthen it. This setting applies
to payload storage; alignment of the input and output arrays remains a
separate property.

Load, operate, store
--------------------

Load and Store accept one-dimensional contiguous arrays. ``offset`` counts
elements from the array's beginning. ``valid_items`` counts elements in
the valid prefix of the selected group's tile. Both must be uniform within
that group.

For a block, a tile holds ``block_threads * items_per_thread`` elements.
For a physical or logical warp, it holds
``warp_width * items_per_thread`` elements. Clamp the valid count to that
range before calling Load or Store. An out-of-range runtime count causes
a device trap and invalidates the CUDA context.

Load leaves invalid slots unchanged unless you pass ``oob_default``.
Store leaves destination elements outside the valid prefix untouched.
Supply an operation-appropriate identity when processing padded data:
zero for sum, one for multiplication, and a suitable upper or lower bound
for minimum or maximum. For a runtime ``oob_default``, use exactly the
payload dtype and the same value across the group.

Algorithm choices
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 20 35 20

   * - Algorithm
     - Payload order
     - Memory access and rearrangement
     - Groups
   * - ``"direct"``
     - Blocked
     - Each thread accesses its consecutive items
     - Block and Warp
   * - ``"striped"``
     - Striped
     - Neighboring threads access neighboring elements at each item index
     - Block and Warp
   * - ``"vectorize"``
     - Blocked
     - Uses vector accesses where the specialization and alignment allow
     - Block and Warp
   * - ``"transpose"``
     - Blocked
     - Uses striped accesses and shared scratch to rearrange values
     - Block and Warp
   * - ``"warp_transpose"``
     - Blocked
     - Rearranges through warp-sized portions of the block tile
     - Block
   * - ``"warp_transpose_timesliced"``
     - Blocked
     - Reuses scratch across those portions
     - Block

The two block warp-transpose variants require a block size divisible by
32. Use string selectors. Begin with ``direct`` for a simple kernel;
measure alternatives with the actual dtype, item count, and surrounding
work. Coalesced accesses can justify rearrangement costs, but the best
choice depends on the kernel.

An explicit layout conversion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Exchange converts per-thread values between blocked and striped layouts.

The common Exchange API also supports ``blocked_to_striped``. Qualified
block scatter modes let you supply destination ranks for finer control.
For scatter, valid ranks and unique active destinations are caller
requirements; duplicate destinations and holes leave unspecified slots.

The :doc:`Exchange visualization <visualizations/exchange>` shows both
layout conversions and ranked scatters, including the holes left by
suppressed writes.

Shuffle operates on a block's flattened blocked tile. The common ``up``
and ``down`` modes shift it by one element and return a fresh payload.
The first ``up`` slot or last ``down`` slot is unspecified. Set that
boundary yourself before consuming it. Qualified scalar ``offset`` and
``rotate`` modes have different distance rules; see :doc:`../coop_api`
before substituting them for an array shift.

Use the :doc:`Shuffle visualization <visualizations/shuffle>` to compare
the array shifts with scalar offsets and rotation.

Warp tile addresses
^^^^^^^^^^^^^^^^^^^

Warp Load and Store automatically add the group's origin within the
block. For width ``G`` and ``K`` items per thread, this origin is
``(linear_thread_rank // G) * G * K``. The explicit ``offset`` is added
after that origin. In a multi-block traversal, pass the block's global
tile origin as ``offset``.

The valid count still belongs to each individual group. Compute it using
both the block origin and the group's origin:

.. literalinclude:: ../../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_programming_guide_examples.py
   :language: python
   :name: coop-pg-warp-copy
   :start-after: # coop-pg-warp-copy-begin
   :end-before: # coop-pg-warp-copy-end
   :dedent: 4

Adding ``group_origin`` to ``offset`` here would count it twice. This
automatic origin applies to Warp Load and Store; ordinary array indexing
in a kernel uses exactly the index you write.

.. _coop-temp-storage:

.. _tempstorage-scratch-used-during-a-collective:

``TempStorage``: scratch used during a primitive
------------------------------------------------

Some algorithms exchange intermediate values through shared memory.
By default, ``cuda.coop`` allocates the scratch they need and inserts
the required reuse barrier. Start with that behavior.

:class:`~cuda.coop.TempStorageLike` names the common interface for explicit
scratch descriptors in API signatures. Construct one inside the kernel with
:func:`~cuda.coop.TempStorage`. Its ``size_in_bytes``, ``alignment``,
``auto_sync``, and ``sharing`` properties control capacity, alignment,
synchronization, and allocation sharing. The active compiler backend must
recognize the descriptor.

An explicit descriptor lets several supported block calls reuse an
allocation. Its contents are opaque; keep application values in
``ThreadData`` or your own arrays.

.. list-table::
   :header-rows: 1

   * - Calls
     - Scratch behavior in the current backend
   * - Direct, striped, or vectorize Load/Store
     - No shared scratch or reuse barrier
   * - Block transpose-family Load/Store
     - Automatic scratch, or an explicit ``TempStorage``
   * - Warp transpose Load/Store
     - Automatic scratch per group; explicit descriptors are rejected
   * - Exchange and Shuffle
     - Compiler-owned scratch and reuse synchronization

Capacity, alignment, and lifetime
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``TempStorage()`` defaults to ``sharing="shared"`` and ``auto_sync=False``,
leaving reuse synchronization to the caller. ``size_in_bytes=None`` and
``alignment=None`` let the compiler determine the requirements. An explicit
capacity must be large enough for the operations using it. An explicit
``alignment`` requests a minimum positive power of two in bytes; the compiler
can strengthen it.
Only ``size_in_bytes`` may be positional. The other options are keyword-only.

``sharing="exclusive"`` gives distinct call sites separate slices. A loop can reach
the same call site again and reuse its slice, so exclusive storage still needs
reuse barriers. It can also consume more shared memory when several calls could
otherwise share a slice. ``sharing`` controls allocation layout independently
of ``auto_sync``.

With the default ``auto_sync=False``, the kernel must provide reuse barriers.
Put an explicit block barrier after each storage-using call and before
reusing scratch, including between loop iterations.

Set ``auto_sync=True`` to insert automatic trailing barriers for scratch reuse.
Without an explicit descriptor, the compiler synchronizes scratch automatically.
An unrelated memory access between calls does not establish a block barrier.
The MVP conservatively rejects merging multiple manually synchronized
``TempStorage`` constructors into one descriptor, including conditional
definitions. Use one constructor and keep its synchronization explicit. This
restriction reflects what the planner can establish; it does not mean that
every rejected program necessarily races.

Scratch lasts for the kernel's execution on that block. It cannot carry
state between blocks or kernel launches. Keep persistent application state in a separate payload.

When the combined scratch requirement exceeds the default static shared-memory
limit, the backend can use dynamic shared memory, subject to the GPU's opt-in
limit. Numba-CUDA-MLIR automatically includes those required bytes in the
launch configuration. You do not need to copy a compiler-reported byte count
into the launch yourself. Requirements above the device limit are rejected.
The required dynamic byte count is a minimum: launch-time shared bytes are not
added to it as a separate allocation. Dynamic cooperative backing supports
alignment requirements up to 16 bytes.

With the currently supported compiler, do not combine user dynamic or
runtime-sized ``cuda.shared.array`` allocations with cooperative scratch.
User static shared arrays may coexist with static cooperative backing, but are
rejected when the cooperative backing becomes dynamic, including an implicit
oversized allocation. These checks apply after helper inlining. Keep both
allocations static within the device limit, use global memory for the user
buffer, or separate the work into kernels. Storage-free operations do not
create this conflict. The compatibility restrictions remain until a released
compiler with the shared-memory fix has passed the coexistence tests.

Extra shared memory can reduce resident blocks per multiprocessor. The
default inferred allocation and unsized shared descriptor are sufficient
for the kernels above; use an explicit capacity when you have a reason to
reserve that amount of shared memory.

Helpers and compile-time values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Device helpers containing primitives must be inlined into the kernel so the
planner can resolve their groups, descriptors, and launch dimensions. Default
helper inlining is supported. A surviving non-inlined primitive helper, a
descriptor escaping through a runtime object, or a primitive inside a
standalone callback receives a compilation error. Move the primitive into the
kernel or an inlined helper; callbacks may perform ordinary device computation.

The MVP does not support ``literal_unroll`` values that determine cooperative
groups, operation selectors, or payload and storage shapes. Write the affected
calls explicitly with compile-time constants. Ordinary runtime loops with fixed
cooperative shapes, and unrelated uses of ``literal_unroll``, remain supported.


Checking and tuning a kernel
----------------------------

Check results before comparing algorithms. Useful cases include one full
tile, several tiles, a single valid element in the final tile, and an empty
input handled on the host. Layout conversions are easier to inspect with
distinct input values. Compare computed values against a CPU reference with a suitable
accumulation dtype and floating-point tolerance.

Keep launch dimensions, logical-warp widths, payload extents, and algorithm
choices consistent with the code. The compiler specializes group operations
using these facts. A runtime tile offset or valid count can change from
one call to the next; the size of a ``ThreadData`` payload must be known
at compile time.

Warm up the kernel before timing it, use device-resident arrays, and account
for asynchronous execution with CUDA events or explicit synchronization.
Then vary one choice at a time: threads per block, items per thread, or a
Load/Store algorithm. More items per thread can amortize primitive
work while increasing register pressure. Scratch-heavy choices consume
shared memory and can reduce occupancy. Measure the complete kernel,
including conversions and synchronization.

If compilation fails, check the group/operation combination, payload dtype,
static parameters, and import order first. The :doc:`API reference
<../coop_api>` records exact signatures; the :doc:`overview <../coop>`
collects operation restrictions and configuration. For generated-source
diagnostics and the compiler integration, see the
:doc:`Developer Overview <developer_overview>`.
