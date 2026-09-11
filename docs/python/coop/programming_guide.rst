.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _cuda.coop.programming_guide:

``cuda.coop`` Programming Guide
===============================

``cuda.coop`` lets threads cooperate inside a Python GPU kernel. You can
load a tile, compute a prefix sum across its elements, and write the result
without leaving the kernel. You choose the participating group and the data
each thread contributes.

This guide assumes you have written a CUDA kernel and know how threads,
blocks, and device arrays work. The examples use Numba-CUDA-MLIR. The
:doc:`installation instructions <../coop>` describe the matching
``cuda-coop`` extra; the current backend requires
``numba-cuda-mlir>=0.5.0,<0.6``.

*This guide describes the experimental Numba-CUDA-MLIR API in the current
PR stack. Operation support varies by group and backend. The examples below
use block and warp operations supported by that stack.*

A first kernel: prefix sums within tiles
---------------------------------------

A prefix sum gives each element the sum of the elements before it. For an
exclusive sum of ``[3, 1, 4, 2]``, the result is ``[0, 3, 4, 8]``.
The kernel below computes a separate exclusive sum for each tile of 256
elements. Each block has 128 threads, with two elements per thread.

.. code-block:: python
   :name: coop-pg-first-kernel

   import numpy as np
   from numba_cuda_mlir import cuda, types

   from cuda import coop


   @cuda.jit
   def scan_tiles(source, destination, count):
       block = coop.this_block()
       items = coop.ThreadData(2, dtype=np.int32)
       tile_size = cuda.blockDim.x * 2
       offset = cuda.blockIdx.x * tile_size
       valid = min(max(count - offset, 0), tile_size)

       coop.load(
           block,
           source,
           items,
           offset=offset,
           valid_items=valid,
           oob_default=0,
       )
       prefixes = coop.exclusive_sum(block, items)
       coop.store(
           block, destination, prefixes, offset=offset, valid_items=valid
       )


   source = (np.arange(785) % 7).astype(np.int32)
   d_source = cuda.to_device(source)
   d_destination = cuda.device_array_like(d_source)
   blocks = (source.size + 255) // 256
   scan_tiles[blocks, 128](d_source, d_destination, source.size)
   result = d_destination.copy_to_host()

   expected = np.empty_like(source)
   for start in range(0, source.size, 256):
       tile = source[start : start + 256]
       expected[start : start + tile.size] = np.cumsum(tile) - tile
   np.testing.assert_array_equal(result, expected)

All threads in a block execute the Load, Scan, and Store. The final block
still launches 128 threads even though its tile has only 17 valid elements.
Load fills the unused slots with zero, and Store writes only the valid
prefix. Zero contributes nothing to the sum.

Each block starts its sum at zero. For one prefix sum spanning the entire
array, you also need to carry the totals between tiles. A later example
does that while one block processes successive tiles. A scan distributed
across independently scheduled blocks needs a device-wide algorithm.

The host code copies input to the GPU once and copies the result back for
checking. Keep arrays on the device when several kernels use them. The
first launch also compiles the kernel; allow for that when measuring time.
For empty input, skip the launch.

The remaining examples reuse the imports above. Each kernel example includes
its own input and result check.

.. _coop-programming-api-choice:

Choosing the common or qualified API
-----------------------------------

The common API is imported with:

.. code-block:: python

   from numba_cuda_mlir import cuda
   from cuda import coop

``cuda.coop`` expresses operations through a common vocabulary of groups,
numeric values, ``ThreadData``, and ``TempStorage``. The active kernel
compiler selects the backend. Start here when these operations cover your
kernel's needs.

The qualified import selects the Numba-CUDA-MLIR API explicitly:

.. code-block:: python
   :name: coop-pg-qualified-import

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
   * - Reduction or scan operator
     - Built-in string operators such as ``"sum"`` and ``"max"``
     - Also accepts supported Python operators and device callbacks;
       callback restrictions depend on the operation and group
   * - Extra Scan results or prefixes
     - Inclusive and exclusive results, including an exclusive initial value
     - Also supports ``aggregate_output``, partial Warp Scan, and Block
       Scan prefix callbacks
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

For example, suppose you need both the exclusive sum and each tile's total.
The qualified Scan can produce both in one call. Here it also consumes an
existing Numba local array:

.. code-block:: python
   :name: coop-pg-qualified-scan

   @cuda.jit
   def scan_tiles_with_totals(source, destination, totals):
       block = numba_coop.this_block()
       items = cuda.local.array(2, dtype=types.int32)
       aggregate = numba_coop.ThreadData(1, dtype=np.int32)
       offset = cuda.blockIdx.x * cuda.blockDim.x * 2

       numba_coop.load(block, source, items, offset=offset)
       prefixes = numba_coop.exclusive_sum(
           block, items, aggregate_output=aggregate
       )
       numba_coop.store(block, destination, prefixes, offset=offset)
       if block.rank() == 0:
           totals[cuda.blockIdx.x] = aggregate[0]


   source = (np.arange(512) % 11).astype(np.int32)
   destination = np.empty_like(source)
   totals = np.empty(2, dtype=np.int32)
   scan_tiles_with_totals[2, 128](source, destination, totals)
   cuda.synchronize()

   tiles = source.reshape(2, 256)
   expected = np.cumsum(tiles, axis=1) - tiles
   np.testing.assert_array_equal(destination.reshape(2, 256), expected)
   np.testing.assert_array_equal(totals, tiles.sum(axis=1))

This version assumes full tiles. The NumPy arrays in this and subsequent
small examples use Numba's host-array transfer support at the launch boundary.

``aggregate_output`` holds the reduction of the input tile. An exclusive
initial value is excluded from that aggregate. You could obtain a total
with a separate common-API reduction, but the qualified call is useful when
you already need a scan and want its aggregate as well.

Using a common spelling expresses a portable API contract. You must still
check that the selected backend implements the requested group, dtype, and
operation. The kernels here also contain Numba launch and indexing code;
porting the complete kernel to another DSL involves those parts too.

Import order
^^^^^^^^^^^^

With the current integration, import Numba-CUDA-MLIR before ``cuda.coop``
to activate the backend automatically. If a dependency imported ``cuda.coop``
first, explicitly import ``cuda.coop.numba_mlir`` with an alias before
compiling. That activates common calls as well.

Keep the alias: bare ``import cuda.coop.numba_mlir`` assigns the top-level
package to the name ``cuda``, replacing an earlier
``from numba_cuda_mlir import cuda`` binding in that scope. These collective
calls belong inside kernels compiled by a compatible backend.

Groups: which threads cooperate
-------------------------------

A group defines the participants in an operation. ``this_block()`` uses
the current block; ``this_warp()`` uses the current physical warp of 32
threads. The compiler obtains the launch dimensions from the kernel launch.
The group factories take no size arguments.

The hierarchy vocabulary includes the following groups. Availability of a
descriptor and availability of a collective on that descriptor are separate
parts of the API.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Expression
     - Participants
     - Current use
   * - ``coop.this_thread()``
     - One thread
     - Hierarchy queries and full built-in Reduce
   * - ``coop.this_warp()``
     - One physical warp
     - Load, Store, Exchange, Reduce, scalar Scan
   * - ``coop.this_warp().group_by(8)``
     - Eight consecutive lanes within a physical warp
     - Logical-warp forms of those operations
   * - ``coop.this_block()``
     - All threads in the block
     - Load, Store, Exchange, Shuffle, Reduce, Scan
   * - ``coop.this_block().group_by(2)``
     - Two consecutive physical warps
     - Mapped-group queries; limited Reduce support
   * - ``coop.this_cluster()``
     - Blocks in the launch's cluster
     - Full built-in Reduce with supported hardware and cluster launch facts
   * - ``coop.this_grid()``
     - The kernel grid
     - Hierarchy queries; grid collectives and grid synchronization are unavailable

``group_by`` counts units in the next inner hierarchy level: threads for a
warp parent, physical warps for a block parent. Thus
``this_block().group_by(2)`` describes 64 threads. For logical Warp
collectives, choose a width of 1, 2, 4, 8, 16, or 32. Use a block size
divisible by 32 for these Warp operations; the last physical warp must be
complete. Nested ``group_by`` calls are unsupported.

The count and ``exhaustive`` flag are compile-time constants. By default,
the partition must cover the parent exactly. A non-exhaustive partition
can leave a remainder. For instance, partitioning a 96-thread block into
pairs of warps leaves the last warp outside a complete group.
``is_member()`` identifies participating threads. Guard queries that require
membership for excluded threads, and check the collective's participation
requirements before using that guard around an operation.

*Mapped groups of physical warps have narrower support than blocks and
logical warps. Their explicit synchronization methods are unavailable, and
the current stack retains an expected scalar Reduce failure pending*
`#10985 <https://github.com/NVIDIA/cccl/pull/10985>`_.
*Use the block and logical-warp forms for the examples in this guide.*

Ranks and sizes
^^^^^^^^^^^^^^^

``group.rank()`` gives the calling thread's rank within the group;
``group.count()`` gives the number of threads in it. Ranks start at zero.
For a multidimensional block, the linear thread rank is
``x + blockDim.x * (y + blockDim.y * z)``.

The optional ``level`` argument lets you query the hierarchy in other units.
For a 128-thread block:

.. list-table::
   :header-rows: 1

   * - Query
     - Meaning
   * - ``block.rank("thread")``
     - Calling thread's linear rank, 0 through 127
   * - ``block.count("thread")``
     - 128 threads
   * - ``block.rank("warp")``
     - Calling warp's rank in the block, 0 through 3
   * - ``block.count("warp")``
     - Four physical warps
   * - ``block.rank("grid")``
     - This block's linear rank in the grid
   * - ``block.count("grid")``
     - Number of blocks in the grid

Queries accept ``thread`` (also ``gpu_thread``), ``warp``, ``block``,
``cluster``, and ``grid`` where the relationship is supported. Mapped groups
can query their constituents and immediate physical parent. Queries above
that parent are rejected.

Results normally use unsigned 32-bit integers; queries involving the grid
use unsigned 64-bit integers. ``rank_as`` and ``count_as`` take an explicit
integer dtype, for example ``block.rank_as(types.int32)``. A signed result
can be convenient for address calculations involving subtraction; choose a
type large enough for the launch.

Independent scans within eight-lane groups
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Suppose each row has eight values. One logical warp can scan each row,
giving four independent row scans per physical warp:

.. code-block:: python
   :name: coop-pg-row-scan

   @cuda.jit
   def scan_rows(source, destination):
       row_group = coop.this_warp().group_by(8)
       index = cuda.grid(1)
       destination[index] = coop.inclusive_sum(row_group, source[index])


   source = (np.arange(256) % 5).astype(np.int32)
   destination = np.empty_like(source)
   scan_rows[2, 128](source, destination)
   cuda.synchronize()
   np.testing.assert_array_equal(
       destination.reshape(32, 8), np.cumsum(source.reshape(32, 8), axis=1)
   )

Warp Scan accepts one scalar per lane. Block Scan also accepts multiple
items per thread. The row example has exactly enough threads for the input;
a more general kernel must handle its final rows explicitly.

Participation and synchronization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every required participant must reach the same collective invocation.
For a block operation, a branch around the call must be uniform across the
block. For a logical-warp operation, it must be uniform within that logical
warp. Complete sibling logical groups can follow different paths.

In particular, putting a block collective inside ``if index < count``
breaks participation on a partial tile. Keep the collective outside the
per-element condition. Use guarded loads or initialized values to handle
missing input, as in the first kernel. An early return by some block threads
has the same problem if the remaining threads later execute a block
collective.

``group.sync()`` provides a barrier for supported groups. Every member must
reach it. ``sync_aligned()`` has the additional requirement that the group
be aligned and converged; use ``sync()`` unless your code establishes that
stronger precondition. For explicit block synchronization in Numba kernels,
``cuda.syncthreads()`` is also available.

An operation's scratch-reuse barrier protects its temporary storage.
Arrange synchronization for your own shared-memory communication as well.
Constructing a group or a ``ThreadData`` object does not synchronize threads.

``ThreadData``: the part of a tile owned by one thread
----------------------------------------------------

``coop.ThreadData(2, dtype=np.int32)`` gives each thread two integer slots.
With 128 threads, the group collectively owns 256 values. Each thread
indexes its own slots with ``items[0]`` and ``items[1]``. To move values
between threads, use an operation such as Exchange, Shuffle, or Scan.

The item count must be a positive compile-time integer. You can use
``items.items_per_thread`` as a loop bound:

.. code-block:: python

   for i in range(items.items_per_thread):
       items[i] = types.int32(items[i] * 2)

Initialize every slot before reading it. Constructing ``ThreadData`` does
not fill it with zeros. A full Load initializes the entire payload; a
partial Load needs either an ``oob_default`` or previously initialized slots
for the missing elements.

Blocked and striped order
^^^^^^^^^^^^^^^^^^^^^^^^^

A collective needs to know how the per-thread slots correspond to the
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

Block Scan interprets an array payload in blocked order. If you loaded it
with the striped algorithm, exchange it into blocked order first. Choosing
the transpose Load algorithm performs that rearrangement as part of Load.

The payload does not carry a runtime layout tag that corrects mismatched
operations. The algorithms you call determine the interpretation. Pairing
a striped Load with a direct Store without a conversion permutes the data.

Dtypes and storage
^^^^^^^^^^^^^^^^^^

The current numeric payload types are signed and unsigned integers of 8,
16, 32, or 64 bits, and 32- or 64-bit floating point. Boolean, half
precision, complex, and structured payloads are outside this contract.
Bitwise operators require integer values.

You may omit ``dtype`` when surrounding operations establish it:
``items = coop.ThreadData(2)`` followed by Load infers the source dtype.
If you initialize the payload yourself, specifying a dtype usually makes
the code easier to follow. Conflicting dtype requirements are errors.

Load writes into the payload supplied by the caller and returns that same
payload. Store preserves its input. Scan, Exchange, and array Shuffle
return fresh payloads, so their input values remain available afterwards.
Reduction returns a scalar, including when each thread contributes several
items.

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
-------------------

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

This kernel loads striped data, exchanges it into blocked order, and then
computes the inclusive sum in the original array order:

.. code-block:: python
   :name: coop-pg-exchange

   @cuda.jit
   def scan_striped_input(source, destination):
       block = coop.this_block()
       items = coop.ThreadData(2, dtype=np.int32)
       coop.load(block, source, items, algorithm="striped")
       blocked = coop.exchange(block, items, mode="striped_to_blocked")
       prefixes = coop.inclusive_sum(block, blocked)
       coop.store(block, destination, prefixes, algorithm="direct")


   source = (np.arange(256) % 13).astype(np.int32)
   destination = np.empty_like(source)
   scan_striped_input[1, 128](source, destination)
   cuda.synchronize()
   np.testing.assert_array_equal(destination, np.cumsum(source))

The common Exchange API also supports ``blocked_to_striped``. Qualified
block scatter modes let you supply destination ranks for finer control.
For scatter, valid ranks and unique active destinations are caller
requirements; duplicate destinations and holes leave unspecified slots.

Shuffle operates on a block's flattened blocked tile. The common ``up``
and ``down`` modes shift it by one element and return a fresh payload.
The first ``up`` slot or last ``down`` slot is unspecified. Set that
boundary yourself before consuming it. Qualified scalar ``offset`` and
``rotate`` modes have different distance rules; see :doc:`../coop_api`
before substituting them for an array shift.

Warp tile addresses
^^^^^^^^^^^^^^^^^^^

Warp Load and Store automatically add the group's origin within the
block. For width ``G`` and ``K`` items per thread, this origin is
``(linear_thread_rank // G) * G * K``. The explicit ``offset`` is added
after that origin. In a multi-block traversal, pass the block's global
tile origin as ``offset``.

The valid count still belongs to each individual group. Compute it using
both the block origin and the group's origin:

.. code-block:: python
   :name: coop-pg-warp-copy

   @cuda.jit
   def copy_warp_tiles(source, destination, count):
       group = coop.this_warp().group_by(8)
       items = coop.ThreadData(2, dtype=np.int32)
       block_origin = cuda.blockIdx.x * cuda.blockDim.x * 2
       group_origin = (cuda.threadIdx.x // 8) * 16
       valid = min(max(count - block_origin - group_origin, 0), 16)

       coop.load(
           group,
           source,
           items,
           offset=block_origin,
           valid_items=valid,
           oob_default=0,
       )
       coop.store(
           group, destination, items, offset=block_origin, valid_items=valid
       )


   source = np.arange(531, dtype=np.int32)
   destination = np.full_like(source, -1)
   copy_warp_tiles[3, 128](source, destination, source.size)
   cuda.synchronize()
   np.testing.assert_array_equal(destination, source)

Adding ``group_origin`` to ``offset`` here would count it twice. This
automatic origin applies to Warp Load and Store; ordinary array indexing
in a kernel uses exactly the index you write.

``TempStorage``: scratch used during a collective
-----------------------------------------------

Some algorithms exchange intermediate values through shared memory.
By default, ``cuda.coop`` allocates the scratch they need and inserts
the required reuse barrier. Start with that behavior.

An explicit ``TempStorage`` descriptor lets several supported block calls
reuse an allocation and lets you control capacity, alignment, and
synchronization. Construct it inside the kernel. Its contents are opaque;
keep application values in ``ThreadData`` or your own arrays.

.. list-table::
   :header-rows: 1

   * - Calls
     - Scratch behavior in the current backend
   * - Direct, striped, or vectorize Load/Store
     - No shared scratch or reuse barrier
   * - Block transpose-family Load/Store; Block Scan
     - Automatic scratch, or an explicit ``TempStorage``
   * - Warp transpose Load/Store; Warp Scan
     - Automatic scratch per group; explicit descriptors are rejected
   * - Exchange and Shuffle
     - Compiler-owned scratch and reuse synchronization

Reduce has its own group-dependent implementation and does not accept
``temp_storage`` in the public signature.

Reusing scratch across operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This version of a tile scan shares one descriptor between the transpose
Load, Scan, and transpose Store:

.. code-block:: python
   :name: coop-pg-shared-scratch

   @cuda.jit
   def scan_with_shared_scratch(source, destination):
       block = coop.this_block()
       scratch = coop.TempStorage()
       items = coop.ThreadData(2, dtype=np.int32)

       coop.load(
           block, source, items, algorithm="transpose", temp_storage=scratch
       )
       prefixes = coop.exclusive_sum(block, items, temp_storage=scratch)
       coop.store(
           block,
           destination,
           prefixes,
           algorithm="transpose",
           temp_storage=scratch,
       )


   source = (np.arange(256) % 7).astype(np.int32)
   destination = np.empty_like(source)
   scan_with_shared_scratch[1, 128](source, destination)
   cuda.synchronize()
   np.testing.assert_array_equal(destination, np.cumsum(source) - source)

The planner sizes and aligns the shared allocation for its uses. The Load
finishes using scratch before Scan reuses it, and Scan finishes before
Store begins using it. Automatic block barriers enforce that ordering.
The loaded values and the returned prefixes remain in their per-thread
payloads while scratch is reused.

Capacity, alignment, and lifetime
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``TempStorage()`` defaults to ``sharing="shared"`` with automatic reuse
synchronization. ``size_in_bytes=None`` and ``alignment=None`` let the
compiler determine the requirements. An explicit capacity must be large
enough for the operations using it. An explicit ``alignment`` requests a
minimum positive power of two in bytes; the compiler can strengthen it.
Only ``size_in_bytes`` may be positional. The other options are keyword-only.

``sharing="exclusive"`` gives distinct call sites separate slices and
disables automatic reuse synchronization. A loop can still reach the same
call site again and reuse its slice. Account for that reuse when deciding
where barriers belong. Exclusive storage also consumes more shared memory
when several calls could otherwise share a slice.

``auto_sync=False`` transfers reuse synchronization to the caller. For
example, this kernel uses explicit block barriers after each storage-using
call, including between loop iterations:

.. code-block:: python
   :name: coop-pg-manual-scratch

   @cuda.jit
   def copy_tiles_with_manual_sync(source, destination):
       block = coop.this_block()
       scratch = coop.TempStorage(auto_sync=False)
       items = coop.ThreadData(2, dtype=np.int32)
       for tile in range(2):
           offset = tile * cuda.blockDim.x * 2
           coop.load(
               block,
               source,
               items,
               offset=offset,
               algorithm="transpose",
               temp_storage=scratch,
           )
           block.sync()
           coop.store(
               block,
               destination,
               items,
               offset=offset,
               algorithm="transpose",
               temp_storage=scratch,
           )
           block.sync()


   source = np.arange(512, dtype=np.int32)
   destination = np.empty_like(source)
   copy_tiles_with_manual_sync[1, 128](source, destination)
   cuda.synchronize()
   np.testing.assert_array_equal(destination, source)

Automatic synchronization is easier to maintain. Disable it only when you
can account for each reuse and have a reason to place barriers yourself.
An unrelated memory access between calls does not establish a block barrier.

Scratch lasts for the kernel's execution on that block. It cannot carry
state between blocks or kernel launches. For running scan state within a
block, use a separate payload as in the prefix-callback example below.

When the combined scratch requirement exceeds the default static shared-memory
limit, the backend can use dynamic shared memory, subject to the GPU's opt-in
limit. Numba-CUDA-MLIR automatically includes those required bytes in the
launch configuration. You do not need to copy a compiler-reported byte count
into the launch yourself. Requirements above the device limit are rejected.

Extra shared memory can reduce resident blocks per multiprocessor. The
default inferred allocation and unsized shared descriptor are sufficient
for the kernels above; use an explicit capacity when you have a reason to
reserve that amount of shared memory.

Reduction and result ownership
------------------------------

``coop.sum`` and ``coop.reduce`` combine the group's inputs into a scalar.
For a ``ThreadData`` input, every item in every thread contributes.
The default ``broadcast=True`` makes the result available to every group
member. With ``broadcast=False``, consume it only on group rank zero.
All required members still execute the reduction.

This kernel writes one sum per block, including a partial final tile:

.. code-block:: python
   :name: coop-pg-reduce

   @cuda.jit
   def tile_sums(source, totals, count):
       block = coop.this_block()
       items = coop.ThreadData(2, dtype=np.int32)
       tile_size = cuda.blockDim.x * 2
       offset = cuda.blockIdx.x * tile_size
       valid = min(max(count - offset, 0), tile_size)
       coop.load(
           block,
           source,
           items,
           offset=offset,
           valid_items=valid,
           oob_default=0,
       )
       total = coop.sum(block, items, broadcast=False)
       if block.rank() == 0:
           totals[cuda.blockIdx.x] = total


   source = (np.arange(785) % 17).astype(np.int32)
   totals = np.empty(4, dtype=np.int32)
   tile_sums[4, 128](source, totals, source.size)
   cuda.synchronize()
   expected = [
       source[start : start + 256].sum()
       for start in range(0, source.size, 256)
   ]
   np.testing.assert_array_equal(totals, expected)

The group-rank test surrounds only the result write. Moving the reduction
into that branch would leave the other threads out of a collective.

For another built-in operator, use ``coop.reduce`` with ``binary_op`` set
to ``"min"``, ``"max"``, ``"multiplies"``, ``"bit_and"``,
``"bit_or"``, or ``"bit_xor"``. Choose the padding value accordingly.
Qualified custom Reduce callbacks have narrower group and result contracts;
check their overloads before replacing a built-in operation.

Partial scalar Reduce also accepts ``valid_items`` on block and warp
groups with ``broadcast=False``. There it counts contributing threads,
must be at least one, and requires a scalar input. The example instead
pads a multi-item Load and reduces the full payload. These two techniques
have different valid-count contracts.

Scan operators and carrying a prefix
-----------------------------------

An inclusive scan includes the current element; an exclusive scan starts
with an initial value and excludes the current element. For sum, the
default exclusive initial value is zero. ``inclusive_sum`` and
``exclusive_sum`` are convenient spellings; ``inclusive_scan`` and
``exclusive_scan`` accept ``scan_op``.

For a non-sum exclusive scan, supply ``initial_value`` with the correct
dtype and meaning for the operator. Inclusive Scan rejects an initial
value. Block array scans flatten their inputs in blocked order and return
one result for every input element.

Built-in operators use the same string vocabulary as Reduce. Scan relies
on an associative operation: regrouping the inputs must preserve the
intended result. Floating-point arithmetic only approximates this
property, so choose numerical tolerances that suit the application.

A custom operator
^^^^^^^^^^^^^^^^^

Use a qualified call to pass a device function. For instance, the
following explicit maximum operator computes a running maximum:

.. code-block:: python
   :name: coop-pg-custom-scan

   @cuda.jit(device=True)
   def maximum(left, right):
       if left > right:
           return left
       return right


   @cuda.jit
   def running_maximum(source, destination):
       block = numba_coop.this_block()
       items = numba_coop.ThreadData(2, dtype=np.int32)
       numba_coop.load(block, source, items)
       result = numba_coop.inclusive_scan(block, items, scan_op=maximum)
       numba_coop.store(block, destination, result)


   source = ((np.arange(256) * 17) % 113 - 51).astype(np.int32)
   destination = np.empty_like(source)
   running_maximum[1, 128](source, destination)
   cuda.synchronize()
   np.testing.assert_array_equal(destination, np.maximum.accumulate(source))

For maximum alone, the common ``scan_op="max"`` already suffices. The
device function shows where to put an application's own associative
operator. It executes on the GPU and must return the payload dtype.
Binary Scan callbacks are stateless in the current API; supported payloads
remain numeric scalars even when a thread owns several items.

Several tiles in one block
^^^^^^^^^^^^^^^^^^^^^^^^^

A Block Scan prefix callback supplies the prefix preceding a tile. A
stateful callback can also update a running total for the next tile. The
following kernel scans 384 values using one block and three successive
128-element tiles:

.. code-block:: python
   :name: coop-pg-prefix-callback

   @cuda.jit(device=True)
   def carry_total(state, tile_total):
       previous = state[0]
       state[0] = previous + tile_total
       return previous


   running_prefix = numba_coop.StatefulFunction(carry_total, types.int64)


   @cuda.jit
   def scan_successive_tiles(source, destination, final_total):
       block = numba_coop.this_block()
       state = numba_coop.ThreadData(1, dtype=types.int64)
       state[0] = types.int64(0)
       scratch = numba_coop.TempStorage()
       for tile in range(3):
           index = tile * cuda.blockDim.x + cuda.threadIdx.x
           destination[index] = numba_coop.exclusive_sum(
               block,
               source[index],
               state,
               prefix_op=running_prefix,
               temp_storage=scratch,
           )
       if block.rank() == 0:
           final_total[0] = state[0]


   source = (np.arange(384) % 9).astype(np.int32)
   destination = np.empty_like(source)
   final_total = np.empty(1, dtype=np.int64)
   scan_successive_tiles[1, 128](source, destination, final_total)
   cuda.synchronize()
   np.testing.assert_array_equal(destination, np.cumsum(source) - source)
   np.testing.assert_array_equal(final_total, [source.sum()])

The callback receives the state first and the tile aggregate second. It
returns the old total as the tile's prefix and saves the new total for the
next call. The state payload is created before the loop, and every thread
initializes its copy identically. Automatic scratch barriers remain enabled.

CUB can invoke the callback in every lane of the block's first warp; only
lane zero's returned prefix is applied. After the calls, read the final
state from thread zero, as above. Other threads' state copies are not
authoritative.

State must be a numeric one-item payload with exactly the dtype declared
by ``StatefulFunction``. Its dtype may differ from the scanned value dtype,
as it does here, but the output still has the scanned dtype. A wider state
does not widen the scan results.

Prefix callbacks are available only on qualified Block Scan calls. They
cannot be combined with ``initial_value`` or ``aggregate_output``. A
stateless prefix callback accepts just the tile aggregate and returns the
prefix. Warp Scan has no prefix callback support.

Each block has its own state. Launching this kernel with several blocks
would require separate input/output ranges and would create independent
running sums. For a whole-array scan across many blocks, use an appropriate
device-wide scan or design the additional inter-block algorithm explicitly.

Checking and tuning a kernel
---------------------------

Check results before comparing algorithms. Useful cases include one full
tile, several tiles, a single valid element in the final tile, and an empty
input handled on the host. Layout conversions are easier to inspect with
distinct input values. For reductions and scans, compare against a CPU
reference with a suitable accumulation dtype and floating-point tolerance.

Keep launch dimensions, logical-warp widths, payload extents, and algorithm
choices consistent with the code. The compiler specializes group operations
using these facts. A runtime tile offset or valid count can change from
one call to the next; the size of a ``ThreadData`` payload must be known
at compile time.

Warm up the kernel before timing it, use device-resident arrays, and account
for asynchronous execution with CUDA events or explicit synchronization.
Then vary one choice at a time: threads per block, items per thread, or a
Load/Store or Scan algorithm. More items per thread can amortize collective
work while increasing register pressure. Scratch-heavy choices consume
shared memory and can reduce occupancy. Measure the complete kernel,
including conversions and synchronization.

If compilation fails, check the group/operation combination, payload dtype,
static parameters, and import order first. The :doc:`API reference
<../coop_api>` records exact signatures; the :doc:`overview <../coop>`
collects operation restrictions and configuration. For generated-source
diagnostics and the compiler integration, see the
:doc:`Developer Overview <developer_overview>`.
