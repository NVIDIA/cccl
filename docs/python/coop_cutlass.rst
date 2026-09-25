.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-cutlass:
.. _cuda.coop.cutlass.programming_guide:
.. _cuda-coop-cutlass-cute-dsl-integration:

CUTLASS Programming Guide
=========================

Use ``cuda.coop`` inside a CuTe kernel to load a tile, reduce or scan its
values, rearrange, sort, or select items, and store the result. The CUTLASS
backend implements these primitives with CUB and CUDAX.

Each thread keeps its items in a ``ThreadData`` object. ``load`` fills that
object and returns ``None``; ``store`` writes its items to memory without
changing them. The examples below show the same kernel using the common API
and the CUTLASS-qualified API.

The :doc:`overview <coop>` introduces the shared concepts, installation, and
primitive families. This guide covers writing CuTe kernels; the
:doc:`CUTLASS Developer Guide <coop/cutlass_developer_guide>` explains how the
compiler integration works. For Numba kernels, see the
:doc:`Numba-CUDA-MLIR Programming Guide <coop/programming_guide>` and
:doc:`Numba-CUDA-MLIR Developer Guide <coop/developer_overview>`.

.. _coop-cutlass-api-choice:

.. _choosing-the-portable-or-qualified-api:

Choosing the common or qualified API
------------------------------------

Start with ``from cuda import coop`` for the common API. Use
``import cuda.coop.cutlass as coop`` when you need the extra controls in the
table below, such as scatter ranks for Exchange or a Scan aggregate.
Both imports call the same implementation inside a CuTe kernel.

.. list-table:: Common and CUTLASS-qualified APIs
   :header-rows: 1
   :widths: 22 38 40

   * - Feature
     - Common ``cuda.coop``
     - Qualified ``cuda.coop.cutlass``
   * - Payloads
     - Fixed per-thread ``ThreadData``; Load fills it in place.
     - Adds CuTe register-tensor and vector conversions, described in
       :ref:`coop-cutlass-register-payloads`.
   * - Operator selection
     - Built-in operator names such as ``"sum"`` and ``"max"``.
     - Also accepts recognized ``operator`` and NumPy aliases. Arbitrary
       Python callbacks remain unsupported.
   * - Scan
     - Block and scalar Warp Scan with the shared initial-value and
       algorithm controls.
     - Adds Warp ``valid_items`` and writable ``aggregate_output`` to all
       five Scan spellings; see :ref:`coop-cutlass-scan`.
   * - Exchange
     - Blocked/striped conversions for block and supported warp groups.
     - Adds block warp-striped conversions, scatter ranks and flags, and
       ``warp_time_slicing``; see :ref:`coop-cutlass-exchange`.
   * - Shuffle
     - Block array Up/Down with unit distance.
     - Adds scalar Offset/Rotate with checked integer distances; see
       :ref:`coop-cutlass-shuffle`.
   * - Merge Sort
     - Built-in ascending/descending keys or pairs, including partial tiles.
     - Also accepts CuTe register tensors and returns fresh ``ThreadData``;
       see :ref:`coop-cutlass-merge-sort`. Custom comparators are unsupported.
   * - Radix Sort and Rank
     - Block payloads with 32- or 64-bit integer keys.
     - Adds scalar and register-tensor inputs, floating-point Sort keys,
       striped Sort results, and Rank bin prefixes; see :ref:`coop-cutlass-radix`.
   * - TopK
     - Block minimum or maximum keys/pairs with common count and scratch controls.
     - Also accepts CuTe register tensors and returns fresh ``ThreadData``;
       see :ref:`coop-cutlass-topk`.

.. _coop-cutlass-differences:

.. _cutlass-specific-behavior-and-current-limits:

CuTe values and supported features
----------------------------------

Use the qualified ``ThreadData`` to work with CuTe register tensors.
``ThreadData.from_register_tensor(fragment)`` copies a fragment into a
payload you can pass to ``store`` or another primitive.
``values.to_register_tensor()`` converts a payload back to a CuTe register
tensor. See :ref:`coop-cutlass-register-payloads`.

Group queries return CuTe scalars. For example, ``block.rank()`` returns a
``cutlass.Uint32`` that you can use in pointer arithmetic or a condition
inside the kernel. Use ``block.rank_as(cutlass.Int32)`` when you need a signed
rank.

All threads in the group must call the primitive, even when ``valid_items``
selects a short tile or only rank zero uses the result. The sections below
describe the requirements for block, warp, and mapped groups.

Reduce and Scan support the built-in operators listed below. Custom
operators and Scan prefix callbacks are not yet supported. The shared
:ref:`coverage table <coop-backends>` lists the implemented primitive families
and their backend support.

.. _coop-cutlass-mixed-backends:

Mixing kernels from both compilers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CUTLASS and Numba-CUDA-MLIR kernels can run in the same process. Use the
selected device's primary CUDA context before allocating memory or launching
kernels with either runtime. Numba-CUDA-MLIR requires this context; it rejects
a context created independently by another runtime.

Synchronize a kernel's work before the other runtime reads its output. Pass
data between kernels through device memory; ``ThreadData`` and CuTe register
tensors are local to the kernel that uses them.

Runtime requirements
--------------------

The base ``cuda-coop`` wheel includes this optional backend. Its initial
development target is Linux with CUDA 13. A supported public CUTLASS package
version has not yet been qualified, so ``cuda-coop`` does not provide a
CUTLASS installation extra.

The dependency-free base wheel does not install compiler prerequisites. A
CUTLASS environment also needs NumPy, ``cuda-pathfinder>=1.2.3``, and
``typing_extensions>=4.12.0`` for runtime discovery and type declarations,
alongside a compatible CuTe compiler and its dependencies.

The CuTe compiler must support linking external NVIDIA LTO-IR into a kernel,
and NVRTC must be available to compile the CUB and CUDAX functions. See the
:ref:`developer guide's compiler requirements <coop-cutlass-compiler-requirements>`
for the required CuTe integration hooks. Importing :mod:`cuda.coop` alone does
not load CUTLASS or initialize CUDA bindings.

Activation and example
----------------------

Register CUTLASS on the host before compiling:

.. code-block:: python

   from cuda import coop

   coop.register("cutlass")

   import cutlass.cute as cute

Registration works in either import order, is safe to repeat, and remains
available when automatic registration is disabled. Importing
``cuda.coop.cutlass`` also registers the backend. For convenience, importing
``cuda.coop`` after ``cutlass`` activates it automatically.

After registration, call the primitives inside ``@cute.kernel`` or a
``@cute.jit`` function called by that kernel.

This example loads two adjacent items per thread and stores a partial tile in
the same blocked layout. ``module`` selects the common or qualified API. The
full example defines the tile dimensions and checks the output against a CPU
reference. :download:`Download the example
<../../python/cuda_coop/examples/cutlass/block_load_store.py>` to run it with a
compatible compiler:

.. literalinclude:: ../../python/cuda_coop/examples/cutlass/block_load_store.py
   :language: python
   :start-after: docs: start cutlass-block-load-store
   :end-before: docs: end cutlass-block-load-store

Block Load and Store support one-, two-, and three-dimensional blocks.
``offset`` selects the beginning of the block's tile and ``valid_items``
specifies the number of valid items in that tile. Load may fill its
out-of-bounds items with ``oob_default``. Without that default, initialize
any items that the valid prefix will not overwrite before reading them.
All threads in the block must call the primitive with uniform controls.

Block algorithms
----------------

The six block algorithms use these register layouts. For linear thread rank
``t``, item index ``i``, block size ``B``, and ``I`` items per thread, blocked
layout accesses tile index ``t * I + i``; striped layout accesses
``t + i * B``.

.. list-table:: Block Load and Store algorithms
   :header-rows: 1

   * - ``algorithm``
     - Register layout
     - Shared scratch
   * - ``direct``
     - Blocked
     - None
   * - ``striped``
     - Striped
     - None
   * - ``vectorize``
     - Blocked
     - None
   * - ``transpose``
     - Blocked
     - Required
   * - ``warp_transpose``
     - Blocked
     - Required
   * - ``warp_transpose_timesliced``
     - Blocked
     - Required

The two warp-transpose block algorithms require a block size divisible by 32.
``vectorize`` uses vector accesses when the type, item count, and address
alignment permit them, with direct accesses as a fallback.

Direct, striped, and vectorized Load/Store use no shared scratch and need no
scratch-reuse barrier, even when passed a ``TempStorage`` descriptor.
``ThreadData(alignment=...)`` requests a minimum payload alignment; it does
not change the logical item layout.

Block scratch and reuse
-----------------------

Transpose algorithms allocate scratch implicitly unless passed
``temp_storage``. Construct one ``TempStorage`` inside the kernel to share
capacity across calls. An omitted size lets the compiler allocate enough
storage for all uses; an explicit byte capacity must accommodate them.
``alignment`` is a minimum: the allocation also satisfies each primitive's
alignment requirements.

``sharing="shared"`` reuses one slice across call sites. With
``sharing="exclusive"``, distinct call sites receive separate slices.
Both policies insert trailing reuse synchronization by default because a
single call site can execute repeatedly in a loop. Set ``auto_sync=False``
only when the kernel calls ``storage.sync()`` before reusing that storage,
including on the next loop iteration.

The following example transforms eight independent tiles. Its default is a
shared descriptor with automatic synchronization; the executable example
also supports exclusive slices and manual synchronization.
:download:`Download the storage example
<../../python/cuda_coop/examples/cutlass/block_storage.py>`:

.. literalinclude:: ../../python/cuda_coop/examples/cutlass/block_storage.py
   :language: python
   :start-after: docs: start cutlass-block-storage
   :end-before: docs: end cutlass-block-storage

Physical Warp Load and Store
----------------------------

``this_warp()`` selects the calling thread's complete 32-lane warp. The block
size must be divisible by 32, and all lanes in each participating warp must
call the primitive with uniform controls. Different warps may use different
``valid_items``, ``oob_default``, and ``offset`` values.

The four warp algorithms use the same layouts as their block counterparts:
``direct`` and ``vectorize`` use blocked layout without scratch, ``striped``
uses striped layout without scratch, and ``transpose`` uses blocked layout
with independent scratch for each warp. Transpose scratch is allocated
implicitly, with automatic warp synchronization for reuse. Explicit
``temp_storage`` is rejected for every warp algorithm.

Each warp addresses a consecutive tile within the block. For ``I`` items per
thread and linear thread rank ``t``, the compiler adds
``(t // 32) * 32 * I`` to the user-provided ``offset``. The linear rank flattens
the exact block dimensions in CUDA order: ``x + block_x * (y + block_y * z)``.
Do not add the within-block warp origin yourself. An offset for a different
block or a later loop iteration remains the caller's responsibility.

``valid_items`` counts the valid prefix of each warp's tile, from zero through
``32 * I``. As with Block Load, a partial load preserves initialized payload
items outside that prefix unless ``oob_default`` is supplied. Store writes
only the valid prefix and preserves its input payload.

This example uses two physical warps in an ``(8, 4, 2)`` block and checks the
independent partial tiles against a CPU reference.
:download:`Download the Warp example
<../../python/cuda_coop/examples/cutlass/warp_load_store.py>`:

.. literalinclude:: ../../python/cuda_coop/examples/cutlass/warp_load_store.py
   :language: python
   :start-after: docs: start cutlass-warp-load-store
   :end-before: docs: end cutlass-warp-load-store

Logical Warp Load and Store
---------------------------

``this_warp().group_by(width)`` partitions each physical warp into consecutive
groups of 1, 2, 4, 8, 16, or 32 threads. The width and ``exhaustive`` flag must
be compile-time constants; the default exhaustive partition covers the whole
physical warp. The enclosing block must still contain complete 32-lane warps.
All four Warp Load and Store algorithms support these logical groups.

Each logical group receives its own tile and, for ``transpose``, independent
implicit scratch. With group width ``W``, linear block rank ``t``, and ``I``
items per thread, the compiler adds ``(t // W) * W * I`` to ``offset``.
Blocked layout uses tile index ``(t % W) * I + i``; striped layout uses
``(t % W) + i * W``. ``valid_items`` describes a prefix of at most ``W * I``
items. Default filling and preservation of initialized invalid items follow
the same rules as physical Warp Load.

Every member of a participating logical group must call the primitive with
uniform controls. Complete sibling groups may take different control-flow
paths or use different offsets and valid counts. Transpose reuse
synchronization is masked to the participating logical group. Explicit
``temp_storage`` remains unsupported, and nested partitions or groups of
physical warps cannot be used for Load and Store.

This example partitions the two physical warps in an ``(8, 4, 2)`` block into
eight groups of eight threads, each loading its own partial tile.
:download:`Download the logical Warp example
<../../python/cuda_coop/examples/cutlass/logical_warp_load_store.py>`:

.. literalinclude:: ../../python/cuda_coop/examples/cutlass/logical_warp_load_store.py
   :language: python
   :start-after: docs: start cutlass-logical-warp-load-store
   :end-before: docs: end cutlass-logical-warp-load-store

Hierarchy queries and synchronization
-------------------------------------

``this_thread()``, ``this_warp()``, ``this_block()``, ``this_cluster()``, and
``this_grid()`` describe the corresponding physical groups. ``rank`` and
``count`` accept a hierarchy level: ``thread`` (also spelled ``gpu_thread``),
``warp``, ``block``, ``cluster``, or ``grid``. Their default result is a CuTe
``Uint32``, or ``Uint64`` when the group or queried level is the grid.
``rank_as(dtype, level="thread")`` and ``count_as`` select a signed or unsigned
8-, 16-, 32-, or 64-bit integer type. Floating and Boolean query types are
unsupported. NumPy integer dtypes and Python ``int`` are also accepted as dtype
selectors; the compiled values are CuTe scalars. ``is_member()`` returns a CuTe
``Uint8`` membership flag.

Mapped groups may query their constituents and immediate physical parent.
Thus ``this_warp().group_by(8)`` supports thread and warp queries, while
``this_block().group_by(2)`` supports thread, warp, and block queries. Queries
above that parent are rejected. With ``exhaustive=False``, trailing units that
cannot form a complete group are excluded; guard rank-dependent work with
``is_member()``. Metadata queries do not synchronize threads.

``sync()`` supports thread, physical warp, logical warp, block, and cluster
groups. Every participating member must reach the synchronization;
``sync_aligned()`` additionally requires an aligned, converged group.
Synchronization of mapped groups of physical warps and grid groups is
unsupported. Queries and synchronization consume the exact dimensions and
launch flags supplied by the compiler. Cluster primitives require consistent
cluster dimensions and launch mode; grid queries also require exact grid
dimensions.

Built-in Reduce and Sum
-----------------------

``reduce(group, value, ...)`` and ``sum(group, value, ...)`` accept a scalar
or fixed per-thread ``ThreadData`` payload. Full-group reductions support
thread, physical and logical warp, block, mapped groups of physical warps,
and cluster groups. Grid reductions are unsupported. All members of a
participating group must call the primitive.

For a mapped group of physical warps, every thread in the enclosing block
must reach the reduction, including nonmembers of a non-exhaustive partition:
setting up the reduction synchronizes the parent block. Restrict use of the
result to participating members; do not guard the reduction itself with
``is_member()``.

The built-in operators are sum, product, minimum, maximum, bitwise AND,
bitwise OR, and bitwise XOR. For example, ``binary_op="max"`` selects maximum,
and ``binary_op="bit_or"`` selects bitwise OR. An omitted operator selects
sum. Bitwise operators require integer values. The qualified API also accepts
known ``operator`` and NumPy callable aliases; arbitrary callbacks are
unsupported.

With the default ``broadcast=True``, every group member may use the scalar
result. With ``broadcast=False``, only group rank zero may use it; the other
members must still call the primitive. Nonmembers of a non-exhaustive mapped
group have no defined result. The input payload remains unchanged.

Full-group reductions without algorithm controls use CUDAX. An explicit block
algorithm or ``valid_items`` selects CUB and requires ``broadcast=False``:

.. list-table:: Reduction controls
   :header-rows: 1

   * - Group and input
     - Supported controls
   * - Block scalar
     - ``valid_items`` and any supported block algorithm
   * - Block multi-item payload
     - An explicit block algorithm, without ``valid_items``
   * - Physical or logical warp scalar
     - ``valid_items``, without an algorithm selector

Block algorithm names are ``raking_commutative_only``, ``raking``, and
``warp_reductions``. A valid prefix contains from one through the group size
contributing members, counting threads rather than payload elements. The count
must be uniform within the group. Zero and out-of-range counts are invalid;
all members still participate even when their values fall outside the prefix.
Reduction scratch is managed by the implementation, including synchronization
for repeated reuse; these calls do not accept ``temp_storage``.

This example uses block rank queries, a full block sum, logical-warp maxima,
and a scalar valid-prefix sum whose result is read only at block rank zero.
:download:`Download the reduction example
<../../python/cuda_coop/examples/cutlass/reduce.py>`:

.. literalinclude:: ../../python/cuda_coop/examples/cutlass/reduce.py
   :language: python
   :start-after: docs: start cutlass-reduce
   :end-before: docs: end cutlass-reduce

.. _coop-cutlass-scan:

Built-in Scan
-------------

``scan``, ``exclusive_scan``, ``inclusive_scan``, ``exclusive_sum``, and
``inclusive_sum`` support block, physical warp, and logical warp groups. Block
primitives accept scalars and fixed multi-item payloads; warp primitives
accept one scalar per lane. A ``ThreadData(1, ...)`` remains an array payload
and is not accepted by Warp Scan. Input values are preserved. A scalar input
returns a scalar; a block payload returns a fresh ``ThreadData`` with the same
dtype and extent in blocked order.

Scan supports the same seven built-in operators as Reduce. Values may be
signed or unsigned 8-, 16-, 32-, or 64-bit integers, or 32- or 64-bit floats;
bitwise operators require integers. ``scan`` defaults to exclusive Sum.
Exclusive Sum starts from typed zero unless ``scan`` or ``exclusive_scan``
supplies ``initial_value``. Other exclusive operators require that initial
value. Inclusive scans do not accept an initial value.

The initial value must be uniform within the group. A typed value must match
the input dtype exactly. Python numeric literals must be finite and
representable in that dtype; integer input requires an integer literal.
Custom operators, prefix callbacks, and callback state are unsupported.

The block algorithms are ``raking`` (the default), ``raking_memoize``, and
``warp_scans``. The last requires a block size divisible by 32. All block Scan
algorithms use scratch and accept ``temp_storage`` with the size, alignment,
sharing, and synchronization rules described above. One descriptor can be
reused between Scan and Load/Store. Warp Scan manages independent scratch per
group and rejects algorithm selectors and explicit storage. Every member of
each participating group must call the primitive, including on repeated
calls and loop iterations.

The qualified CUTLASS API adds two controls to all five Scan spellings:

.. list-table:: Qualified Scan controls
   :header-rows: 1

   * - Keyword
     - Contract
   * - ``valid_items``
     - Warp-only valid prefix of one through the group width, uniform within
       the group. Every lane participates; only lanes below the count have
       defined scan results.
   * - ``aggregate_output``
     - Writable one-element ``ThreadData`` in the input dtype, inferred if
       omitted from the descriptor. Receives the aggregate of input values,
       excluding the initial value, at every group member.

For a partial warp scan, the aggregate includes only the valid prefix and is
available even on lanes outside that prefix. A zero count is invalid. The
common API does not expose these two keywords. The qualified backend
also accepts CuTe register tensors for block Scan and returns ``ThreadData``;
use its conversion methods when a register-tensor result is needed.

:download:`Download the Scan example
<../../python/cuda_coop/examples/cutlass/scan.py>`:

.. literalinclude:: ../../python/cuda_coop/examples/cutlass/scan.py
   :language: python
   :start-after: docs: start cutlass-scan
   :end-before: docs: end cutlass-scan

.. _coop-cutlass-exchange:

Exchange layouts and scatter
----------------------------

``exchange(group, values, mode=...)`` rearranges fixed per-thread payloads
across a block, physical warp, or logical warp. It returns a fresh
``ThreadData`` with the same dtype and extent and preserves the input. Scalar
payloads are unsupported. Values may use the ten numeric dtypes supported by
Scan.

The common API modes are ``striped_to_blocked`` (the default) and
``blocked_to_striped``. For group rank ``t``, item index ``i``, group size ``G``,
and ``I`` items per thread, blocked layout holds tile index ``t * I + i``;
striped layout holds ``t + i * G``. Exchange changes which thread holds each
item without reading or writing global memory.

Physical and logical Warp Exchange use the same complete-warp launch and
participation requirements as Warp Load and Store, including logical widths
1, 2, 4, 8, 16, and 32. Every member of a participating group must invoke the
primitive; complete sibling groups may take different control-flow paths.
Each group has independent scratch and masked reuse synchronization. Block
Exchange requires every block thread to participate.

The qualified API adds these block-only modes:

.. list-table:: Qualified Block Exchange modes
   :header-rows: 1

   * - Mode
     - Additional requirements
   * - ``warp_striped_to_blocked``, ``blocked_to_warp_striped``
     - Convert between blocked layout and a striped layout within each
       physical warp. The block size must be divisible by 32.
   * - ``scatter_to_blocked``, ``scatter_to_striped``
     - Supply ``ranks`` giving each input item's destination in the tile.
   * - ``scatter_to_striped_guarded``
     - Supply ``ranks``; negative ranks skip the corresponding input items.
   * - ``scatter_to_striped_flagged``
     - Supply ``ranks`` and ``valid_flags``; zero flags skip input items.

Ranks must be signed 8-, 16-, 32-, or 64-bit integers. Flags may use any
signed or unsigned integer dtype; Boolean flags are unsupported. Each
auxiliary payload must have the same item count as ``values``. The caller
must ensure that participating ranks are unique and within the tile range.
Destination slots that receive no item are undefined. Guarded and flagged
scatters still require every block thread to participate, and preserve
values, ranks, and flags.

``warp_time_slicing=True`` lets block layout conversions and ordinary
scatters share scratch between physical warps. Guarded/flagged scatter and
Warp Exchange reject this option. Ordinary block layouts and scatters allow
blocks with incomplete physical-warp tails. Scratch and trailing reuse
synchronization are managed by the backend; Exchange does not accept
``temp_storage``.

.. _coop-cutlass-shuffle:

Block Shuffle
-------------

``shuffle(block, values, mode="down")`` shifts the flattened blocked payload
by one item: output item ``j`` receives input item ``j + 1``. With
``mode="up"``, it receives item ``j - 1``. The final Down item and first Up
item are undefined; repair or exclude that boundary before reading it. These
array primitives in the common API require ``distance=1`` and return a fresh
``ThreadData``, preserving the input.

The qualified API also accepts a scalar per thread with ``mode="offset"``
or ``mode="rotate"``. Offset reads the value from block rank
``rank + distance`` without wrapping; results outside the block are
undefined. Its distance may be negative or zero and must fit a signed 32-bit
integer. Rotate wraps around the block and requires
``1 <= distance < block_size`` with at least two threads. Distances may be
runtime integers and may differ between threads. Supported typed distances
are signed 8-, 16-, 32-, or 64-bit integers and unsigned 8-, 16-, or 32-bit
integers; values are checked before narrowing.

Every block thread must reach Shuffle even when only some results are used.
The backend manages scratch and reuse synchronization. Shuffle does not
accept explicit storage or prefix/suffix outputs. Warp groups are
unsupported.

This example converts blocked registers to striped registers, shifts that
payload down, repairs its final boundary, and stores the result. It checks
the layout and shift against an independent CPU reference.
:download:`Download the Exchange and Shuffle example
<../../python/cuda_coop/examples/cutlass/exchange_shuffle.py>`:

.. literalinclude:: ../../python/cuda_coop/examples/cutlass/exchange_shuffle.py
   :language: python
   :start-after: docs: start cutlass-exchange-shuffle
   :end-before: docs: end cutlass-exchange-shuffle

.. _coop-cutlass-merge-sort:

Built-in Merge Sort
-------------------

``merge_sort_keys(group, keys, ...)`` sorts a group's tile in ascending
order; ``descending=True`` reverses the order. ``merge_sort_pairs(group,
keys, values, ...)`` carries each value with its key. Both inputs and
results use blocked layout. These primitives preserve their inputs and
return fresh ``ThreadData`` payloads with the same dtypes and item counts;
the pairs spelling returns ``(sorted_keys, sorted_values)``. Equal keys have
no stability guarantee.

Keys and values use fixed, equal per-thread extents and may have different
numeric dtypes: signed or unsigned 8-, 16-, 32-, or 64-bit integers, or
32- or 64-bit floats. Readable payloads do not need mutable item access.
Floating keys must obey a strict weak ordering; NaN ordering is not defined.
The qualified API additionally accepts CuTe register tensors and immutable
register values, including mixed ``ThreadData`` and register-tensor pairs.
Results remain ``ThreadData``. Custom comparison callbacks are unsupported.

Block Merge Sort requires a power-of-two total thread count; multidimensional
blocks are supported. Physical warps and logical widths 1, 2, 4, 8, 16, and
32 require complete enclosing physical warps. Every member of a participating
group must call the primitive with uniform controls. The sort applies to each
group independently, rather than to the entire array.

For a partial tile, provide both ``valid_items`` and ``oob_default``.
The valid prefix contains between zero and the tile capacity, counting items
in blocked order. The sentinel must sort after valid keys: use an upper bound
for ascending order or a lower bound for descending order. Only the first
``valid_items`` output positions are defined. All group members participate,
including those with no valid items.

The count may be a runtime signed integer up to 64 bits or unsigned integer
up to 32 bits. Counts outside the tile range are rejected before narrowing.
A typed sentinel must match the key dtype exactly; ordinary Python numeric
literals must be representable in that dtype. ``descending`` is a compile-time
Boolean. Counts and sentinels must be uniform within each group.

Block sorts accept ``temp_storage`` with the size, alignment, sharing, and
reuse rules described above. Warp sorts manage independent scratch per group
and reject explicit storage. The example below reuses one descriptor for an
ascending key sort and a descending pair sort, then stores the original
payloads to verify that they remain unchanged. It checks key order and
key/value association against independent CPU references.
:download:`Download the Merge Sort example
<../../python/cuda_coop/examples/cutlass/merge_sort.py>`:

.. literalinclude:: ../../python/cuda_coop/examples/cutlass/merge_sort.py
   :language: python
   :start-after: docs: start cutlass-merge-sort
   :end-before: docs: end cutlass-merge-sort

.. _coop-cutlass-radix:

Radix Sort and Rank
-------------------

``radix_sort_keys(block, keys, ...)`` returns sorted keys, and
``radix_sort_pairs(block, keys, values, ...)`` returns sorted keys and their
associated values. ``radix_rank(block, keys, ...)`` instead returns each
input item's position in the order of a selected digit. It preserves the
input arrangement; use the returned ranks when assigning destinations.
All three primitives preserve their inputs. Equal selected digits retain
flattened blocked input order in both ascending and descending modes.

These primitives require a complete physical block, including multidimensional
blocks. Every block thread participates with identical controls and per-thread
extents. A block tile contains at most 65,535 items. Warp and mapped groups
are unsupported. Inputs use blocked layout,
and array results are fresh ``ThreadData`` payloads with the same item count.
Read-only inputs are accepted. Sort preserves the key and value dtypes;
Rank returns signed ``cutlass.Int32`` values. The
:doc:`Radix visualization <coop/visualizations/radix>` illustrates the relation
between digits, ranks, and sorted positions.

The common API accepts payloads of ``int32``, ``uint32``, ``int64``, or
``uint64`` keys. Pair values may use any of the ten numeric dtypes supported
by Scan; key and value extents must match. The qualified API additionally
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

Signed integer keys invert their sign bit before selecting digits. Qualified
Sort also accepts ``float32`` and ``float64`` keys. Floating keys invert all
bits when negative and only the sign bit when nonnegative. Negative and
positive zero compare equivalently; NaNs follow transformed-bit ordering.
Returned keys retain their original bit representations.

Results use blocked layout by default. Qualified Sort's compile-time
``blocked_to_striped=True`` maps item ``i`` at thread ``t`` to sorted index
``i * block_threads + t``. This applies to both outputs of a pair sort.
Match the Store algorithm to this register layout when writing a contiguous
sorted tile. Sort accepts ``temp_storage`` with the size, alignment, sharing,
and synchronization rules described above. It does not accept a valid count,
comparison callback, or algorithm selector.

Rank digits and bin prefixes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rank keys are always 32- or 64-bit integers, including through the qualified
API. Signed keys invert the sign bit before digit extraction. Its bounds
and ``radix_bits`` must be compile-time integers, selecting one through eight
bits within the key width. Omitted ``end_bit`` means ``begin_bit + radix_bits``,
or ``begin_bit + 4`` when both controls are omitted. An explicit end and width
must describe the same interval. Defaults do not clamp to the key width.
Rank manages scratch and reuse synchronization automatically and does not
accept ``temp_storage``.

Qualified Rank adds ``exclusive_digit_prefix``, a writable signed Int32
``ThreadData`` output distinct from the keys. Its dtype may be inferred.
For digit width ``R`` and block size ``T``, each thread supplies
``P = max(1, ceil(2**R / T))`` slots. Slot ``i`` of thread ``t`` owns bin
``t * P + i`` in both directions. Each prefix counts keys with smaller digits
in ascending mode or greater digits in descending mode. Slots beyond the
number of bins are undefined and must not be read. These bin counts are
separate from the returned per-item ranks.

This example sorts full signed keys, orders pairs by their transformed high
digit, and computes that digit's inverse ranks. The qualified path also uses
striped Sort output and descending bin prefixes. CPU checks verify stable
pair order, the signed-key transformation, ranks, and input preservation.
:download:`Download the Radix Sort and Rank example
<../../python/cuda_coop/examples/cutlass/radix.py>`:

.. literalinclude:: ../../python/cuda_coop/examples/cutlass/radix.py
   :language: python
   :start-after: docs: start cutlass-radix
   :end-before: docs: end cutlass-radix

.. _coop-cutlass-topk:

TopK selection
--------------

``topk_min_keys`` and ``topk_max_keys`` select a tile's smallest or largest
keys. ``topk_min_pairs`` and ``topk_max_pairs`` also carry each selected value
with its key. Controls follow the :func:`common TopK contract
<cuda.coop.topk_min_keys>`: provide ``k``, optionally limit the input with
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

The qualified API additionally accepts CuTe register tensors and immutable
register values, including mixed register-tensor and ``ThreadData`` pairs.
Its results remain ``ThreadData``, and its controls match the common API.
Scalar inputs are unsupported. Block scratch follows the size, alignment,
sharing, and reuse synchronization rules described above.

The :doc:`TopK visualization <coop/visualizations/topk>` shows the selected
prefix. This example selects the smallest keys and largest pairs from a
partial tile, reusing scratch and preserving both original payloads. Its CPU
checks compare selected multisets and pair identities without assuming
output order or stable ties. The qualified path also demonstrates register
payload conversion.
:download:`Download the TopK example
<../../python/cuda_coop/examples/cutlass/topk.py>`:

.. literalinclude:: ../../python/cuda_coop/examples/cutlass/topk.py
   :language: python
   :start-after: docs: start cutlass-topk
   :end-before: docs: end cutlass-topk

.. _coop-cutlass-register-payloads:

Qualified register payloads
---------------------------

Import ``cuda.coop.cutlass`` as ``coop`` when a kernel needs CuTe register
conversions. The qualified ``ThreadData`` provides ``from_register_tensor``
and ``to_register_tensor`` for register-memory tensors, and ``from_vector``
and ``to_tensor_ssa`` for immutable register values. These conversions use the
same fixed per-thread item count as the load/store payload.

An initialized ``ThreadData`` can cross CuTe runtime loops and branches while
retaining its fixed item count, dtype, and requested alignment. Initialize
every item in every participating thread before carrying the payload across
a runtime control-flow boundary.

Load and Store infer the memory element type from the CuTe operand. If a
producer or tensor adapter loses the intended unsigned element type, use
``cute.recast_tensor`` to restore that type before passing the tensor to
``load`` or ``store``.

.. code-block:: python

   import cuda.coop.cutlass as coop

   # Inside a CuTe kernel, with a register-memory fragment:
   values = coop.ThreadData.from_register_tensor(fragment)
   coop.store(coop.this_block(), destination, values)
