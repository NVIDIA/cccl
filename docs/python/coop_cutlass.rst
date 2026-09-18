.. _coop-cutlass:

``cuda.coop.cutlass``: CuTe DSL integration
===========================================

The CUTLASS backend provides group queries and synchronization, Load and
Store, and built-in Reduce and Sum inside CuTe DSL kernels.
It uses the same group-first calls and in-place
payload contract as :mod:`cuda.coop`: ``load`` fills an existing
``ThreadData`` and returns ``None``; ``store`` leaves its input payload
unchanged. Reductions return a scalar and preserve their input payload.

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

A compatible CuTe compiler must provide scoped trace finalization, active
compiler-environment ownership, exact launch dimensions and flags, and
external NVIDIA LTO-IR linking. Successful import checks the Python
capabilities; compiling and running a kernel also requires a working NVRTC
and terminal linker. Importing :mod:`cuda.coop` alone does not load CUTLASS
or initialize CUDA bindings.

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

The active CuTe compiler selects the backend while tracing a kernel. A
CUTLASS installation alone does not make portable operations callable on the
host. A failed optional activation reports the missing capability and leaves
the portable namespace available.

This example loads two adjacent items per thread and stores a partial tile in
the same blocked layout. ``module`` selects the portable or qualified API. The
full example defines the tile dimensions and checks the output against a CPU
reference. :download:`Download the example
<../../python/cuda_coop/examples/cutlass/block_load_store.py>` to run it with a
compatible compiler:

.. literalinclude:: ../../python/cuda_coop/examples/cutlass/block_load_store.py
   :language: python
   :start-after: docs: start cutlass-block-load-store
   :end-before: docs: end cutlass-block-load-store

Block Load and Store use the exact launch dimensions supplied by CuTe,
including multidimensional blocks. ``offset`` selects the beginning of the
block's tile and ``valid_items`` specifies the number of valid items in that
tile. Load may fill its out-of-bounds items with ``oob_default``. Without that default,
initialize any items that the valid prefix will not overwrite before reading
them. All threads in the block must call the operation with uniform controls.

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

Direct, striped, and vectorized operations emit no storage pointer or reuse
barrier, including when passed a ``TempStorage`` descriptor.
``ThreadData(alignment=...)`` requests a minimum payload alignment; it does
not change the logical item layout.

Block scratch and reuse
-----------------------

Transpose algorithms allocate scratch implicitly unless passed
``temp_storage``. Construct one ``TempStorage`` inside the kernel to share
capacity across calls. An omitted size is determined from the operations'
exact C++ storage layouts; an explicit byte capacity must accommodate all
uses. ``alignment`` is a minimum: the allocation also satisfies the C++
operations' alignment requirements.

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
call the operation with uniform controls. Different warps may use different
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

Every member of a participating logical group must reach its collective with
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
launch flags supplied by the compiler. Cluster operations require consistent
cluster dimensions and launch mode; grid queries also require exact grid
dimensions.

Built-in Reduce and Sum
-----------------------

``reduce(group, value, ...)`` and ``sum(group, value, ...)`` accept a scalar
or fixed per-thread ``ThreadData`` payload. Full-group reductions support
thread, physical and logical warp, block, mapped groups of physical warps,
and cluster groups. Grid reductions are unsupported. All members of a
participating group must invoke the collective.

For a mapped group of physical warps, every thread in the enclosing block
must reach the reduction, including nonmembers of a non-exhaustive partition:
its collective setup synchronizes the parent block. Restrict use of the
result to participating members; do not guard the collective itself with
``is_member()``.

The built-in operators are sum, product, minimum, maximum, bitwise AND,
bitwise OR, and bitwise XOR. For example, ``binary_op="max"`` selects maximum,
and ``binary_op="bit_or"`` selects bitwise OR. An omitted operator selects
sum. Bitwise operators require integer values. The qualified API also accepts
known ``operator`` and NumPy callable aliases; arbitrary callbacks are
unsupported.

With the default ``broadcast=True``, every group member may use the scalar
result. With ``broadcast=False``, only group rank zero may use it; the other
members must still call the collective. Nonmembers of a non-exhaustive mapped
group have no defined result. The input payload remains unchanged.

Full-group operations without algorithm controls use CUDAX. An explicit block
algorithm or ``valid_items`` selects CUB and requires ``broadcast=False``:

.. list-table:: Direct reduction controls
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

Keep compiler-owned payloads within their originating DSL. Separate kernels
may use CUTLASS and Numba-CUDA-MLIR in the same process; their register values
and type systems are not interchangeable.
