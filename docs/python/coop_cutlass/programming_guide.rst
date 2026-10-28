.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-cutlass:
.. _cuda.coop.cutlass.programming_guide:
.. _cuda-coop-cutlass-cute-dsl-integration:

Programming Guide
=========================

Use ``cuda.coop.cutlass`` inside a CuTe kernel to move data, reduce or scan
values, rearrange payloads, sort or select keys, compare neighbors, count
samples, and decode runs. These operations compile to CUB device functions.
The :doc:`API reference <api>` lists signatures and result types.

.. code-block:: python

   import cuda.coop.cutlass as coop

Each thread keeps its items in a ``ThreadData`` object.
``load(group, source, items_per_thread=N)`` returns that payload;
``store`` writes its items to memory. The :doc:`overview <index>` covers
installation, :doc:`concepts` explains the execution rules, and the
:doc:`Developer Guide <developer_guide>` follows compilation and linking.

.. _coop-cutlass-api-choice:

Using the API
-------------

Importing ``cuda.coop.cutlass`` activates its CuTe compiler integration.
Call the primitives inside ``@cute.kernel`` or a ``@cute.jit`` function
called by that kernel. Compiler markers do not execute operations on the host.

.. _coop-cutlass-differences:

.. _cutlass-specific-behavior-and-current-limits:

CuTe values and supported features
----------------------------------

Use ``ThreadData`` to work with CuTe register tensors.
``ThreadData.from_register_tensor(fragment)`` copies a fragment into a
payload you can pass to ``store`` or another primitive.
``values.to_register_tensor()`` converts a payload back to a CuTe register
tensor. See :ref:`coop-cutlass-register-payloads`.

Construct payloads with ``coop.ThreadData`` inside a CuTe kernel.
They support writable Scan aggregates and Radix Rank prefixes, along with
CuTe register-tensor and immutable-vector conversions. A Python object does
not become a supported kernel payload merely by implementing a typing
protocol.

Group queries return CuTe scalars. For example, ``block.rank()`` returns a
``cutlass.Uint32`` that you can use in pointer arithmetic or a condition
inside the kernel. Use ``block.rank_as(cutlass.Int32)`` when you need a signed
rank.

All threads in the group must call the primitive, even when ``valid_items``
selects a short tile or only rank zero uses the result. The sections below
describe the requirements for block, warp, and mapped groups.

Reduce and Scan support the built-in operators listed below. Custom operators
and Scan prefix callbacks are not yet supported. See the :doc:`API reference
<api>` for operation signatures and return values.

.. _coop-cutlass-requirements:

Runtime requirements
--------------------

Install the standalone ``cuda-coop-cutlass`` distribution and a compatible
CUTLASS CuTe compiler as described in :doc:`index`. The supported compiler
integration requires external NVIDIA LTO-IR linking, and NVRTC compiles the
CUB providers. See :ref:`compiler requirements
<coop-cutlass-compiler-requirements>` for the required interfaces.

.. _coop-cutlass-load-store:

A complete prefix-sum example
-----------------------------

This example computes exclusive prefix sums across a block: each output is
seven plus the sum of all preceding input values. For input
``[0, 1, 2, 3, ...]``, it produces ``[7, 7, 8, 10, ...]``. Load returns the
per-thread payload, Scan returns a new payload, and Store writes the result.
All three operations reuse one ``TempStorage`` with automatic synchronization.

The kernel takes ``items_per_thread`` as a compile-time argument. The full
example launches one ``(8, 4, 2)`` block and defaults to two adjacent items
per thread, giving 128 input values. Its host entry point checks the output against NumPy.
:download:`Download the example
<../../../python/cuda_coop_cutlass/examples/scan.py>` to run it with a compatible
compiler:

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/scan.py
   :language: python
   :start-after: docs: start cutlass-scan
   :end-before: docs: end cutlass-scan

Block Load and Store support one-, two-, and three-dimensional blocks.
``offset`` selects the beginning of the block's tile and ``valid_items``
specifies the number of valid items in that tile. Load may fill its
out-of-bounds items with ``oob_default``. Without that default, items beyond
``valid_items`` have unspecified values after Load, even if you initialized
them before the call. Supply ``oob_default`` or write those items after Load
before you read them. All threads in the block must call the primitive with
uniform controls.

.. _coop-cutlass-payload-types:

Payloads, dtypes, and result ownership
--------------------------------------

``ThreadData`` holds a fixed number of values per thread. In blocked order,
thread ``t`` owns tile positions ``t * items_per_thread + i``. In striped
order, it owns positions ``t + i * group_size``. A Load/Store algorithm
determines the layout expected by that call; the payload does not carry a
layout tag. See the :doc:`Load <visualizations/load>` and
:doc:`Exchange <visualizations/exchange>` visualizations for the mappings.

Pass the item count as ``items_per_thread: cutlass.Constexpr`` on the
``@cute.kernel`` and its ``@cute.jit`` launcher, then load values with
``coop.load(group, source, items_per_thread=items_per_thread)``.
Forward the host value through the launcher to the kernel. CuTe specializes
the count during compilation;
it remains fixed while the kernel executes. Use
``cutlass.range_constexpr(items_per_thread)`` when indexing each slot.

Load infers the payload's element type from the memory operand. For generated
values and writable auxiliary outputs, construct
``coop.ThreadData(items_per_thread)`` directly. Leave its element type
unspecified for normal use; consuming primitives can infer it from
homogeneous initialized values. Use typed scalar assignments, such as
``cutlass.Int32(expression)``, when the computation needs a specific numeric
representation. Supported payload types are signed and unsigned integers of
8, 16, 32, or 64 bits and 32- or 64-bit floating point. An individual primitive
can accept a smaller set; for example, bitwise operators require integers.
Boolean, half-precision, complex, and structured payloads are unsupported.

NumPy types select a numeric representation; expressions inside the kernel
are CuTe values. Store requires the payload dtype to match the destination
element type. Cast arithmetic results explicitly when necessary, for example
with ``cutlass.Int32(value)``. Integer sums can overflow, and a parallel
floating-point sum can differ from a sequential CPU sum because the order
of additions differs.

Load with ``items_per_thread`` returns a fresh payload. Both typed and
dtype-free constructors remain supported for generated values and auxiliary
outputs. CuTe reconstructs payload values across runtime control flow; an
alias bound before a runtime loop does not become a mutable storage reference
to the loop's payload.

Store returns ``None``. Transpose Store
algorithms may rearrange the input payload; copy values before Store if they
are needed later. For payload
inputs, Scan, Exchange, sorting, selection, and array Shuffle return fresh
payloads. Supported scalar forms return CuTe scalars. Reduce returns one
scalar even when each thread contributes multiple items. Read results
only at the positions or threads where the primitive defines them.

Index payloads with compile-time integers and initialize each item before
reading it. ``coop.load(group, source, items_per_thread=N, alignment=16)``
requests at least 16-byte alignment when the returned payload's storage is
materialized. Its alignment follows ``ThreadData(N, alignment=16)``: a positive
compile-time power of two, preserved across payload reconstruction and register
exports. Input and output memory alignment is separate; the kwarg makes
no promise about the source pointer, including a pointer with an element offset.

The compiler decides which values remain in registers
and which spill to local memory. The :ref:`conversion methods
<coop-cutlass-register-payloads>` connect payloads to CuTe register tensors
and immutable register values.

``ThreadData`` holds typed scalar expressions. Load allocates a temporary
CuTe register tensor for the provider's result and builds the returned
payload from its scalar expressions. To write an existing native register
tensor, use ``tensor.store(values.to_tensor_ssa(like=tensor))`` with matching
shape and item order. The tensor retains its native storage and aliases.
``values.to_register_tensor()`` instead creates a new register tensor.

.. _coop-shared-load-store:

Shared-memory staging
~~~~~~~~~~~~~~~~~~~~~

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

.. _coop-cutlass-dtype-inference:

Element types in advanced interop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Raw MLIR integer values may carry a width such as ``i32`` without signedness.
The adapter cannot determine whether that value represents a signed or
unsigned integer from its width alone. Preserve the intended type in the
producer or wrap the value in the appropriate CUTLASS scalar type. Explicit
payload element-type metadata is also available for this case and must match
the integer width.

An explicit type does not initialize missing items or make conflicting typed
values compatible. The ordinary Load and typed-assignment examples retain
enough information to infer their payload types.

.. _coop-cutlass-helpers:

Helpers and compile-time values
-------------------------------

A ``@cute.jit`` helper called by a ``@cute.kernel`` can contain cooperative
operations. Its calls are traced in the enclosing kernel's compiler
environment and contribute to that kernel's provider bundle and scratch
requirements. Every required group member must reach a cooperative call,
including when it appears in a helper or loop.

Payload extents, group mappings, dtype selectors, algorithm names, and
``TempStorage`` constructor options must be known while tracing. Use
``cutlass.range_constexpr`` when a loop index selects payload items or
constructs different static calls. Runtime loops may repeat a fixed call
shape, and an initialized ``ThreadData`` can pass through CuTe runtime
branches and loops. Scalar controls such as ``valid_items`` and ``offset``
may be runtime values where the primitive allows them.

A helper containing a cooperative call is different from an operator passed
to Reduce, Scan, or Merge Sort. CUTLASS currently accepts the documented
built-in operators and aliases; it does not compile arbitrary device
callbacks.

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
Returning Load's ``alignment=...`` and
``ThreadData(items_per_thread, alignment=...)`` request a minimum payload
alignment; neither changes the logical item layout or guarantees vectorized
memory accesses.

.. _coop-cutlass-storage:

Block scratch and reuse
-----------------------

Block transpose algorithms and Block Reduce allocate scratch implicitly
unless passed ``temp_storage``. Construct one ``TempStorage`` inside the
kernel to share capacity across calls. An omitted size lets the compiler
allocate enough storage for all uses; an explicit byte capacity must
accommodate them.
``alignment`` is a minimum: the allocation also satisfies each primitive's
alignment requirements.

Scratch belongs to one kernel execution on one block. It cannot preserve
application state between blocks or launches. CUTLASS materializes scratch
through CuTe's shared-memory allocator after tracing has collected the
requirements; CuTe owns the resulting kernel's shared-memory accounting.
See :ref:`the allocation walkthrough <coop-cutlass-scratch-allocation>` for
the compiler path.

``sharing="shared"`` reuses one slice across call sites. With
``sharing="exclusive"``, distinct call sites receive separate slices. This
uses more shared memory to avoid barriers needed solely for cross-call scratch
reuse when automatic synchronization is disabled. Both policies default to
``auto_sync=False``. The kernel must call ``storage.sync()`` before reusing
that storage, including on the next loop iteration. Set ``auto_sync=True`` to
insert trailing reuse synchronization after each storage-using call. Without
an explicit descriptor, the compiler manages scratch and its reuse
synchronization automatically.

By default, the following example transforms eight independent tiles with
``sharing="shared"`` and ``auto_sync=True``. Its ``run_example`` function also
accepts ``sharing="exclusive"``, and ``manual_sync=True`` replaces automatic
synchronization with ``storage.sync()`` calls.
:download:`Download the storage example
<../../../python/cuda_coop_cutlass/examples/block_storage.py>`:

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/block_storage.py
   :language: python
   :start-after: docs: start cutlass-block-storage
   :end-before: docs: end cutlass-block-storage

Typed workspaces for native CuTe code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``storage.reserve(num_elems, dtype, alignment=None)`` returns an ordinary
one-dimensional CuTe shared tensor. Its element count, dtype, and optional
byte alignment are compile-time constants. Supported dtypes are 8/16/32/64-bit
signed and unsigned integers, ``Float16``, ``BFloat16``, ``Float32``, and
``Float64``, including equivalent NumPy types where available.

Reservations inherit the descriptor's sharing policy. With ``sharing="shared"``,
all reservations and cooperative scratch using that descriptor alias one
region sized for the largest requirement. With ``sharing="exclusive"``,
each reservation and primitive call site receives a separate region.
Different descriptors and ordinary ``SmemAllocator`` allocations remain
disjoint. CuTe accounts for their combined size at launch.

Descriptors used for reservations require ``auto_sync=False``. Before a new
phase reuses shared bytes, finish the previous phase's accesses and library
operations, then synchronize participating threads as required. In particular,
load inputs into registers and synchronize before invoking a cooperative
primitive whose scratch aliases those inputs. A block barrier does not replace
a library's asynchronous completion or release operation.

A reserved tensor with a supported Load/Store dtype can also serve as the
shared source or Store destination in the staging example above. Keep live
staging data separate from scratch that an operation may overwrite. Direct
and striped Load/Store need no cooperative scratch; a transpose operation
requires separate scratch while it reads or writes that live staging tensor.

The following example passes a reserved tensor to native CuTe code, reuses its
bytes for a cooperative reduction, and checks that a separate native shared
array retains its values. :download:`Download the workspace example
<../../../python/cuda_coop_cutlass/examples/shared_workspace.py>`:

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/shared_workspace.py
   :language: python
   :start-after: docs: start cutlass-shared-workspace
   :end-before: docs: end cutlass-shared-workspace

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
``32 * I``. As with Block Load, payload items outside that prefix are
unspecified unless ``oob_default`` is supplied, even if initialized before
Load. Store writes only the valid prefix and may rearrange its input payload
when using a transpose algorithm.

This example uses two physical warps in an ``(8, 4, 2)`` block and checks the
independent partial tiles against a CPU reference.
:download:`Download the Warp example
<../../../python/cuda_coop_cutlass/examples/warp_load_store.py>`:

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/warp_load_store.py
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
items. Default filling and unspecified invalid items follow the same rules
as physical Warp Load.

Every member of a participating logical group must call the primitive with
uniform controls. Complete sibling groups may take different control-flow
paths or use different offsets and valid counts. Transpose reuse
synchronization is masked to the participating logical group. Explicit
``temp_storage`` remains unsupported, and nested partitions or groups of
physical warps cannot be used for Load and Store.

This example partitions the two physical warps in an ``(8, 4, 2)`` block into
eight groups of eight threads, each loading its own partial tile.
:download:`Download the logical Warp example
<../../../python/cuda_coop_cutlass/examples/logical_warp_load_store.py>`:

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/logical_warp_load_store.py
   :language: python
   :start-after: docs: start cutlass-logical-warp-load-store
   :end-before: docs: end cutlass-logical-warp-load-store

.. _coop-cutlass-hierarchy:

Hierarchy queries and synchronization
-------------------------------------

``this_thread()``, ``this_warp()``, ``this_block()``, ``this_cluster()``, and
``this_grid()`` describe the corresponding physical groups. ``rank`` and
``count`` accept a hierarchy level: ``thread`` (also spelled ``gpu_thread``),
``warp``, ``block``, ``cluster``, or ``grid``. Their default result is a CuTe
``Uint32``, or ``Uint64`` when the group or queried level is the grid.
``rank_as(dtype, level="thread")`` and ``count_as`` select a signed or
unsigned 8-, 16-, 32-, or 64-bit integer type. Floating and Boolean query
types are unsupported. NumPy integer dtypes and Python ``int`` are also
accepted as dtype selectors; the compiled values are CuTe scalars.
``is_member()`` returns a CuTe ``Uint8`` membership flag.

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

.. _coop-cutlass-reduce:

Built-in Reduce and Sum
-----------------------

``reduce(group, value, ...)`` and ``sum(group, value, ...)`` use CUB for block,
physical-warp, and logical-warp reductions. Each member supplies a scalar or
a fixed per-thread ``ThreadData`` payload; every payload item contributes to
the group's result. Logical warp widths are 1, 2, 4, 8, 16, or 17 through
32, and the enclosing block must contain complete physical warps. For widths
17 through 31, construct the group with ``exhaustive=False`` and guard the
call with ``group.is_member()`` to exclude trailing lanes. CUB requires a
non-power-of-two group to be the only logical group in its physical warp.
All members of a participating group must call the primitive.

The built-in operators are sum, product, minimum, maximum, bitwise AND,
bitwise OR, and bitwise XOR. For example, ``binary_op="max"`` selects maximum,
and ``binary_op="bit_or"`` selects bitwise OR. An omitted operator selects
sum. Bitwise operators require integer values. The API also accepts
known ``operator`` and NumPy callable aliases; arbitrary callbacks are
unsupported.

Only group rank zero may use the scalar result; the other members must still
call the primitive. The input payload remains unchanged.

Block calls can select a CUB algorithm. A scalar call can limit its input to
a valid prefix:

.. list-table:: Reduction controls
   :header-rows: 1

   * - Group and input
     - Supported controls
   * - Block scalar
     - ``valid_items``, a block algorithm, and ``temp_storage``
   * - Block multi-item payload
     - A block algorithm and ``temp_storage``, without ``valid_items``
   * - Physical or logical warp scalar
     - ``valid_items``, without an algorithm selector
   * - Physical or logical warp multi-item payload
     - No ``valid_items``, algorithm selector, or explicit ``temp_storage``

Block algorithm names are ``raking_commutative_only``, ``raking``, and
``warp_reductions`` (the default). A valid prefix contains from one through
the group size contributing members, counting threads rather than payload
elements. The count must be uniform within the group. Zero and out-of-range
counts are invalid; all members still participate even when their values fall
outside the prefix.

Omitting ``temp_storage`` uses compiler-managed scratch with automatic
synchronization. Block Reduce accepts a ``TempStorage`` descriptor with the
same sizing, alignment, sharing, and synchronization policy as Block
Load/Store. An explicit descriptor defaults to ``auto_sync=False``: call
``storage.sync()`` before reusing it, or construct it with
``auto_sync=True``. Warp Reduce always uses compiler-managed scratch with
synchronization scoped to its participating lanes.

This example uses block rank queries, a full block sum, logical-warp maxima,
and a scalar valid-prefix sum whose result is read only at block rank zero.
:download:`Download the reduction example
<../../../python/cuda_coop_cutlass/examples/reduce.py>`:

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/reduce.py
   :language: python
   :start-after: docs: start cutlass-reduce
   :end-before: docs: end cutlass-reduce

.. _coop-cutlass-scan:

Built-in Scan
-------------

``scan``, ``exclusive_scan``, ``inclusive_scan``, ``exclusive_sum``, and
``inclusive_sum`` support block, physical warp, and logical warp groups. Block
primitives accept scalars and fixed multi-item payloads; warp primitives
accept one scalar per lane. A ``ThreadData(1)`` remains an array payload and
is not accepted by Warp Scan. Input values are preserved. A scalar input
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

All five Scan spellings support these controls:

.. list-table:: CUTLASS Scan controls
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
available even on lanes outside that prefix. A zero count is invalid. Block Scan also
accepts CuTe register tensors for block Scan and returns ``ThreadData``; use
its conversion methods when a register-tensor result is needed.

The :ref:`introductory prefix-sum example <coop-cutlass-load-store>` uses
``exclusive_scan`` with an initial value of seven and shares scratch with
Load and Store.

.. _coop-cutlass-exchange:

Exchange layouts and scatter
----------------------------

``exchange(group, values, mode=...)`` rearranges fixed per-thread payloads
across a block, physical warp, or logical warp. It returns a fresh
``ThreadData`` with the same dtype and extent and preserves the input. Scalar
payloads are unsupported. Values may use the ten numeric dtypes supported by
Scan.

The API modes are ``striped_to_blocked`` (the default) and
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

The API adds these block-only modes:

.. list-table:: CUTLASS Block Exchange modes
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
array primitives in the API require ``distance=1`` and return a fresh
``ThreadData``, preserving the input.

The API also accepts a scalar per thread with ``mode="offset"``
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
<../../../python/cuda_coop_cutlass/examples/exchange_shuffle.py>`:

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/exchange_shuffle.py
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
The API additionally accepts CuTe register tensors and immutable
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
A typed sentinel must match the key dtype exactly. For ordinary Python
literals, integer keys require integers within the key dtype's range.
Floating keys accept integer or floating literals within range, and allow
infinite bounds. A Python or NumPy NaN sentinel is rejected. CuTe values are
not checked for NaN; callers must supply a valid bound. ``descending`` is a
compile-time Boolean. Counts and sentinels must be uniform within each group.

Block sorts accept ``temp_storage`` with the size, alignment, sharing, and
reuse rules described above. Warp sorts manage independent scratch per group
and reject explicit storage. The example below reuses one descriptor for an
ascending key sort and a descending pair sort, then stores the original
payloads to verify that they remain unchanged. It checks key order and
key/value association against independent CPU references.
:download:`Download the Merge Sort example
<../../../python/cuda_coop_cutlass/examples/merge_sort.py>`:

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/merge_sort.py
   :language: python
   :start-after: docs: start cutlass-merge-sort
   :end-before: docs: end cutlass-merge-sort

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
``uint64`` keys. Pair values may use any of the ten numeric dtypes supported
by Scan; key and value extents must match. The API additionally
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
sorted tile. Sort accepts ``temp_storage`` with the size, alignment, sharing,
and synchronization rules described above. It does not accept a valid count,
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
:download:`Download the Radix Sort and Rank example
<../../../python/cuda_coop_cutlass/examples/radix.py>`:

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
Scalar inputs are unsupported. Block scratch follows the size, alignment,
sharing, and reuse synchronization rules described above.

The :doc:`TopK visualization <visualizations/topk>` shows the selected
prefix. This example selects the smallest keys and largest pairs from a
partial tile, reusing scratch and preserving both original payloads. Its CPU
checks compare selected multisets and pair identities without assuming
output order or stable ties. The example also demonstrates register
payload conversion.
:download:`Download the TopK example
<../../../python/cuda_coop_cutlass/examples/topk.py>`:

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/topk.py
   :language: python
   :start-after: docs: start cutlass-topk
   :end-before: docs: end cutlass-topk

.. _coop-cutlass-reduce-batched:

Batched Warp Reduction
-----------------------

``reduce_batched(warp, values)`` reduces each per-thread payload slot across
the warp independently. A payload with three features produces three
aggregates, rather than combining all features into one sum. Physical warps
and logical widths of 1, 2, 4, 8, 16, and 32 are supported. Every member of the
selected warp participates; other logical warps may take another branch.

The batch count ``B`` is the positive compile-time extent of each lane's input
payload. For ``B`` batches and ``W`` lanes, each returned ``ThreadData`` has
``ceil(B / W)`` slots. With ``output_layout="striped"``, slot ``i`` in lane
``r`` holds batch ``r + i * W``. With ``"blocked"``, it holds batch
``r * ceil(B / W) + i``. Guard reads and stores when the batch index reaches
``B``. Inputs remain unchanged, and the result retains their element type.

The operation accepts readable ``ThreadData``, CuTe register payloads,
built-in operator strings, and the built-in aliases supported by Reduce. There is
no ``temp_storage`` argument: the CUB provider uses register exchanges and
needs no shared scratch or trailing storage-reuse barrier.

:download:`Download the feature-sum example
<../../../python/cuda_coop_cutlass/examples/reduce_batched.py>`:

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/reduce_batched.py
   :language: python
   :start-after: docs: start cutlass-reduce-batched
   :end-before: docs: end cutlass-reduce-batched

See the :doc:`Batched Warp Reduction visualization
<visualizations/reduce-batched>` for the batch-to-lane mapping.

.. _coop-cutlass-register-payloads:

CUTLASS register payloads
---------------------------

``ThreadData`` provides
``from_register_tensor`` and ``to_register_tensor`` for register-memory
tensors, and ``from_vector`` and ``to_tensor_ssa`` for immutable register
values. These conversions use the same fixed per-thread item count as the
load/store payload.

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

.. _coop-cutlass-producer-interop:

Producer-owned register loads
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

To inspect generated C++, set ``CUDA_COOP_SOURCE_DUMP_DIR`` before compilation.
Use the final linked cubin to assess inlining, barriers, shared memory, and
register use. A provider's source or intermediate PTX does not establish what
remains in the kernel. Use Compute Sanitizer race checking when changing
scratch reuse or synchronization.

.. _coop-cutlass-launch-facts:

Launch dimensions and resources
-------------------------------

Specify the block dimensions in the CuTe launch, including all dimensions of a
multidimensional block. Primitives specialize for those exact dimensions. Grid
queries also need exact grid dimensions; cluster operations require exact
cluster dimensions and the corresponding launch flags. Grid reductions and
synchronization are unsupported. A maximum thread bound cannot substitute for
the actual participating group size. Missing required facts cause a
compilation error; see :ref:`the compiler launch contract
<coop-cutlass-exact-launch-facts>`.

More items per thread can increase register use, and additional scratch can
reduce the number of resident blocks. Check the compiled kernel's resource
usage as well as its execution time. Use inferred scratch capacity and
alignment unless the kernel needs an explicit allocation policy.

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

This example decodes a 64-item window starting at decoded index 3, then
reuses the scratch to write the complete stream at destination offset 5.
Each of the 32 threads supplies ``items_per_thread`` runs of length two. Bulk
decoding therefore writes ``64 * items_per_thread`` items through
``2 * items_per_thread`` internal 32-item windows. The input run extent and
each operation's decoded window extent are independent.

.. literalinclude:: ../../../python/cuda_coop_cutlass/tests/runtime/test_run_length_examples.py
   :language: python
   :start-after: example-begin
   :end-before: example-end
   :dedent: 4
