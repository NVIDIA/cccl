.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-programming-concepts:

Programming concepts
====================

Cooperative operations let a group of threads work on data together.
This page explains the participating threads, their payloads and
layouts, and the rules for results and temporary storage.

See the :doc:`Numba-CUDA-MLIR Programming Guide <programming_guide>`
for complete kernels and launch examples.

.. _coop-backend-registration:

Registering a backend
---------------------

Call :func:`cuda.coop.register` on the host before compiling kernels to
select the backend explicitly:

.. code-block:: python

   from cuda import coop

   coop.register("numba-cuda-mlir")

   from numba_cuda_mlir import cuda

Registration loads the backend and installs its compiler hooks, so this
works regardless of whether ``cuda.coop`` or Numba-CUDA-MLIR was imported
first. Repeated calls are safe and return ``None``. The spelling
``"numba_cuda_mlir"`` is also accepted. Registration requires the backend's
dependencies to be installed; it does not install packages.

For convenience, importing ``cuda.coop`` after ``numba_cuda_mlir`` also
registers the backend automatically. A standalone ``cuda.coop`` import does
not discover or load optional compilers. Explicit registration is useful in
libraries and notebooks where another import may already have loaded
``cuda.coop``.

Importing the backend namespace also registers it:

.. code-block:: python

   import cuda.coop.numba_mlir as numba_coop

Use ``numba_coop`` when mixing common and backend calls. If your program uses
only the backend namespace, you can import it as ``coop`` instead. See the
:ref:`namespace FAQ <coop-faq-numba-only>` and
:ref:`API comparison <coop-programming-api-choice>`.


Kernel API
----------

The common root and qualified backend expose matching entry points:

.. code-block:: python

   from numba_cuda_mlir import cuda, types

   from cuda import coop

   # Inside a Numba-CUDA-MLIR kernel:
   block = coop.this_block()
   items = coop.ThreadData(2)
   tile_items = cuda.blockDim.x * 2
   tile_offset = cuda.blockIdx.x * tile_items
   valid_items = count - tile_offset
   if valid_items < 0:
       valid_items = 0
   elif valid_items > tile_items:
       valid_items = tile_items
   coop.load(
       block,
       source,
       items,
       valid_items=valid_items,
       oob_default=0,
       offset=tile_offset,
   )
   coop.store(
       block,
       destination,
       items,
       valid_items=valid_items,
       offset=tile_offset,
   )

For a complete copy kernel, including allocation, launch, tail handling, and
an output check, see the example in :func:`cuda.coop.load`. The API reference
also includes tested examples for :func:`cuda.coop.store`,
:func:`cuda.coop.ThreadData`, and :func:`cuda.coop.TempStorage`.

Use the qualified namespace when backend-specific types or controls are
required:

.. code-block:: python

   import cuda.coop.numba_mlir as coop

Both spellings are compiler markers. Calls must occur in a compatible compiler
context; they are not host-side data movement operations.


Groups and thread data
----------------------

:func:`cuda.coop.this_block` describes the current CUDA thread block, and
:func:`cuda.coop.this_warp` describes the current 32-thread physical warp. A
physical warp can be partitioned with ``this_warp().group_by(width)`` into
consecutive logical warps of 1, 2, 4, 8, 16, or 32 threads. Load and Store
support all three forms. The enclosing block must contain a multiple of 32
threads, with no incomplete final physical warp. For a multidimensional block,
threads are linearized in x-major order. Every member of a participating group
must reach its collective; complete sibling logical groups may take different
control-flow paths.

The common group vocabulary also includes thread, cluster, grid, and mapped
groups of physical warps, but those are not Load or Store targets.
``ThreadGroup`` objects are descriptor-only in this release. ``group_by`` is
compile-time vocabulary for describing a static partition. Runtime query,
membership, and synchronization methods such as
``rank``, ``count``, ``rank_as``, ``count_as``, ``sync``, ``sync_aligned``, and
``is_member`` are not exposed.


Participation and synchronization
---------------------------------

Every member of a participating group must reach the same cooperative call.
A branch around a block operation must be uniform across the block; a branch
around a logical-warp operation must be uniform within that logical warp.
Complete sibling logical groups may follow different paths. Warp operations
require a block size divisible by 32, with no incomplete final physical warp.

Do not put a block Load or Store inside a per-element ``if index < count``
condition. Use ``valid_items`` to describe the valid prefix while all block
threads participate. An early return by some threads also violates participation
if the remaining threads later execute a block operation.

A scratch-reuse barrier protects temporary storage. Its presence depends on
the algorithm and storage policy; arrange explicit synchronization wherever
application-owned shared memory requires it. A barrier does not make
divergent participation safe.


Per-thread payloads
-------------------

``ThreadData(items_per_thread, dtype=None, *, alignment=None)`` describes the
fixed-size register payload owned by each participating thread. Common and
qualified calls use the same inference rules: an untyped Load output infers
its dtype from the source, and Store combines the destination dtype with
payload writes. Load fills the supplied output in place and returns ``None``.

Both namespaces accept ``alignment`` as a compile-time positive power of two
in bytes. It specifies minimum alignment when the compiler materializes
payload storage; ``None`` lets the compiler choose. For example,
``coop.ThreadData(4, dtype=np.float32, alignment=16)`` requests at least
16-byte alignment. The backend may use stronger alignment, including for
requests smaller than its minimum allocation alignment. This option does not
assert alignment of source or destination arrays passed to Load or Store.

Payload slots start uninitialized. Write every slot before reading it; a
partial Load needs ``oob_default`` or previously initialized values for its
invalid slots. :class:`cuda.coop.ThreadDataLike` names the shared payload
interface in type signatures. Protocol compatibility alone does not make an
arbitrary Python object a supported kernel value.

The payload's ``items_per_thread`` attribute is a compile-time integer and
can be used as a loop bound inside a kernel, including through payload aliases.

Supported payload types are signed and unsigned 8-, 16-, 32-, and 64-bit
integers plus 32- and 64-bit floating-point values. Boolean, 16-bit floating
point, complex, and mismatched payload types are rejected before NVRTC
compilation.


Load and Store semantics
------------------------

The signatures are:

.. code-block:: python

   load(
       group, source, output, /, *,
       algorithm="direct",
       valid_items=None,
       oob_default=None,
       offset=None,
       temp_storage=None,
   ) -> None

   store(
       group, destination, value, /, *,
       algorithm="direct",
       valid_items=None,
       offset=None,
       temp_storage=None,
   ) -> None

``valid_items`` counts the valid prefix of the selected group tile, not the
number of valid items per thread. A Warp-group tile contains
``group_size * items_per_thread`` elements, where ``group_size`` is 32 for
``this_warp()`` or the width passed to ``group_by``. The count must be uniform
within that group. With Load, invalid output slots are unspecified unless
``oob_default`` is supplied, even if initialized before Load. A runtime
default must also be uniform within the group. A default is valid only when ``valid_items`` is present. With Store,
elements outside the valid prefix are not written.

.. warning::

   ``valid_items`` must satisfy
   ``0 <= valid_items <= group_size * items_per_thread``. Static values outside
   that range are rejected while planning. Runtime values are checked rather
   than saturated; do not rely on CUB's oversized-count behavior. An invalid
   runtime value executes a deterministic device trap before narrowing to
   CUB's integer parameter, and that trap poisons the current CUDA context.
   Clamp grid-stride and tail counts as in the example above. Run intentional
   failure probes in disposable processes. For a Warp-group call, a block-wide
   remainder is not a valid count: subtract that group's tile origin and clamp
   the result to ``[0, group_size * items_per_thread]``.

``offset`` is an element offset into the source or destination. It is
independent of ``valid_items`` and is not measured in bytes. The value must be
uniform within each participating group; different groups may use different
offsets. Static offsets must be nonnegative; a runtime offset is a
caller-enforced nonnegative precondition. Source and destination arrays must be
one-dimensional and contiguous. Store accepts both scalar values and
multi-item ``ThreadData`` payloads.

Runtime ``valid_items`` and ``offset`` accept signed integer types through 64
bits and unsigned integer types through 32 bits. Boolean, floating-point, and
``uint64`` runtime values are rejected. A runtime ``oob_default`` is already
typed by the compiler and must exactly match the Load payload dtype. Ordinary
Python integer and floating-point literals are converted contextually and
range-checked against that dtype before provider generation.

For a Warp group of width ``group_size``, the compiler first advances the
memory base by
``group_index * (group_size * items_per_thread)`` and then applies the caller's
``offset``. The group index is the x-major linear thread rank divided by the
group size, so every physical or logical Warp group in a block addresses a
distinct tile. In a multi-block traversal, the caller offset must also include
the block's global tile origin. Do not add the compiler-provided group origin
again. Runtime offsets must leave enough signed 64-bit range for the last group
origin in the block; static offsets are checked during planning.

Store payloads must have exactly the destination dtype. Numba-CUDA-MLIR may
promote integer arithmetic even when its operands are 32-bit. Cast a computed
value explicitly before storing it:

.. code-block:: python

   value = types.int32(source[cuda.threadIdx.x] + 1)
   coop.store(block, destination, value, algorithm="direct")


Data layouts and algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Both common and qualified entry points use the same string algorithm
vocabulary: ``direct``, ``striped``,
``vectorize``, ``transpose``, ``warp_transpose``, and
``warp_transpose_timesliced``. All six algorithms are executable with the
Numba-CUDA-MLIR backend. ``direct`` and ``vectorize`` use blocked ordering, so
each thread owns a contiguous segment of the tile. ``striped`` exposes striped
ordering, where item ``i`` for a thread is separated from its next item by the
block size. The three transpose algorithms use striped memory transactions but
present blocked ``ThreadData`` to the caller. The two warp-transpose variants
perform that reordering within each warp and require a block size divisible by
32.

Physical and logical Warp Load and Store support ``direct``, ``striped``,
``vectorize``, and ``transpose``. Their layouts follow the same rules at the
selected group width: ``direct`` and ``vectorize`` expose blocked payloads,
``striped`` exposes a striped payload, and ``transpose`` uses striped memory
transactions while exposing a blocked payload. Common and qualified calls
use the same lowercase string selectors. Selectors are normalized to lowercase
underscore-delimited strings. Enum and integer selectors, including ``0``, are
rejected.

For group size ``G``, items per thread ``K``, thread rank ``t``, and item index
``i``, blocked order uses tile position ``t * K + i``; striped order uses
``t + i * G``. The payload has no runtime layout tag that corrects a mismatched
Load/Store pair.

Store consumes the arrangement associated with its selected algorithm. The
transpose Store implementations copy the payload before calling CUB, so Store
never modifies the caller's scalar or ``ThreadData`` value while CUB performs
its in-place reordering.


Temporary storage
-----------------

Block Load and Store accept an optional ``TempStorage`` descriptor.

Scratch belongs to one block during its kernel execution. A descriptor does
not carry data between blocks or kernel launches.

For example:

.. code-block:: python

   scratch = coop.TempStorage(
       size_in_bytes=None,
       alignment=None,
       auto_sync=False,
       sharing="shared",
   )

Only ``size_in_bytes`` may be positional; ``alignment``, ``auto_sync``, and
``sharing`` are keyword-only. As with ``ThreadData``, ``alignment=None`` lets
the compiler choose. An explicit positive power of two sets minimum alignment
in bytes. The planner may strengthen it to satisfy every primitive using the
storage. Integer-like values implementing ``__index__`` are accepted. An
explicit ``size_in_bytes`` must still be large enough for the planned storage.

For block, physical Warp, and logical Warp operations, ``direct``, ``striped``,
and ``vectorize`` are storage-free. They default-construct the CUB primitive,
report zero temporary bytes, and emit no shared-memory allocation, storage
pointer, or synchronization barrier. For a block call, an explicit descriptor,
including an unsized descriptor, is validated as compile-time vocabulary but
does not change code generation for those algorithms.
Construct ``TempStorage`` inside the kernel; the current Numba-CUDA-MLIR
frontend does not resolve module-global storage descriptors. A descriptor may
be passed to a device function that Numba-CUDA-MLIR inlines into the kernel,
which is the default, but it cannot cross into a separately compiled device
function.

The block ``transpose``, ``warp_transpose``, and ``warp_transpose_timesliced`` use CUB
temporary storage. Without a descriptor, the compiler allocates the
specialization's exact storage and inserts a block reuse barrier. An explicit
descriptor selects shared or exclusive ownership, requests capacity and
alignment, or opts into dynamic shared memory. The provider remains
authoritative for the required byte count and alignment, and the backend
validates the descriptor against the concrete lowering plan.

Sharing selects only the slice layout: ``sharing="shared"`` overlaps every call
that passes the same descriptor on one region, while ``sharing="exclusive"``
gives each call site its own slice. A call site inside a loop reuses its slice
under either layout, so ``auto_sync`` is independent of ``sharing`` and
defaults to ``False`` for both.

The synchronization model is deliberately simple. A descriptor names one
region; distinct descriptors and compiler-owned storage never alias each
other. With ``auto_sync=True``, the compiler appends
``cuda.syncthreads()`` for block groups or ``cuda.syncwarp(mask)`` for Warp
groups immediately after every call that consumes the storage, including the
last one, and never inserts a barrier before a call. That trailing barrier
orders reuse of the temporary storage. Its insertion depends on scratch use,
so arrange explicit barriers for application-owned shared memory. It disappears when
``auto_sync=False`` (the default). The caller issues
``cuda.syncthreads()`` between consecutive uses of the descriptor, and a call
site inside a loop counts as a reuse on every iteration. Compiler-owned storage
always synchronizes.

The compiler stages every descriptor and every compiler-owned requirement of a
kernel into one shared-memory backing. When that backing exceeds the 48 KiB
static limit, through an explicit ``size_in_bytes`` or through large implicit
requirements, it moves to dynamic shared memory and the launch reserves the
exact byte count. Supported Numba-CUDA-MLIR releases do not separate static and dynamic shared
allocations reliably. A kernel using cooperative temporary storage must not
also declare a zero-sized or runtime-sized ``cuda.shared.array``. When
cooperative backing becomes dynamic, user static shared arrays are also
unsupported. Keep both user arrays and cooperative backing static, or move the
user data out of shared memory. Storage-free operations do not add this
restriction.

With ``auto_sync=False``, a descriptor must originate from exactly one
constructor site. Selecting between multiple manual-sync constructors is an
MVP restriction: the compiler cannot prove that caller barriers protect the
merged region, even when a particular program supplies sufficient barriers.

Cooperative calls in device helpers must be inlined into the kernel; use
``@cuda.jit(device=True, inline="always")`` when selecting the helper's
policy explicitly. Standalone collective helpers and collectives inside
standalone callbacks are unsupported. For the MVP, ``literal_unroll``
values cannot determine cooperative payload extents, group dimensions,
selectors, or descriptor constructor arguments. Write separate calls with
explicit constants, or use an ordinary loop with one fixed cooperative shape.
An unrelated ``literal_unroll`` loop does not add this restriction.

Warp ``transpose`` uses compiler-owned storage with one disjoint slice per
physical or logical group. The compiler inserts ``syncwarp`` with the exact
logical-group mask. Both the common and qualified APIs reject explicit
``TempStorage`` for every Warp Load and Store algorithm, including the
storage-free modes.
