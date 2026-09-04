.. _cccl-python-coop:

``cuda.coop``: Cooperative Group Primitives
============================================

``cuda.coop`` provides cooperative CUDA primitives for Python kernel DSLs.
The initial backend integrates with Numba-CUDA-MLIR and supports Load, Store,
Exchange, Shuffle, Reduce, and Scan across their supported thread-group scopes.
Its portable descriptors and planning records let primitive families share one
dispatch, storage, and compilation model.

Installation
------------

Install the extra matching the CUDA major version used to compile the kernel:

.. code-block:: console

   python -m pip install "cuda-coop[numba-cuda-mlir-cu13]"
   # Use numba-cuda-mlir-cu12 with CUDA 12.

The base ``cuda-coop`` distribution contains the portable API, type
declarations, and a coherent bundle of CUB, Thrust, libcu++, and CUDAX headers.
Installed-wheel compilation uses that bundle by default. Development from a
CCCL source checkout uses the matching checkout headers, and
``CUDA_COOP_CCCL_ROOT`` can select another source checkout or ``cuda-coop``
header bundle. Importing :mod:`cuda.coop` does not require Numba-CUDA-MLIR or
an accessible GPU.

The Numba backend is intentionally limited to
``numba-cuda-mlir>=0.5.0,<0.6``. It currently uses a guarded compatibility shim
for private 0.5.x compiler registration APIs, so another runtime series is
rejected before compiler registries are changed. Replacing that shim with an
upstream public API is follow-up work.

Backend activation
------------------

When using the portable namespace with Numba-CUDA-MLIR, import the compiler
runtime first:

.. code-block:: python

   from numba_cuda_mlir import cuda

   from cuda import coop

Because Numba-CUDA-MLIR is already imported, importing :mod:`cuda.coop`
automatically activates its compiler hooks. A standalone :mod:`cuda.coop`
import does not discover or load optional compiler runtimes or CUDA bindings.

If :mod:`cuda.coop` was imported first, activate the backend explicitly before
compiling a kernel:

.. code-block:: python

   from cuda import coop
   import cuda.coop.numba_mlir as _coop_numba_mlir  # Activate portable calls.

Importing :mod:`cuda.coop` first and compiling without that explicit
activation is unsupported. Numba-CUDA-MLIR then reports the portable marker as
unknown, typically as ``Unknown attribute 'this_block'``, because its compiler
hooks were not registered.

Keep the alias on the qualified activation import. A bare
``import cuda.coop.numba_mlir`` binds the name ``cuda`` in the importing
scope. If that name already refers to the object imported by
``from numba_cuda_mlir import cuda``, the bare import replaces it and later
``@cuda.jit`` uses the wrong module.

Alternatively, import :mod:`cuda.coop.numba_mlir` as ``coop`` to use the
qualified namespace. Its shared operations use the same signatures, selector
strings, and inference rules as the portable namespace; it adds only backend
memory namespaces and payload-alignment controls in this release.

Configuration
-------------

Runtime environment variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION``
   A truthy value disables automatic backend activation during
   :mod:`cuda.coop` import. Explicit qualified-backend import still works.

``CUDA_COOP_CCCL_ROOT``
   Selects a CCCL source checkout or a ``cuda-coop`` header bundle. An invalid
   configured root is an error; resolution does not fall back to another CCCL
   source.

``CUDA_COOP_ENABLE_CACHE``
   A truthy value enables the persistent compiler cache under
   ``~/.cache/cccl``. The value is read when the backend cache module is
   imported.

``CUDA_COOP_NUMBA_MLIR_NVRTC_DUMP_DIR``
   Writes content-addressed pre-NVRTC CUDA source files to this directory for
   compiler diagnostics.

``CUDA_PATH``
   Supplies ``<value>/include`` as a CUDA header candidate if
   ``cuda-pathfinder`` does not resolve one.

``CUDA_HOME``
   Supplies ``<value>/include`` after ``CUDA_PATH`` under the same fallback
   rule.

``CUDA_ROOT``
   Supplies ``<value>/include`` after ``CUDA_HOME`` under the same fallback
   rule.

If those mechanisms do not resolve CUDA headers,
``/usr/local/cuda/include`` is tried last.

For the two Boolean switches, values are case-insensitive; ``0``, ``false``,
``no``, ``off``, and the empty string are false.

Build-time CMake variables
^^^^^^^^^^^^^^^^^^^^^^^^^^

``CUDA_COOP_INSTALL_HEADER_BUNDLE``
   Defaults to ``ON``. Installs the private CCCL header and CMake-package
   bundle into the wheel.

``CUDA_COOP_ALLOW_DIRTY_HEADER_BUNDLE``
   Defaults to ``OFF``. Allows a Git-worktree bundle when selected inputs are
   changed or ``git status`` cannot verify them, and records its source
   revision as ``unknown``.

``CUDA_COOP_CCCL_SOURCE_REVISION``
   Defaults to empty. Supplies the revision token recorded instead of deriving
   it from Git. A dirty or unverifiable Git worktree still records ``unknown``.

Kernel API
----------

The portable root and qualified backend expose matching entry points:

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
   loaded = coop.load(
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
       loaded,
       valid_items=valid_items,
       offset=tile_offset,
   )

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
consecutive logical warps of 1, 2, 4, 8, 16, or 32 threads. Load, Store,
Exchange, and Scan support block, physical-Warp, and logical-Warp forms;
Shuffle is block-only. The enclosing block must contain a multiple of 32
threads, with no incomplete final physical warp. For a multidimensional block,
threads are linearized in x-major order. Every member of a participating group
must reach its collective; complete sibling logical groups may take different
control-flow paths.

The portable group vocabulary also includes thread, cluster, grid, and mapped
groups of physical warps. Full built-in Reduce uses the thread, cluster, and
mapped forms; data movement and Scan do not. Grid collectives remain
unsupported. ``ThreadGroup`` objects are descriptor-only in this release.
``group_by`` is compile-time vocabulary for describing a static partition.
Runtime query, membership, and synchronization methods such as
``rank``, ``count``, ``rank_as``, ``count_as``, ``sync``, ``sync_aligned``, and
``is_member`` are not exposed.

``ThreadData(items_per_thread, dtype=None)`` describes the fixed-size register
payload owned by each participating thread. Portable and qualified calls use
the same inference rules: an untyped Load output infers its dtype from the
source, and Store combines the destination dtype with payload writes. Load
returns the identical output object supplied by the caller; it does not
allocate or substitute another container.

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
   ) -> output

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
within that group. With Load, invalid output slots remain unchanged unless
``oob_default`` is supplied; a runtime default must also be uniform within the
group. A default is valid only when ``valid_items`` is present. With Store,
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

Both portable and qualified entry points use the same string algorithm
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
transactions while exposing a blocked payload. Portable and qualified calls
use the same lowercase string selectors. Selectors are normalized to lowercase
underscore-delimited strings. Enum and integer selectors, including ``0``, are
rejected.

Store consumes the arrangement associated with its selected algorithm. The
transpose Store implementations copy the payload before calling CUB, so Store
never modifies the caller's scalar or ``ThreadData`` value while CUB performs
its in-place reordering.

Exchange semantics
------------------

The portable signature is:

.. code-block:: python

   exchange(group, value, /, *, mode="striped_to_blocked") -> ThreadData

``value`` must be a fixed-size ``ThreadData`` payload. The result is a fresh
payload with the same dtype and extent; Exchange does not modify ``value``.
Block, physical Warp, and logical Warp groups support
``striped_to_blocked`` and ``blocked_to_striped``. In blocked order, thread
``t`` owns consecutive tile indices beginning at
``t * items_per_thread``. In striped order, its item ``i`` has tile index
``t + i * group_size``.

The qualified :func:`cuda.coop.numba_mlir.exchange` entry point also accepts
local arrays. Block groups additionally support warp-striped conversions,
scatter-to-blocked, scatter-to-striped, guarded scatter, flagged scatter, and
warp time slicing. Physical and logical Warp groups retain the two portable
layout modes. Scatter ``ranks`` are relative to the block tile, must have a
signed integer dtype, and must have the same extent as ``value``.
``valid_flags`` are required only by flagged scatter, must have a non-boolean
integer dtype, and must have that same extent.

For unguarded block scatter, every rank must be in
``[0, group_size * items_per_thread)``. Guarded scatter skips negative ranks,
but every nonnegative rank must still be in range. Flagged scatter uses only
ranks whose corresponding flag is nonzero; those active ranks must be in
range. These runtime bounds and unique active destinations are caller
preconditions. Holes and duplicate destinations produce unspecified result
slots. ``warp_time_slicing=True`` reduces BlockExchange storage and is not
available for Warp groups or guarded and flagged scatter modes.

Shuffle semantics
-----------------

Shuffle is block-only. The portable signature is:

.. code-block:: python

   shuffle(group, value, /, *, mode="down", distance=1) -> ThreadData

The portable API accepts only ``ThreadData``, ``up`` or ``down``, and the
fixed distance ``1``. The flattened blocked tile moves by one item. The first
``up`` result or last ``down`` result is unspecified; all other slots come
from the adjacent tile position. The returned payload is fresh and ``value``
is unchanged.

The qualified :func:`cuda.coop.numba_mlir.shuffle` entry point also accepts
scalar values with ``offset`` or ``rotate`` mode. Offset distance is signed,
may be negative, and may vary by thread, but it must fit a signed 32-bit
integer. A static overflow is rejected during compilation; a runtime overflow
executes a device trap before CUB's parameter is narrowed. Within that range,
a source rank outside the block leaves that thread's result unspecified.
Rotate distance may be static or runtime and must satisfy
``0 < distance < block_threads``. An invalid runtime Rotate distance also
executes a device trap. A trap invalidates that CUDA context, so validate
untrusted distances before launching a kernel. Array values remain limited to
unit ``up`` and ``down``; boundary-output projections are not part of this
release.

Scan semantics
--------------

The five portable spellings are ``scan``, ``exclusive_scan``,
``inclusive_scan``, ``exclusive_sum``, and ``inclusive_sum``. ``scan`` chooses
its form with ``mode="exclusive"`` or ``mode="inclusive"``. Every spelling
returns a fresh value with the same scalar or per-thread-array shape and dtype
as its input; the input remains unchanged.

Block Scan accepts a scalar or fixed-size ``ThreadData`` payload and supports
the lowercase ``raking``, ``raking_memoize``, and ``warp_scans`` algorithm
strings. The qualified :mod:`cuda.coop.numba_mlir` spelling also accepts fixed
local arrays. Physical and logical Warp Scan accept one scalar per lane and
have no algorithm or explicit-storage selector.

Sum is the default operation. The three general Scan spellings accept the same
built-in string aliases as Reduce. The qualified spelling also recognizes the
corresponding Python ``operator`` functions and NumPy ufuncs, and accepts a
stateless device callback. Non-sum exclusive Scan requires an
``initial_value`` matching the payload dtype; ordinary Python literals are
checked and converted in that context. Inclusive Scan rejects an initial
value.

The qualified spelling adds ``aggregate_output``, an exact-dtype one-item
``ThreadData`` or local array populated with the group aggregate. That
aggregate excludes an exclusive initial value. Warp forms also accept
``valid_items`` to scan the first N lanes by group rank, with
``1 <= N <= warp_width``; only those N result lanes are defined. The initial
value and ``valid_items`` must be uniform across all participating members.
Invalid runtime values execute a device trap before CUB's integer argument is
formed and invalidate the current CUDA context. Block Scan rejects
``valid_items``. These two controls are intentionally absent from the portable
root API.

All Scan forms use CUB temporary storage. Block calls may use compiler-owned,
caller-owned, or dynamic shared storage and append ``syncthreads`` unless a
caller-owned descriptor explicitly sets ``auto_sync=False``. Physical and
logical Warp calls use one compiler-owned slice per Warp and append
``syncwarp`` with the participating mask. Prefix callbacks and running-prefix
state are deferred.

.. literalinclude:: ../../python/cuda_coop/examples/numba_mlir/block_scan.py
   :language: python
   :start-after: docs: start numba-block-scan
   :end-before: docs: end numba-block-scan

Temporary storage
-----------------

Block Load and Store accept an optional ``TempStorage`` descriptor:

.. code-block:: python

   scratch = coop.TempStorage(
       size_in_bytes=None,
       alignment=None,
       auto_sync=None,
       sharing="shared",
   )

For block, physical Warp, and logical Warp operations, ``direct``, ``striped``,
and ``vectorize`` are storage-free. They default-construct the CUB primitive,
report zero temporary bytes, and emit no shared-memory allocation, storage
pointer, or synchronization barrier. For a block call, an explicit descriptor,
including an unsized descriptor, is validated as compile-time vocabulary but
does not change code generation for those algorithms.
Construct ``TempStorage`` inside the kernel; the current Numba-CUDA-MLIR
frontend does not resolve module-global storage descriptors.

The block ``transpose``, ``warp_transpose``, and
``warp_transpose_timesliced`` algorithms use CUB temporary storage. Without a
descriptor, the compiler allocates the specialization's exact storage and
inserts a block reuse barrier. An explicit descriptor makes that storage
caller-owned: it may select shared or exclusive ownership, request capacity and
alignment, or opt into dynamic shared memory. Shared storage may request
automatic reuse synchronization; exclusive storage must be synchronized by the
caller when it is reused. The generated provider remains authoritative for the
required byte count and alignment, and the backend validates the descriptor
against the concrete lowering plan.

Warp ``transpose``, Warp Exchange, and Warp Scan use compiler-owned storage
with one disjoint slice per physical or logical group. The compiler inserts
``syncwarp`` with the exact logical-group mask. Exchange and Shuffle always use
compiler-owned storage and append a group-scoped reuse barrier. Block Scan may
instead use implicit, caller-owned, or dynamic storage. Both the portable and
qualified APIs reject explicit ``TempStorage`` for every Warp Load and Store
algorithm, including the storage-free modes, and for Warp Scan.

Compilation and headers
-----------------------

``cuda-coop`` compiles providers against its configured CCCL root, the active
source checkout during in-tree development, or its installed header bundle, in
that order. It never substitutes the CUDA Toolkit's copy of CUB. CUDA headers,
NVRTC, ``nvrtc-builtins``, and nvJitLink must resolve to a compatible toolkit
root. The resulting compiler artifacts and caches include the launch
dimensions, dtype and item extent, storage ABI, compute capability, compiler
options, ordered header identity, and toolkit-library identity.

See :doc:`coop_api` for the public API reference.
