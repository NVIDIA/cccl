.. _cccl-python-coop:

``cuda.coop``: Cooperative Load and Store
==========================================

``cuda.coop`` provides cooperative CUDA primitives for Python kernel DSLs.
The initial release integrates with Numba-CUDA-MLIR and supports Load and Store
for blocks, complete physical warps, and power-of-two logical warps. Its
portable descriptors and planning records are designed so that later groups
and primitive families can be added without changing the public dispatch
model.

Installation
------------

The base install has no Python package dependencies:

.. code-block:: console

   python -m pip install cuda-coop

For Numba-CUDA-MLIR, install the extra matching your CUDA major version:

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
``numba-cuda-mlir>=0.5.0,<0.6``. Its private compiler API module
provides access to overload templates, IR, datamodels, and the registries
needed to roll back a failed activation. It does not adapt between runtime
versions. Other runtime series are rejected before compiler registries change.

Backend registration
--------------------

Call ``register`` on the host before compiling a kernel:

.. code-block:: python

   from cuda import coop

   coop.register("numba-cuda-mlir")

   from numba_cuda_mlir import cuda

Registration works in either import order and is safe to repeat. It also
accepts ``"numba_cuda_mlir"``. The backend dependencies must already be
installed. A standalone ``cuda.coop`` import does not load an optional
compiler; importing it after Numba-CUDA-MLIR activates the backend
automatically unless ``CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION`` is set.

Importing the backend namespace also registers it:

.. code-block:: python

   import cuda.coop.numba_mlir as numba_coop

You can use ``as coop`` when only using this backend. Keep the alias so the
import does not replace a ``cuda`` name imported from Numba-CUDA-MLIR.
The common API will also support future backends; CUTLASS support is planned.

Configuration
-------------

Runtime environment variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION``
   A truthy value disables automatic backend activation during
   :mod:`cuda.coop` import. Explicit registration and qualified-backend import still work.

``CUDA_COOP_CCCL_ROOT``
   Selects a CCCL source checkout or a ``cuda-coop`` header bundle. An invalid
   configured root is an error; resolution does not fall back to another CCCL
   source.

``CUDA_COOP_ENABLE_CACHE``
   A truthy value enables the persistent compiler cache. The value is read
   when the backend cache module is imported.

``XDG_CACHE_HOME``
   On Linux and other POSIX systems, sets the cache base directory; entries
   are stored in ``<value>/cccl``. Unset, empty, or relative values fall back
   to ``~/.cache/cccl``. Read when the backend cache module is imported.

``LOCALAPPDATA``
   On Windows, sets the cache base directory; entries are stored in
   ``<value>\cccl``. Unset, empty, or relative values fall back to
   ``~\AppData\Local\cccl``. Read when the backend cache module is imported.

``CUDA_COOP_SOURCE_DUMP_DIR``
   Writes generated CUDA source to this directory for compiler diagnostics.
   Files use ``cuda_coop_<backend>_<hash>.cu`` names so different backends can
   share a directory. Set it before compiling; the Numba backend also writes
   the source when its provider compilation cache is hit. Unset or empty
   disables dumping.

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

The portable group vocabulary also includes thread, cluster, grid, and mapped
groups of physical warps, but those are not Load or Store targets.
``ThreadGroup`` objects are descriptor-only in this release. ``group_by`` is
compile-time vocabulary for describing a static partition. Runtime query,
membership, and synchronization methods such as
``rank``, ``count``, ``rank_as``, ``count_as``, ``sync``, ``sync_aligned``, and
``is_member`` are not exposed.

``ThreadData(items_per_thread, dtype=None, *, alignment=None)`` describes the
fixed-size register payload owned by each participating thread. Portable and
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
defaults to ``True`` for both.

The synchronization model is deliberately simple. A descriptor names one
region; distinct descriptors and compiler-owned storage never alias each
other. With ``auto_sync`` enabled, which is the default, the compiler appends
``cuda.syncthreads()`` for block groups or ``cuda.syncwarp(mask)`` for Warp
groups immediately after every call that consumes the storage, including the
last one, and never inserts a barrier before a call. That trailing barrier
exists only to order reuse of the temporary storage; it is not a general
barrier for the kernel's own shared-memory traffic and disappears when
``auto_sync=False``. With ``auto_sync=False`` the caller issues
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
logical-group mask. Both the portable and qualified APIs reject explicit
``TempStorage`` for every Warp Load and Store algorithm, including the
storage-free modes.

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
