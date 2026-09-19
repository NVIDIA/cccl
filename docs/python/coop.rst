.. _cccl-python-coop:

``cuda.coop``: Cooperative Group Primitives
============================================

.. toctree::
   :hidden:
   :maxdepth: 2

   Overview <self>
   coop/programming_guide
   coop/neighbor-operations
   coop/developer_overview
   coop/visualizations/index
   coop/glossary
   coop/faqs

``cuda.coop`` provides cooperative CUDA primitives for Python kernel DSLs.
Threads work together to :doc:`load <coop/visualizations/load>` and
:doc:`store <coop/visualizations/store>` tiles, rearrange values with
:doc:`Exchange <coop/visualizations/exchange>` and
:doc:`Shuffle <coop/visualizations/shuffle>`, or compute
:doc:`reductions <coop/visualizations/reduce>` and
:doc:`scans <coop/visualizations/scan>` inside a kernel. They can also
:doc:`sort keys and associated values <coop/visualizations/merge-sort>` within a group or
compute :doc:`radix sorts and digit ranks <coop/visualizations/radix>` within a block.
:doc:`TopK <coop/visualizations/topk>` selects a block's smallest or largest keys without
sorting the full tile. Blocks can compare neighboring items with
:doc:`Adjacent Difference <coop/visualizations/adjacent-difference>` and
:doc:`Discontinuity <coop/visualizations/discontinuity>`, count samples with
:doc:`Histogram <coop/visualizations/histogram>`, or expand compressed runs
with :doc:`Run Length Decode <coop/visualizations/run-length-decode>`.
:doc:`Batched Warp Reduction <coop/visualizations/reduce-batched>` computes
an independent reduction for each per-thread payload slot.

The common ``cuda.coop`` API describes those operations independently of a
kernel compiler. Numba-CUDA-MLIR is the first supported backend; CUTLASS
support is planned. The backend namespace adds features specific to its
compiler. See :ref:`Which namespace should I use? <coop-faq-namespaces>`.

Start with the :doc:`Programming Guide <coop/programming_guide>` to write a
kernel. The :doc:`Visualizations <coop/visualizations/index>` show where each
value goes, and the :doc:`Glossary <coop/glossary>` explains terms such as
:term:`blocked` and :term:`striped`. The :doc:`FAQs <coop/faqs>` cover API
choices and common questions. For compiler integration details, see the
:doc:`Developer Overview <coop/developer_overview>`.

Installation
------------

Install ``cuda-coop`` without adding Python package dependencies:

.. code-block:: console

   python -m pip install cuda-coop

The wheel includes the common API, every shipped DSL integration (including
``cuda.coop.numba_mlir``), type declarations, and a matching bundle of CUB,
Thrust, libcu++, and CUDAX headers. The base install declares no Python
package dependencies. You can import ``cuda.coop`` without a compiler or GPU;
using an integration requires its backend dependencies to be installed.

For Numba-CUDA-MLIR, install the extra matching your CUDA major version:

.. code-block:: console

   python -m pip install "cuda-coop[numba-cuda-mlir-cu13]"
   # Use numba-cuda-mlir-cu12 with CUDA 12.

Both commands install the same ``cuda-coop`` wheel with the same DSL
integrations. The extra only adds the dependency requirements declared in
``pyproject.toml`` so pip installs the supported Numba-CUDA-MLIR stack for
CUDA 13. The current integration requires ``numba-cuda-mlir>=0.5.0,<0.6``.
Installing an extra does not register a backend in a running Python process;
see :ref:`installation versus registration <coop-faq-installed-extra>`.

Installed-wheel compilation uses the bundled CCCL headers. Development from a
CCCL source checkout uses its matching headers. ``CUDA_COOP_CCCL_ROOT`` can
select another source checkout or ``cuda-coop`` header bundle.

.. _coop-backend-registration:

Registering a backend
--------------------

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

Configuration
-------------

Runtime environment variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the Boolean switches below, *truthy* means any value other than the
empty string, ``0``, ``false``, ``no``, or ``off``. For example, ``1``,
``true``, ``yes``, and ``on`` are all truthy. Values are case-insensitive,
and leading and trailing whitespace is ignored. An unset variable is false.

``CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION``
   A truthy value disables automatic backend activation during
   :mod:`cuda.coop` import. Explicit ``coop.register(...)`` and
   backend imports still work.

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

On Linux and other POSIX systems, ``/usr/local/cuda/include`` is tried last.
Windows uses ``cuda-pathfinder`` or the configured toolkit roots above; it
does not try the Unix fallback. If no valid CUDA include directory is found,
compilation reports a header-resolution error.

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

The common and qualified APIs expose matching entry points:

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
consecutive logical warps of 1, 2, 4, 8, 16, or 32 threads. Load, Store,
Exchange, Scan, and Merge Sort support block, physical-Warp, and logical-Warp
forms. Shuffle, Radix Sort, Radix Rank, TopK, Adjacent Difference,
Discontinuity, Histogram, and Run Length Decode are block-only. Batched
Reduction supports physical and logical warps. For Warp primitives, the enclosing block must contain a multiple of 32 threads, with
no incomplete final physical warp. For a multidimensional block, threads are
linearized in x-major order. Every member of a participating group must reach
its primitive; complete sibling logical groups may take different control-flow
paths.

The common group vocabulary also includes thread, cluster, grid, and mapped
groups of physical warps. Full built-in Reduce uses the thread, cluster, and
mapped forms; data movement and Scan do not. Grid primitives remain
unsupported. ``ThreadGroup`` exposes the C++ hierarchy query surface.
``rank(level="thread")`` and ``count(level="thread")`` accept ``thread`` (or
``gpu_thread``), ``warp``, ``block``, ``cluster``, and ``grid``. Their default
result is the unsigned product type used by the corresponding C++ hierarchy
operation: normally ``uint32``, and ``uint64`` when the group or queried outer
level is the grid. ``rank_as(dtype, level="thread")`` and
``count_as(dtype, level="thread")`` select an explicit signed or unsigned 8-,
16-, 32-, or 64-bit integer dtype. ``is_member()`` returns an integer
membership flag.

``sync()`` and ``sync_aligned()`` expose the matching non-grid barriers. Every
participating member must reach ``sync()``. ``sync_aligned()`` additionally
requires an aligned and converged group. Grid synchronization is unavailable
because the backend cannot request a cooperative grid launch.

``group_by`` remains static compiler vocabulary: ``count`` and ``exhaustive``
must be compile-time constants. A mapped threads-within-warp group can query
its threads and immediate parent Warp. A mapped warps-within-block group can
query its threads, physical Warps, and immediate parent block. Queries above
the immediate physical parent are rejected. Mapped warps-within-block groups
support queries and ``is_member()`` but not ``sync()`` or ``sync_aligned()``;
their block-barrier lifetime requires a future planner-owned contract. For a
non-exhaustive partition, use ``is_member()`` to guard rank-dependent work for
excluded threads. Do not use that branch to skip a primitive unless the
primitive's participation contract explicitly permits it; every required
group or parent-group participant must still reach the primitive.

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

Both common and qualified entry points use the same string algorithm
vocabulary: ``direct``, ``striped``,
``vectorize``, ``transpose``, ``warp_transpose``, and
``warp_transpose_timesliced``. All six algorithms are executable with the
Numba-CUDA-MLIR backend. ``direct`` and ``vectorize`` use blocked ordering, so
each thread owns a contiguous segment of the tile. ``striped`` exposes striped
ordering, where item ``i`` for a thread is separated from its next item by the
block size. The three transpose algorithms use striped memory transactions but
present :term:`blocked` ``ThreadData`` to the caller. The two warp-transpose
variants perform that reordering within each warp and require a block size
divisible by 32.

Physical and logical Warp Load and Store support ``direct``, ``striped``,
``vectorize``, and ``transpose``. Their layouts follow the same rules at the
selected group width: ``direct`` and ``vectorize`` expose blocked payloads,
``striped`` exposes a striped payload, and ``transpose`` uses striped memory
transactions while exposing a blocked payload. Common and qualified calls
use the same lowercase string selectors. Selectors are normalized to lowercase
underscore-delimited strings. Enum and integer selectors, including ``0``, are
rejected.

Store consumes the arrangement associated with its selected algorithm. The
transpose Store implementations copy the payload before calling CUB, so Store
never modifies the caller's scalar or ``ThreadData`` value while CUB performs
its in-place reordering.

Exchange semantics
------------------

The common signature is:

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
warp time slicing. Physical and logical Warp groups retain the two common
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

Shuffle is block-only. The common signature is:

.. code-block:: python

   shuffle(group, value, /, *, mode="down", distance=1) -> ThreadData

The common API accepts only ``ThreadData``, ``up`` or ``down``, and the
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

The five common spellings are ``scan``, ``exclusive_scan``,
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
checked and converted in that context. A block-prefix callback can supply that
prefix instead. Inclusive Scan rejects an initial value.

The qualified spelling adds ``aggregate_output``, an exact-dtype one-item
``ThreadData`` or local array populated with the group aggregate. That
aggregate excludes an exclusive initial value. Warp forms also accept
``valid_items`` to scan the first N lanes by group rank, with
``1 <= N <= warp_width``; only those N result lanes are defined. The initial
value and ``valid_items`` must be uniform across all participating members.
Invalid runtime values execute a device trap before CUB's integer argument is
formed and invalidate the current CUDA context. Block Scan rejects
``valid_items``. These two controls are intentionally absent from the common
API.

Block prefix callbacks
^^^^^^^^^^^^^^^^^^^^^^

All five qualified Block Scan spellings accept a stateless or stateful prefix
callback through the ``prefix_op`` keyword. A stateless callback receives the
block aggregate and returns the prefix that precedes the current tile:

.. code-block:: python

   from numba_cuda_mlir import cuda, types

   import cuda.coop.numba_mlir as coop


   @cuda.jit(device=True)
   def prefix_after_aggregate(block_aggregate):
       return block_aggregate + 7


   # Inside a kernel:
   result = coop.exclusive_sum(
       coop.this_block(),
       value,
       prefix_op=prefix_after_aggregate,
   )

A stateful callback receives its state payload first and the block aggregate
second. Wrap it in ``StatefulFunction`` and pass the state as the third
positional argument:

.. code-block:: python

   @cuda.jit(device=True)
   def carry_prefix(state, block_aggregate):
       previous = state[0]
       state[0] = previous + block_aggregate
       return previous


   running_prefix = coop.StatefulFunction(carry_prefix, types.int64)

   # Inside a kernel, before a loop over tiles:
   state = coop.ThreadData(1, dtype=types.int64)
   state[0] = types.int64(0)
   result = coop.exclusive_sum(
       coop.this_block(),
       value,
       state,
       prefix_op=running_prefix,
   )

The state must be a numeric one-item ``ThreadData`` or local array. Its dtype
must exactly match ``StatefulFunction.dtype``, but may differ from the scanned
payload dtype. Keep the same state payload alive across repeated scans to
carry the prefix between tiles. Every participating thread must initialize
its state cell to the same contents before the first primitive.

CUB may invoke the prefix callback in every lane of the block's first warp,
but only lane 0's returned prefix is applied to the scan. Other per-thread
state copies are not authoritative; after one or more calls, consume the final
state only from thread 0. The callback is mutually exclusive with
``initial_value`` and ``aggregate_output``. It is not available for physical
or logical Warp Scan, through the common :mod:`cuda.coop` API, as a stateful
binary ``scan_op``, or with structured state.

All Scan forms use CUB temporary storage. Block calls may use compiler-owned,
caller-owned, or dynamic shared storage and append ``syncthreads`` unless a
caller-owned descriptor explicitly sets ``auto_sync=False``. Physical and
logical Warp calls use one compiler-owned slice per Warp and append
``syncwarp`` with the participating mask. Prefix callbacks do not change these
rules: when repeated calls reuse Block Scan storage, retain the automatic
barrier or issue ``syncthreads`` after each call when ``auto_sync=False``. The
prefix state is persistent per-thread data, not CUB temporary storage.

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
policy explicitly. Standalone primitive helpers and primitives inside
standalone callbacks are unsupported. For the MVP, ``literal_unroll``
values cannot determine cooperative payload extents, group dimensions,
selectors, or descriptor constructor arguments. Write separate calls with
explicit constants, or use an ordinary loop with one fixed cooperative shape.
An unrelated ``literal_unroll`` loop does not add this restriction.

Warp ``transpose``, Warp Exchange, and Warp Scan use compiler-owned storage
with one disjoint slice per physical or logical group. The compiler inserts
``syncwarp`` with the exact logical-group mask. Exchange and Shuffle always use
compiler-owned storage and append a group-scoped reuse barrier. Block Scan may
instead use implicit, caller-owned, or dynamic storage. Both the common and
qualified APIs reject explicit ``TempStorage`` for every Warp Load and Store
algorithm, including the storage-free modes, and for Warp Scan. Batched
Reduction also uses compiler-owned storage per warp and has no
``temp_storage`` argument. Adjacent Difference, Discontinuity, Histogram,
and both Run Length Decode forms accept explicit block scratch; prepared
RLD tables live only for the duration of each call.

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
