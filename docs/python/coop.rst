.. _cccl-python-coop:

Cooperative Algorithms (``cuda.coop``)
======================================

.. warning::
   ``cuda.coop`` is an experimental API and is subject to change.

``cuda.coop`` provides cooperative Load and Store operations for Python GPU
kernels. Its shared API describes the participating threads, per-thread data,
and algorithm. A compiler backend lowers those calls to CUDA device code.

Installation
------------

.. code-block:: bash

   python -m pip install cuda-coop

The base package contains the shared API, Block and Warp Load/Store planning,
and the CCCL headers needed by compiler integrations. Backend integrations are
added separately. The base package has no Python package dependencies. Importing the base package does not require a CUDA device.

.. _coop-backend-registration:

Backend registration
--------------------

``coop.register("numba-cuda-mlir")`` explicitly selects the Numba-CUDA-MLIR
integration; ``"numba_cuda_mlir"`` is an accepted spelling. Call it on the
host before compiling kernels. This core-only package does not include the
adapter yet, so requesting it raises an informative ``ImportError``.

.. _coop-thread-groups:

Groups and algorithms
---------------------

Use ``coop.this_block()`` for a thread block, ``coop.this_warp()`` for a
physical warp of 32 threads, and ``coop.this_warp().group_by(width)`` for a
logical warp of 1, 2, 4, 8, 16, or 32 threads.

The factories describe the current kernel launch and take no size arguments.
Constructing a group does not synchronize threads. A descriptor's availability
does not imply that an operation supports it: thread, cluster, grid, and mapped
groups of physical warps are not Load or Store targets. Runtime rank, size,
membership, and synchronization queries are not exposed in this API layer.

``group_by`` counts units in the next inner hierarchy level: threads for a
warp parent and physical warps for a block parent. Its ``count`` and keyword-only
``exhaustive`` arguments are compile-time constants. An exhaustive partition
must divide the parent exactly; ``exhaustive=False`` permits a remainder.
Nested partitions are unsupported. See :meth:`cuda.coop.ThreadGroup.group_by`
for a host-side descriptor example.

Block Load and Store accept ``direct``, ``striped``, ``vectorize``,
``transpose``, ``warp_transpose``, and ``warp_transpose_timesliced``. Physical
and logical Warp operations accept ``direct``, ``striped``, ``vectorize``, and
``transpose``. The planner selects the corresponding CUB specialization and
records storage and synchronization requirements for the compiler backend.

``load(group, source, output)`` fills a ``ThreadData`` object in place and
returns ``None``. ``store(group, destination, value)`` writes a per-thread
value or ``ThreadData`` payload. Both accept a nonnegative element ``offset``
and a ``valid_items`` count within the selected group tile. Load also accepts
``oob_default`` with ``valid_items`` to fill invalid output slots.

.. _coop-participation:

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

.. _coop-thread-data:

Per-thread payloads
-------------------

``ThreadData(items_per_thread, dtype=None, *, alignment=None)`` describes a
fixed-size payload owned by each thread. With 128 threads and two items per
thread, a group tile contains 256 values. Each thread reads and writes its own
slots with ``items[0]`` and ``items[1]``. The positive compile-time item count
is also available as ``items.items_per_thread``.

The compiler can infer an omitted dtype from a supported producer such as
Load. An explicit ``alignment`` is a minimum storage alignment in bytes and
must be a positive compile-time power of two. It does not assert alignment
of the arrays passed to Load or Store.

The contents are uninitialized. Write every item before reading it. A full
Load initializes the payload; a partial Load leaves invalid slots unspecified
unless ``oob_default`` is provided, even if initialized before Load. Construction does not synchronize threads.

:class:`cuda.coop.ThreadDataLike` describes this payload interface for typing.
Use :func:`cuda.coop.ThreadData` inside a supported kernel to construct one.
An arbitrary Python object implementing the protocol does not automatically
become a compiler-supported payload.

.. _coop-data-layouts:

Data layout
-----------

For four threads holding two items each, a blocked layout assigns consecutive
items to each thread: thread 0 holds ``[0, 1]``, thread 1 holds ``[2, 3]``, and
so on. A striped layout assigns consecutive items across threads: thread 0
holds ``[0, 4]``, thread 1 holds ``[1, 5]``, and so on.

The ``direct`` and ``vectorize`` algorithms produce blocked per-thread data.
``striped`` produces striped per-thread data. Transpose algorithms use shared
memory to combine coalesced accesses with blocked per-thread data.

For group size ``G``, items per thread ``K``, thread rank ``t``, and item index
``i``, blocked order uses tile position ``t * K + i``; striped order uses
``t + i * G``. Store interprets its payload according to its algorithm, so
pair Load and Store algorithms with matching layouts. The payload carries no
runtime layout tag that corrects a mismatched pair.

.. _coop-temp-storage:

Temporary storage
-----------------

``direct``, ``striped``, and ``vectorize`` require no shared temporary storage.
Transpose algorithms require storage and synchronization. The planner lets the
backend allocate storage automatically. Block operations can also accept a
``TempStorage`` object when a kernel needs to control its allocation or reuse.
Warp operations require backend-managed storage with a separate slice for
each participating group.

Scratch belongs to one block during its kernel execution. A descriptor does
not carry data between blocks or kernel launches.

``TempStorage(size_in_bytes=None, *, alignment=None, auto_sync=False,
sharing="shared")`` describes scratch for supported block calls. An omitted
capacity lets the compiler size the allocation from its uses. An explicit
capacity must be positive and large enough for those uses. ``alignment`` sets
a positive power-of-two minimum in bytes; the compiler may strengthen it.

``sharing="shared"`` allows call sites using the same descriptor to overlap
their scratch slices. ``sharing="exclusive"`` gives distinct call sites
separate slices. Repeated execution of a call site, including a loop, still
reuses its slice under either policy.

``TempStorage()`` defaults to ``auto_sync=False``; passing ``None`` also
disables automatic reuse barriers. The caller must synchronize before reuse.
Use ``TempStorage(auto_sync=True)`` to request a trailing barrier after each
scratch-using call. Omitting the ``temp_storage`` argument is different: the
compiler manages the allocation and its reuse barriers automatically.

:class:`cuda.coop.TempStorageLike` is the descriptor interface used in type
signatures. The scratch contents are opaque; keep application data in
``ThreadData`` or application-owned arrays.

The shared API functions are compiler markers; invoking an operation outside
a supported kernel compiler raises an error. Group descriptors can be created
and inspected from ordinary Python.

See :doc:`coop_api` for the shared API reference.
