.. _cudax-sharded:

Sharded containers and algorithms
==================================

.. contents::
   :depth: 2

Sharded containers partition one logical array across the places of a single
process — devices, or sub-device locality domains — while keeping a common
address space. They extend the cooperation-scope structure CUDA algorithms
already follow: a primitive at one scope runs the previous scope's primitive
locally and combines results using what the new scope shares (registers and
shuffles within a warp, shared memory within a block, global memory within a
device). At the places scope, what is shared is one virtual address space with
placed pages; at the multi-process/multi-node scope, where nothing is shared,
communicator-based algorithms take over
(see the MGMN algorithms built on ``__multi_gpu``).

The sharded API lives in the ``cuda::experimental::sharded`` namespace and is
available through the ``cuda/experimental/sharded.cuh`` header. It builds on
the standalone :ref:`places <cudax-places>` layer; execution resources come
from a :ref:`place_group <places-place-group>`.

sharded_array
-------------

``sharded_array<T>`` is a 1D array whose shards each carry their own
placement: a ``data_place``, an ``exec_place`` and a reference stream.

.. code-block:: cpp

   using namespace cuda::experimental::sharded;

   auto group     = place_group::by_locality_domains(); // re-exported from places
   const size_t n = 1u << 28;
   auto data      = sharded_array<double>::allocate(group, n);

   iota(group, data, 0.0);
   double total = sum(group, data);   // per-place CUB + combine

Factory naming follows a two-word rule: ``adopt`` = zero-copy view over
caller-owned memory (the container becomes a view and the caller owes the
memory's lifetime); ``from_*`` = builds owned storage by copying or
transforming its input. ``sharded_array<T>::adopt(shards)`` is the named
form of the adopting constructor.

``allocate_contiguous`` places the shards inside *one* contiguous virtual
address range (VMM-backed via ``localized_array``): logical shard boundaries
are exact, physical ownership snaps to the allocation granularity, and
``contiguous_data()`` hands the whole array to unmodified single-pointer
consumers. Because the range is mapped once, shard sizes are fixed;
size-mutating operations must refuse such arrays.

Composing with a caller stream: ``fork_from`` / ``join_into``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sharded work runs on per-shard streams; callers usually have *one* stream
carrying the surrounding computation. ``fork_from(stream)`` declares that the
shard streams depend on the work currently enqueued on the caller stream (one
event recorded on it, every shard stream waits); ``join_into(stream)`` is the
mirror (one event per shard stream, the caller stream waits on all). Both are
ordering declarations, not synchronizations — the host returns immediately:

.. code-block:: cpp

   producer<<<grid, block, 0, s>>>(...); // writes data's memory on stream s
   data.fork_from(s);                    // shards now depend on the producer
   transform(group, data, out, op);      // per-shard work on the shard streams
   out.join_into(s);                     // s now depends on every shard
   consumer<<<grid, block, 0, s>>>(...); // sees all results; no host sync

The events come from a small pool owned by the container (created lazily,
reused across calls), so adopted arrays over foreign streams are supported
identically. Both members are capture-safe: inside an active CUDA graph
capture the record/wait pairs become graph dependencies, making
``fork_from``/``join_into`` the composition idiom between a captured caller
stream (or graph) and the per-shard work.

Algorithms
----------

The algorithm family:

- elementwise: ``fill``, ``sequence``, ``iota``, ``tabulate``, ``generate``,
  ``for_each``, ``transform`` (in-place, unary, binary) — no cross-place
  stage;
- ``reduce`` / ``sum`` / ``min`` / ``max``: per-place CUB ``DeviceReduce``
  plus a combine of the per-place partials;
- ``inclusive_scan`` / ``exclusive_scan``: per-place CUB ``DeviceScan``, then
  per-place prefixes folded back in place;
- ``adjacent_difference``: local differences plus one boundary element per
  shard;
- ``count`` / ``count_if``: per-place CUB transform-reduce plus a sum of the
  per-place counts;
- ``histogram_even``: per-place CUB ``DeviceHistogram`` plus a per-bin sum of
  the per-place histograms;
- ``copy_if`` / ``filter`` / ``remove_if``: per-place CUB ``DeviceSelect``
  compaction in place, then shard sizes and offsets are updated;
- ``unique``: per-place CUB ``DeviceSelect::Unique`` in place, then
  duplicates straddling shard boundaries are trimmed with an O(1) size
  decrement per boundary.

Algorithm temporaries are drawn from each shard's place through the group's
per-place memory resources.

Size-mutating algorithms and the contiguous backing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``copy_if`` / ``filter`` / ``remove_if`` and ``unique`` shrink shard sizes in
place (capacities are unchanged; ``reset_sizes_to_capacity()`` reuses the
buffers). On a contiguous array this is unrepresentable: shrinking a shard
would leave a gap between its valid elements and the next shard's, falsifying
the read-as-one-array contract of ``contiguous_data()``, while compacting
across the gap would migrate elements onto other places than the caller asked
for. These algorithms therefore throw ``std::invalid_argument`` on contiguous
(``allocate_contiguous``) arrays, leaving them untouched. Read-only
algorithms (``count`` / ``count_if``, ``histogram_even``, ``reduce`` et al.)
remain available on every sharded array, contiguous ones included.
CUDA graph capture
------------------

The sharded surface splits cleanly along the capture boundary: elementwise
computation records into a CUDA graph; everything that allocates, transfers
host data or synchronizes refuses cleanly instead of corrupting the capture.

What captures
~~~~~~~~~~~~~

The elementwise algorithms (``fill``, ``sequence``, ``iota``, ``tabulate``,
``generate``, ``for_each``, ``transform``) called with ``blocking = false``
are pure per-shard kernel launches on the shards' reference streams, so they
capture with the containers' fork/join members — the documented way to
compose sharded work with a caller stream or graph: begin capture on an
origin stream, ``fork_from(origin)`` so every shard stream depends on it,
record the pipeline, and ``join_into(origin)`` before ending the capture
(the record/wait pairs become graph dependencies):

.. code-block:: cpp

   cudaStreamBeginCapture(origin, cudaStreamCaptureModeGlobal);
   data.fork_from(origin);                          // fork

   transform(group, data, out, op, /*blocking=*/false);
   for_each(group, out, update, /*blocking=*/false);

   out.join_into(origin);                           // join
   cudaStreamEndCapture(origin, &graph);

The captured graph is placement-faithful: each shard's kernels are recorded
from that place's stream, and the per-place SM confinement of those streams
survives instantiation and replay (pinned by an SM-id check in the test
suite). Replays recompute from the *current* contents of the shards, so inputs
may be rewritten — outside the graph — between launches. Contiguous
(``allocate_contiguous``) arrays capture transparently: the VMM mappings
pre-exist the capture, and per-shard stages compose with whole-array kernels
through ``contiguous_data()`` inside one graph.

What must stay outside — and how it fails
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Everything else on the surface performs host-side work a graph cannot
represent. Those operations *refuse* under an active capture by throwing
``std::runtime_error`` before touching the stream; the check is a safe
query, so the ongoing capture remains VALID and keeps accepting supported
work. The refusing set:

- container allocation: ``allocate`` (all overloads) and
  ``allocate_contiguous``;
- host transfers: ``copy_from_host``, ``copy_to_host``, ``copy_between``;
- synchronization: ``sharded_array::sync`` and ``place_group::sync`` —
  and therefore the elementwise algorithms when called
  with ``blocking = true``, which throw at their final sync (their kernels
  are already recorded; the capture is still valid and can be completed or
  abandoned);
- the synchronous algorithms, all of which stage per-place partials through
  the host: ``reduce`` / ``sum`` / ``min`` / ``max``, ``inclusive_scan`` /
  ``exclusive_scan``, ``count`` / ``count_if``, ``histogram_even``,
  ``copy_if`` / ``filter`` / ``remove_if``, ``unique``,
  ``adjacent_difference``.

The guards also refuse when a global-mode capture is active anywhere in the
process, since the underlying synchronization would invalidate it under the
CUDA capture rules. Shard adoption and ``slice`` are host-only bookkeeping
and remain usable during capture; ``place_group`` construction and lazy
stream materialization record nothing into a graph. Construct groups and
containers before capturing, and destroy owning containers outside capture
(freeing is stream-ordered and is not guarded).

Graph-owned memory
~~~~~~~~~~~~~~~~~~

``place_memory_resource`` stream-ordered allocation is itself capturable: an
``allocate`` on a capturing stream records a graph memory node drawing from
the place's (locality-domain) pool, so placed temporaries can live inside a
graph when the allocate/deallocate pair is enclosed in the capture. An
allocation *not* freed inside the graph stays live after a launch — the
captured pointer is readable — and relaunching before freeing fails; the
pointer can be released outside the graph with ``cudaFreeAsync``, which
re-arms the launch. The shipped contract stays simple — allocate outside
capture; capture computation only — and the test suite pins both the
recorded-allocation semantics and the failure shape of violations.

sharded_csr and the sparse products: a closed library as the engine
-------------------------------------------------------------------

``sharded_csr<T>`` is a row-partitioned CSR sparse matrix: one shard per
place of a ``place_group``, each shard a self-contained CSR operator for a
contiguous row range (nnz slice plus offsets rebased to zero), stored in its
place's memory. Because every shard is a complete CSR matrix, a CLOSED
library that only understands pointers and a stream can consume it with one
ordinary call per shard — the container carries the placement, the library
never changes. The container is vendor-free and ships in the umbrella header.
``sharded_csr::from_device`` ingests a CSR whose arrays already live on the
device; per the ``from_*`` naming rule it builds owned storage — offsets are
rebased into container-owned shards and colinds/values are copied
device-to-device into the shards' places, so nothing aliases the caller's
arrays and they may be freed once it returns.

The cuSPARSE-backed products live in the separate opt-in header
``<cuda/experimental/sharded_sparse.cuh>``, which requires the cuSPARSE
development headers (it ``#error``\ s otherwise, like
``<cuda/experimental/cufile.cuh>``) and linking against cuSPARSE:

.. code-block:: cpp

   #include <cuda/experimental/sharded_sparse.cuh>

   auto group = place_group::by_locality_domains();
   sharded_csr<double> A(group, rows, cols, h_offsets, h_colinds, h_values);
   auto y = A.make_row_partitioned();          // disjoint row blocks, no combine
   spmv(group, A, d_x, y, alpha, beta);        // one confined call per shard
   auto C = A.make_row_partitioned(n_cols, /* contiguous */ true);
   spmm(group, A, d_B, C, n_cols);             // C readable as ONE array

Each call runs one cuSPARSE call per shard on the shard's place stream
(``cusparseSetStream``). Per-(shard, operation) library state — handle,
descriptors, workspace, preprocessed plan — is created lazily on the first
call into the container's type-erased ``lib_state()`` slots, built once
against the shard's fixed addresses, and reused for the matrix's lifetime;
subsequent calls only rebind the dense pointers when they change. The row
partition makes the output row blocks disjoint, so there is never a combine
step, and outputs compose with ``allocate_contiguous`` backings unchanged.

Dense operands (``x``, ``B``) are plain device pointers readable from every
place. Which per-place COPIES of a re-read operand should exist — and when a
write makes them stale — is a coherence question that belongs to the binding
tier: an STF ``logical_data`` can materialize and cache a per-place instance
and hand its pointer to these calls; the container deliberately does not
absorb that role.

Measured rebalance
~~~~~~~~~~~~~~~~~~

An nnz-balanced row split (the default) is not a TIME-balanced split: with
each shard confined to its place's SMs, a call finishes at max(shard time),
and skewed row-length distributions make the default split pay the full
skew. ``spmv_shard_times`` / ``spmm_shard_times`` measure each shard solo
through the exact call path of the products, and
``sharded_csr::time_balanced_boundaries`` converts one measurement into a
time-equalizing split via a piecewise-rate model:

.. code-block:: cpp

   auto times = spmm_shard_times(group, A, d_B, C, n_cols);
   auto bounds = sharded_csr<double>::time_balanced_boundaries(
     rows, h_offsets, A.interior_boundaries(), times);
   sharded_csr<double> A2(group, rows, cols, h_offsets, h_colinds, h_values, bounds);

One calibration round is amortized over every subsequent call on the rebuilt
matrix — the natural fit for iterative consumers that reuse one operator
across many products. Rates shift as rows change shards, so repeat the round
(keeping the best measured split) when the skew is extreme.
