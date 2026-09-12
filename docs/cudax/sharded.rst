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

   auto group     = place_group{make_locality_domain_grid()}; // both re-exported from places
   const size_t n = 1u << 28;
   auto data      = sharded_array<double>::allocate(group, n);

   iota(data, 0.0);
   double total = sum(data);          // per-shard CUB + combine

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
   zip_transform(out, op, data);         // per-shard work on the shard streams
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

Algorithms are written against the concepts (see below), not against the
container: the container algorithms are one *instantiation* of the concept
tier, materialized by ``place_group`` — ``sharded_array`` models
``sharded_view`` and ``self_bound``, so ``algo(a, ...)`` *is* the generic
algorithm instantiated with the container. (Earlier revisions carried
``(place_group&, sharded_array&)`` signatures alongside; they were a
redundant spelling of that instantiation — the group parameter was unused in
every algorithm body — and have been removed.)

The algorithm family (``view`` = any ``sharded_view``; every algorithm has
an explicit-environments form ``algo(view, envs, ...)`` and, for self-bound
structures, the one-argument form ``algo(view, ...)``; a trailing per-call
environment selects the contract):

- elementwise (asynchronous forms available): ``fill``, ``sequence``,
  ``iota``, ``tabulate``, ``generate``, ``for_each``, ``transform``
  (in place); ``zip_transform`` is the out-of-place spelling for any arity
  (``out[i] = op(in1[i], in2[i], ...)``, in-place into an input supported,
  co-partitioning checked);
- ``reduce`` / ``sum`` / ``min`` / ``max``: per-shard CUB ``DeviceReduce``
  plus a deterministic combine — the synchronous forms return the value;
  ``reduce_into`` is the asynchronous form, writing the aggregate through a
  device-writable output iterator on the call environment's stream
  (capture-legal);
- ``inclusive_scan`` / ``exclusive_scan`` / ``inclusive_sum`` /
  ``exclusive_sum``: reduce-then-scan — per-shard totals, a host prefix over
  the P totals, then per-shard seeded scans in place;
- ``adjacent_difference``: per-shard differences with each predecessor's
  boundary element staged through pinned host memory;
- ``sort``: global in-place sort, each shard keeping its original
  boundaries (a contiguous array reads as one globally sorted array
  afterwards). The shared-address-space engine: local per-shard sorts,
  exact splitters by multi-sequence selection, and a fused gather-merge
  loading across shard boundaries through the one address space the places
  share — it requires every shard on device-backed places of one device
  and refuses otherwise (sorting across separate address spaces is a
  distinct engine, arriving separately);
- ``segmented_reduce``: per-segment aggregates via per-shard CUB
  ``DeviceSegmentedReduce`` — the segments description is two offset views
  co-partitioned with the output (for CSR-shaped data, shifted aliases of
  one row-offsets buffer per shard). A member of the map family despite the
  name: segments are shard-local, there is no cross-shard combine, and the
  stream-bearing call form records under graph capture;
- ``count`` / ``count_if``: per-shard CUB transform-reduce plus a host sum;
- ``histogram_even``: per-shard CUB ``DeviceHistogram`` plus a per-bin sum;
- ``copy_if`` / ``filter`` / ``remove_if`` / ``unique``: in-place per-shard
  ``DeviceSelect`` over any ``owning_sharded`` structure, sizes committed
  through one atomic ``commit_sizes`` (mutation capability probed at entry
  by committing the current sizes — contiguous backing refuses there,
  before anything changes).

Algorithm temporaries are drawn from each shard's environment's memory
resource; host staging for the synchronous combines comes from a memory
resource on the per-call environment when present, and from a cached pinned
arena otherwise.

Concepts: what the algorithms are written against
--------------------------------------------------

``<cuda/experimental/__sharded/concepts.cuh>`` names the requirements the
algorithms consume — the concepts ARE the API surface; the container is the
reference model and ``place_group`` the reference provider. Three tiers,
with three lifecycles:

- **The view** (``sharded_view``): an indexed collection of shard
  descriptors — for each shard a contiguous element range (``data``,
  ``size``), its *region* in the global index space (``global_offset``), and
  an equality-comparable *place* identity. Views are plain data: no element
  ownership, no capacity, no execution resources (``basic_shard_view`` is
  the ready-made portable descriptor type). Semantic guarantees — regions
  pairwise disjoint, ordered, tiling ``[0, extent)`` exactly — are checkable
  with ``validate()``. ``owning_sharded`` refines the view with per-shard
  ``capacity``; it is where the size-mutating algorithms live.
- **Per-shard environments** (``sharded_env`` / ``sharded_env_range``):
  standard queryable environments supplying the stream to order shard ``i``'s
  work on (``cuda::get_stream``, mandatory) and — for scratch-bearing
  algorithms — a memory resource (``cuda::mr::get_memory_resource``).
  Structures built by a provider answer ``default_envs`` (the ``self_bound``
  concept, in the spirit of ``std::execution``'s ``get_env``); anything else
  is used through the explicit-environment overloads.
- **The per-call environment**: resources of the scope the whole call is
  ordered against. A stream present in it selects the *asynchronous
  contract* (the call forks from and joins into that stream and never
  synchronizes with the host); no stream selects the synchronous convenience
  form. ``sync_policy::forbid`` (the ``get_sync_policy`` query) turns every
  would-be host synchronization into a ``std::runtime_error`` thrown before
  any work, leaving all state valid — the same discipline as the capture
  guards.

Generic algorithms need no execution-place object: work launched into a
shard's stream executes in the stream's context with the stream's SM
confinement (``stream_scope`` supplies the one thing a launch needs from the
calling thread — device currency — derived from the stream itself; see
``test/sharded/stream_scope.cu``).

The pilot generic entry points are ``transform`` (in-place unary) and the
synchronous ``reduce``:

.. code:: cpp

   auto arr = sharded_array<double>::allocate(group, n);   // self-bound
   sharded::transform(arr, op);                            // envs derived
   double r = sharded::reduce(arr, cuda::std::plus<>{}, 0.0);

   // explicit environments (any sharded_view, foreign structures included)
   auto envs = sharded::default_envs(arr);
   sharded::transform(arr, envs, op);

   // asynchronous: ordered against a caller stream, no host synchronization
   const auto sp  = cuda::std::execution::prop{cuda::get_stream, stream_ref{s}};
   const auto env = cuda::std::execution::env{sp};
   sharded::transform(arr, op, env);

The container models the concepts as-is (see
``test/sharded/concepts/models.cu``), and independently-written structures
model them with zero adapters (``test/sharded/concepts/foreign_models.cu``:
a hand-rolled model over raw buffers and caller streams, and a
``vector<span<T>>`` upgraded by ``make_sharded_view``).

Size-mutating algorithms and the contiguous backing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``copy_if`` / ``filter`` / ``remove_if`` and ``unique`` shrink shard sizes in
place (capacities are unchanged; ``reset_sizes_to_capacity()`` reuses the
buffers). ``copy_if`` also has an out-of-place form —
``copy_if(src, dst, pred)`` — selecting from a read-only view into an owning
destination whose per-shard sizes become the data-dependent selected counts
(the frontier shape: derive a new ragged structure without destroying the
source; per shard the destination's capacity must cover the source's size). On a contiguous array this is unrepresentable: shrinking a shard
would leave a gap between its valid elements and the next shard's, falsifying
the read-as-one-array contract of ``contiguous_data()``, while compacting
across the gap would migrate elements onto other places than the caller asked
for. These algorithms therefore throw ``std::invalid_argument`` on contiguous
(``allocate_contiguous``) arrays, leaving them untouched. Read-only
algorithms (``count`` / ``count_if``, ``histogram_even``, ``reduce`` et al.)
remain available on every sharded array, contiguous ones included.
Asynchrony and composition
--------------------------

One rule
~~~~~~~~

A *lane* is one ordering domain: one stream per place — the environments'
streams (``place_group::envs(lane_id)``, a container's ``default_envs``, or
any environment range you build yourself). The composition contract is one
rule:

  An asynchronous call (a stream-bearing per-call environment) enqueues each
  shard's work on ``envs[i]`` and touches nothing else.

Consecutive calls on the same environments are therefore ordered per lane by
stream order, and independent across lanes: a chain of calls is a pipeline
with no per-call synchronization anywhere, fields living on different lanes
(``allocate(..., lane_id)``) overlap by construction, and everything beyond
stream order is said explicitly with the verbs below. The asynchronous forms
never synchronize with the host; a call environment carrying
``sync_policy::forbid`` turns any would-be host synchronization anywhere in
the surface into an exception thrown *before* work is enqueued.

Results attach to their output's timeline: ``reduce_into`` delivers the
aggregate on the call environment's stream, so awaiting a result means
synchronizing that one stream — not the lanes that produced it, which are
already free to run the next iteration's work. (Terminators like
``reduce_into`` are the one place call-stream edges remain: their fold
consumes every lane's partial, so every lane joins the call stream — that is
the operation's meaning, not a composition policy.)

The verbs
~~~~~~~~~

All are free functions over any environment range; all are stream/event
mechanics only.

- ``barrier(envs)`` — synchronize every lane with the host. Refuses under
  ``sync_policy::forbid`` and under capture.
- ``barrier(envs, stream)`` — make ``stream`` wait for all work on every
  lane: event edges, non-blocking, capture-legal. The pipeline-boundary
  form: join the lanes into a caller's timeline, a capture origin, or a
  communicator's stream.
- ``lane_wait(envs, i, {j, ...})`` and
  ``lane_wait(envs_to, i, envs_from, {j, ...})`` — declare a cross-lane (or
  cross-field) dependency: lane ``i`` waits for the named source lanes.
  Event edges, capture-legal. A forgotten ``lane_wait`` between genuinely
  coupled lanes is a race — the same honesty as any stream programming.
- ``lane_sync(envs, i)`` — synchronize one lane with the host (refuses under
  ``forbid``/capture).
- ``copy_to_host`` / ``copy_from_host`` — synchronous by contract; natural
  pipeline endpoints.

Sealed calls, per call
~~~~~~~~~~~~~~~~~~~~~~

A call environment carrying ``composition::bracketed`` (the
``get_composition`` query) restores the fork-all/join-all seal around that
one call: every shard's work waits for the call stream, and the call stream
waits for every shard. Use it when a single call must compose with a foreign
stream as one opaque unit; leave the default (``composition::lane_ordered``)
everywhere else — a sealed call in the middle of a pipeline routes every
lane through one timeline and serializes the pipeline's width.

A two-field pipeline
~~~~~~~~~~~~~~~~~~~~

The shape that motivates the contract (see
``examples/places/sharded_multi_field_pipeline.cu`` for the complete
program): two fields on distinct lanes, per-field chains overlapping, one
declared coupling per iteration, and a convergence check that awaits only
the residual's stream:

.. code-block:: cpp

   for (int k = 0; k < iters; k++)
   {
     transform(x, envs_x, step_x, ce_x);                    // x's lanes
     transform(y, envs_y, step_y, ce_r);                    // y's lanes (overlaps x)
     reduce_into(x, envs_x, d_s, plus, 0.0f, ce_x);         // scalar on cx
     for (size_t i = 0; i < envs_y.size(); i++)
       lane_wait(envs_y, i, cx_range, {0});                 // the one coupling edge
     transform(y, envs_y, couple{d_s}, ce_r);
     barrier(envs_y, stream_ref{cx});                       // slot-reuse reverse edge
     reduce_into(y, envs_y, h_res, plus, 0.0f, ce_r);       // residual on cr
   }
   cudaStreamSynchronize(cr);                               // await the RESULT

CUDA graph capture
------------------

The sharded surface splits cleanly along the capture boundary: elementwise
computation records into a CUDA graph; everything that allocates, transfers
host data or synchronizes refuses cleanly instead of corrupting the capture.

What captures
~~~~~~~~~~~~~

The asynchronous forms — the elementwise family, ``zip_transform``,
``segmented_reduce`` and ``reduce_into`` called with a stream-bearing
per-call environment — are pure per-shard stream work, so a pipeline
captures directly. Under the lane-ordered contract the pipeline forks the
lanes from the capture origin ONCE, records its chain (per-lane stream order
becomes graph edges within each lane; distinct lanes become graph-level
parallelism), and joins the lanes back with the stream barrier:

.. code-block:: cpp

   const auto ce = cuda::std::execution::env{
     cuda::std::execution::prop{cuda::get_stream, stream_ref{origin}}};
   auto envs = default_envs(out);

   cudaStreamBeginCapture(origin, cudaStreamCaptureModeGlobal);
   out.fork_from(origin);                 // the lanes join the capture, once
   zip_transform(out, envs, op, ce, data);
   for_each(out, envs, update, ce);
   reduce_into(out, envs, residual_slot, cuda::std::plus<>{}, 0.0, ce);
   barrier(envs, stream_ref{origin});     // the lanes rejoin the origin
   cudaStreamEndCapture(origin, &graph);

A lane-ordered call whose call stream is capturing while the lanes are not
refuses at entry (the work would silently escape the graph); fork the lanes
first, or seal that call with ``composition::bracketed``.

The last line is new capability relative to the container era: the
asynchronous reduce keeps its cross-shard combine on-device (a deterministic
fold kernel bitwise-identical to the synchronous host fold), so the whole
iterate-and-reduce shape replays as one graph, with the aggregate landing in
a device or pinned location per replay.

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
- synchronization: ``sharded_array::sync`` and ``place_group::sync``;
- the synchronous forms, all of which stage per-shard partials through the
  host and refuse at ENTRY, before any work is enqueued: ``reduce`` /
  ``sum`` / ``min`` / ``max``, the scans, ``count`` / ``count_if``,
  ``histogram_even``, ``adjacent_difference``,
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
