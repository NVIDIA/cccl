.._cudax
    - sharded
    :

    Sharded containers and algorithms
  == == == == == == == == == == == == == == == == =

  ..contents::
    : depth : 2

      Sharded containers partition one logical array across the places of a single process — devices
    , or sub - device locality domains — while keeping a common address space.They extend the cooperation
           - scope structure CUDA algorithms already follow
    : a primitive at one scope runs the previous scope's primitive locally and combines results using what the new scope
      shares(registers and shuffles within a warp, shared memory within a block, global memory within a device)
        .At the places scope
    , what is shared is one virtual address space with placed pages;
at the multi - process / multi - node scope, where nothing is shared,
  communicator
    - based algorithms take over(see the MGMN algorithms built on ``__multi_gpu``)
          .

        The sharded API lives in the ``cuda::experimental::sharded`` namespace and is available through the ``cuda
        / experimental / sharded.cuh`` header.It builds on the standalone : ref :`places<cudax - places>` layer;
execution resources come from a
    : ref
    :`place_group<places - place - group>`.

    sharded_array-- -- -- -- -- -- -

``sharded_array<T>`` is a 1D array whose shards each carry their own placement : a ``data_place``
    , an ``exec_place`` and a reference stream
          .

          ..code
        - block::cpp

        using namespace cuda::experimental::sharded;

auto group     = place_group::by_locality_domains(); // re-exported from places
const size_t n = 1u << 28;
auto data      = sharded_array<double>::allocate(group, n);

iota(group, data, 0.0);
double total = sum(group, data); // per-place CUB + combine

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
