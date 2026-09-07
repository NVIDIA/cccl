.. _cccl-python-coop:

``cuda.coop``: Cooperative GPU Primitives
==========================================

``cuda.coop`` provides group-oriented building blocks for GPU kernels. A
kernel names its participating CUDA thread group, describes the values owned
by each thread with :func:`~cuda.coop.ThreadData`, and applies a collective
to that group. Block and warp scopes are supported, and the same portable
calls compile through any registered Python DSL backend.

The root API is backend-independent:

.. code-block:: python

   import numpy as np

   from cuda import coop

   block = coop.this_block()
   items = coop.ThreadData(2, dtype=np.int32)
   loaded = coop.load(block, source, items)
   total = coop.reduce(block, loaded)
   coop.store(block, destination, loaded)

The compiler integration supplies launch facts such as the block dimensions;
they are not repeated in the operation calls. Importing ``cuda.coop`` probes
each installed compiler backend and activates the compatible ones
automatically; a kernel traced by that compiler then lowers the portable
calls through it. Two backends ship today:

* :doc:`coop_cutlass` — the CUTLASS CuTe DSL backend
  (:mod:`cuda.coop.cutlass`).
* :doc:`coop_numba_mlir` — the Numba-CUDA-MLIR backend
  (:mod:`cuda.coop.numba_mlir`).

Operation families
------------------

The portable common profile certifies these group-first operations. Each
takes the participating :class:`~cuda.coop.ThreadGroup` as its first
argument, and every thread of the group must participate:

* **Load and Store** — move one group tile between contiguous, pointer-backed
  memory and per-thread :func:`~cuda.coop.ThreadData` registers, with
  partial-tile ``valid_items``/``oob_default`` controls and element offsets.
* **Reduce and Sum** — combine per-thread values with a built-in or
  user-supplied operator; the root rank owns the scalar result unless
  broadcasting is requested.
* **Scan** — ``scan``, ``exclusive_scan``, ``inclusive_scan``,
  ``exclusive_sum``, and ``inclusive_sum`` prefix operations.
* **Exchange** — rearrange per-thread items between blocked and striped
  arrangements.
* **Shuffle** — shift the group's flattened items ``up`` or ``down`` by one
  position.
* **Adjacent Difference and Discontinuity** — neighbor differences and
  head/tail flagging across the group tile.
* **Merge Sort** — ``merge_sort_keys`` and ``merge_sort_pairs`` with optional
  custom comparators and partial-tile controls.
* **Radix Sort and Radix Rank** — ``radix_sort_keys``, ``radix_sort_pairs``,
  and keys-only ``radix_rank``, with bit-range controls.
* **Histogram** — accumulate a striped per-thread bin counter payload from
  group samples.
* **Run-Length Decode** — expand run/length pairs into a decoded group tile.
* **TopK** — ``topk_max_keys``, ``topk_min_keys``, ``topk_max_pairs``, and
  ``topk_min_pairs`` selection.

Warp-scoped groups (``this_warp`` and warp groups mapped from a block
partition) certify load/store, reduce, scan, exchange, and merge sort. The
common numeric family is ``uint8``, ``int32``, ``uint32``, ``int64``,
``uint64``, ``float32``, and ``float64``; integer-key operations use the
narrower families documented with each operation.

CUDA thread block load and store
--------------------------------

Both operations accept a :class:`~cuda.coop.ThreadGroup` as their first
argument. :func:`cuda.coop.load` reads a fixed number of items per thread into
``ThreadData``; :func:`cuda.coop.store` writes ``ThreadData`` back to the
destination. Every thread in the CUDA thread block must participate in the
same collective; a subset of threads cannot invoke it. The source and
destination must be contiguous, pointer-backed memory.

Statically compact row-major, column-major, and hierarchical layouts are
accepted. Load and Store traverse their raw pointers in linear storage order;
they do not apply logical multidimensional indexing or layout order.

The optional ``valid_items`` count identifies the valid prefix of a partially
filled block tile. When supplied, ``oob_default`` fills the remaining Load
positions. Without it, those positions are unspecified and must not be
consumed or stored. ``store`` can use the same count to leave the destination
tail untouched. Both operations accept an element ``offset`` into the source
or destination; it is not a byte offset. These controls are independent and
keyword-only.

API reference
-------------

See :doc:`coop_api` for the portable root API and the qualified backend
surfaces.
