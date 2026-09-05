.. _cccl-python-coop:

``cuda.coop``: Cooperative GPU Primitives
==========================================

``cuda.coop`` provides group-oriented building blocks for GPU kernels. A
kernel names a participating CUDA thread group, describes values owned by each
thread with :func:`~cuda.coop.ThreadData`, and applies a collective to that
group.

The root API is backend-independent.

In this schematic kernel fragment, ``source`` and ``destination`` are pointer
arguments and ``count`` is the runtime number of valid tile items:

.. code-block:: python

   import numpy as np

   from cuda import coop

   block = coop.this_block()
   items = coop.ThreadData(2, dtype=np.int32)
   loaded = coop.load(block, source, items, valid_items=count, oob_default=0)
   total = coop.sum(block, loaded)
   coop.store(block, destination, loaded, valid_items=count)

The compiler integration supplies launch facts such as the block dimensions;
they are not repeated in the operation calls. Importing :mod:`cuda.coop`
remains compiler-free; a compatible compiler context activates its backend
when a collective is traced. The initial portable contract covers:

* load and store; and
* reduce, sum, and inclusive or exclusive scans.

Group memory operations
-----------------------

Both operations accept a :class:`~cuda.coop.ThreadGroup` as their first
argument. :func:`cuda.coop.load` reads a fixed number of items per thread into
``ThreadData``; :func:`cuda.coop.store` writes ``ThreadData`` back to the
destination. Each member of the selected group must participate in the same
collective. The source and destination must be contiguous, pointer-backed
memory.

Statically compact row-major, column-major, and hierarchical layouts are
accepted. Load and store traverse their raw pointers in linear storage order;
they do not apply logical multidimensional indexing or layout order.

The optional ``valid_items`` count identifies the valid prefix of a partially
filled group tile. ``load`` fills tail positions with the ``oob_default``
sentinel. ``store`` can use the same count to leave the destination tail
untouched. Both operations accept an element ``offset`` into the source or
destination; it is not a byte offset. ``oob_default`` requires ``valid_items``;
the element offset is independent. All three controls are keyword-only.
The portable load/store surface accepts block, physical-warp, and logical-warp
groups. Temporary storage and the ``warp_transpose`` and
``warp_transpose_timesliced`` algorithms are block-only; unsupported groups are
rejected before the backend creates compiler artifacts.

Portable selection and collective operations
---------------------------------------------

The root API validates group kind, dtype, payload shape, static controls, and
result layout before dispatch. Qualified backends may expose additional
selectors, but calls through :mod:`cuda.coop` use the portable intersection.

API reference
-------------

See :doc:`coop_api` for the portable root API.
