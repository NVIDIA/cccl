.. _cccl-python-coop:

``cuda.coop``: Cooperative GPU Primitives
==========================================

``cuda.coop`` provides group-oriented building blocks for GPU kernels. A
kernel names its participating CUDA thread block, describes the values owned
by each thread with :func:`~cuda.coop.ThreadData`, and applies a collective
to that block. The initial release provides block-wide load and store
operations.

The root API is backend-independent:

.. code-block:: python

   import numpy as np

   from cuda import coop

   block = coop.this_block()
   items = coop.ThreadData(2, dtype=np.int32)
   loaded = coop.load(block, source, items)
   coop.store(block, destination, loaded)

The compiler integration supplies launch facts such as the block dimensions;
they are not repeated in the operation calls. A capable compiler context
activates its backend automatically. Importing :mod:`cuda.coop.cutlass`
provides the equivalent qualified API when code should name CUTLASS
explicitly. See :doc:`coop_cutlass` for installation and a complete runnable
example.

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

See :doc:`coop_api` for the portable root API and the qualified CUTLASS API.
