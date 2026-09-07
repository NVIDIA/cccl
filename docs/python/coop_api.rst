.. _cuda_coop-module:

``cuda.coop`` API Reference
===========================

Portable API
------------

.. automodule:: cuda.coop
   :members:
   :imported-members:

CUTLASS API
-----------

``cuda.coop.cutlass`` validates its optional runtime dependency when imported,
so its small public surface is documented explicitly for dependency-free
documentation builds. The qualified operations have the same contracts as the
portable operations above.

.. py:module:: cuda.coop.cutlass

.. py:class:: ThreadData(items_per_thread, dtype=None)

   Create an uninitialized per-thread register payload.

   This is the qualified export of :func:`cuda.coop.ThreadData`.

.. py:class:: ThreadGroup

   Descriptor for the current CUDA thread block.

   This is the qualified export of :class:`cuda.coop.ThreadGroup`.

.. py:function:: this_block()

   Return a descriptor for the current CUDA thread block.

   See :func:`cuda.coop.this_block`.

.. py:function:: load(group, source, items, /, *, valid_items=None, oob_default=None, offset=None)

   Collectively load one block tile into a per-thread payload.

   See :func:`cuda.coop.load`.

.. py:function:: store(group, destination, items, /, *, valid_items=None, offset=None)

   Collectively store one per-thread payload as one block tile.

   See :func:`cuda.coop.store`.
