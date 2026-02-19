.. _cccl-python-coop-temp-storage:

Temp Storage and Shared Memory
==============================

Many cooperative primitives use temporary shared memory. ``cuda.coop`` exposes
this explicitly via :class:`coop.TempStorage`, which represents a shared-memory
buffer with a known size and alignment.

Basic usage
-----------

.. code-block:: python

   temp_storage = coop.TempStorage(bytes, alignment)
   result = coop.block.sum(x, items_per_thread=1, temp_storage=temp_storage)

The ``bytes`` and ``alignment`` values can be obtained from a pre-created
primitive (see :ref:`two-phase usage <cccl-python-coop-two-phase>`).

Sharing temp storage across primitives
--------------------------------------

When multiple primitives share the same shared-memory allocation, you can
compute a single size and alignment and reuse it:

.. code-block:: python

   bytes_total = max(prim_a.temp_storage_bytes, prim_b.temp_storage_bytes)
   align = max(prim_a.temp_storage_alignment, prim_b.temp_storage_alignment)

   @cuda.jit
   def kernel(...):
       temp_storage = coop.TempStorage(bytes_total, align)
       prim_a(..., temp_storage=temp_storage)
       prim_b(..., temp_storage=temp_storage)

The :func:`coop.gpu_dataclass` helper can compute ``temp_storage_bytes_sum``,
``temp_storage_bytes_max``, and ``temp_storage_alignment`` for a collection of
primitives.

Synchronization behavior
------------------------

``TempStorage`` defaults to ``auto_sync=True``. When enabled, ``cuda.coop``
automatically inserts a ``__syncthreads()`` after each primitive call that uses
explicit temp storage. Set ``auto_sync=False`` when you want to manage
synchronization manually.

Constraints
-----------

``TempStorage`` sizes and alignments must be compile-time constants for a given
kernel specialization. If you change sizes, Numba will recompile the kernel
for that specialization.
