.. _cccl-python-coop-temp-storage:

Temp Storage and Shared Memory
==============================

Many cooperative primitives use temporary shared memory. ``cuda.coop`` exposes
this explicitly via :class:`coop.TempStorage`, which represents a shared-memory
buffer with a known size and alignment.

In CUB C++ this scratch region is traditionally called ``TempStorage``, and
``cuda.coop`` keeps that terminology for familiarity and parity. For the
currently supported block and warp collectives, temp storage is usually the
shared-memory-backed scratch region that the primitive uses internally.

Basic usage
-----------

.. code-block:: python

   temp_storage = coop.TempStorage(bytes, alignment)
   result = coop.block.sum(x, items_per_thread=1, temp_storage=temp_storage)

The ``bytes`` and ``alignment`` values can be obtained from a pre-created
primitive (see :ref:`two-phase usage <cccl-python-coop-two-phase>`).

You can also bind temp storage with getitem sugar:

.. code-block:: python

   temp_storage = coop.TempStorage()
   result = coop.block.reduce[temp_storage](
       x,
       binary_op=op,
       items_per_thread=1,
   )

This is equivalent to passing ``temp_storage=temp_storage`` explicitly.

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

Automatic inference
-------------------

You do not always need to pre-compute size and alignment. In single-phase code,
``coop.TempStorage()`` can act as an intent object and let rewrite infer the
required size and alignment from the primitives that use it.

.. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_load_store_api.py
   :language: python
   :dedent:
   :start-after: example-begin load_store_single_phase_implicit_temp_storage_kernel
   :end-before: example-end load_store_single_phase_implicit_temp_storage_usage

Why use ``coop.TempStorage()`` instead of ``cuda.shared.array()``?
-----------------------------------------------------------------

``cuda.shared.array()`` is the low-level primitive. ``coop.TempStorage()`` adds
cooperative-primitive-specific semantics on top:

* infers required size and alignment from the primitive calls,
* keeps several primitives coordinated around one shared region,
* supports ``sharing=`` policies,
* supports ``auto_sync=`` so rewrite can insert required barriers,
* works with ``primitive[temp_storage](...)`` sugar.

Use ``cuda.shared.array()`` directly when you want to manage the entire layout
yourself. Use ``coop.TempStorage()`` when the storage belongs to cooperative
primitives and you want cuda.coop to manage the sharp edges.

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

Two-phase is optional here: explicit ``TempStorage(bytes, alignment)`` works in
both single-phase and two-phase code, while plain ``TempStorage()`` lets the
single-phase rewrite machinery fill in the details.
