.. _cccl-python-coop:

``cuda.coop``: Cooperative GPU Primitives
==========================================

``cuda.coop`` provides compiler-neutral contracts for cooperative primitives
inside Python GPU kernels. The initial API describes a CUDA thread block and
reduces one scalar value per thread with CUB BlockReduce.

The root API is intentionally small:

.. code-block:: python

   from cuda import coop

   block = coop.this_block()
   total = coop.sum(block, value)
   if thread_index == 0:
       output[0] = total

All block threads must invoke the collective in converged control flow. The
result is defined only for block rank zero, so other threads must not consume
it. A compiler integration supplies the exact block dimensions from verified
launch facts; dimensions are not repeated in the operation call.

CUDA thread block reduction
----------------------------

:func:`cuda.coop.reduce` accepts the built-in selectors ``sum``,
``multiplies``, ``min``, ``max``, ``bit_and``, ``bit_or``, and ``bit_xor``.
Common spelling aliases such as ``+``, ``add``, and ``minimum`` are also
accepted. :func:`cuda.coop.sum` is the dedicated sum form.

The optional ``valid_items`` argument reduces the prefix consisting of block
ranks ``[0, valid_items)``. Every block thread still participates in the
collective. A static value must be between one and the block size, inclusive.

The optional ``algorithm`` selector chooses one deterministic CUB BlockReduce
algorithm: ``raking_commutative_only``, ``raking``, or ``warp_reductions``.
The default is ``warp_reductions``.

Importing :mod:`cuda.coop` does not make a compiler backend mandatory. Calling
a reduction outside a compatible compiler context raises a structured error.

API reference
-------------

See :doc:`coop_api` for the portable root API.
