.. _cccl-python-coop-single-phase:

Single-Phase Usage
==================

Single-phase usage is the default for ``cuda.coop``: you call a primitive
directly inside a CUDA kernel, and the rewriter specializes the primitive
based on the arguments you pass.

Block example
-------------

The snippet below shows a block-wide reduction with a custom operator:

.. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_reduce_api.py
   :language: python
   :dedent:
   :start-after: example-begin imports
   :end-before: example-end imports

.. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_reduce_api.py
   :language: python
   :dedent:
   :start-after: example-begin reduce
   :end-before: example-end reduce

Warp example
------------

Warp collectives are similar, but operate on a single warp of threads:

.. literalinclude:: ../../python/cuda_cccl/tests/coop/test_warp_scan_api.py
   :language: python
   :dedent:
   :start-after: example-begin imports
   :end-before: example-end imports

.. literalinclude:: ../../python/cuda_cccl/tests/coop/test_warp_scan_api.py
   :language: python
   :dedent:
   :start-after: example-begin exclusive-sum
   :end-before: example-end exclusive-sum

Out-parameters
--------------

Some primitives return additional information through out-parameters. For
example, block scans can write the block aggregate into a 1-element array:

.. code-block:: python

   block_aggregate = cuda.local.array(1, numba.int32)
   result = coop.block.exclusive_sum(
       value,
       block_aggregate=block_aggregate,
   )

Notes
-----

* Most primitives require ``items_per_thread`` to be a compile-time constant.
* For warp collectives, only consecutive threads in the same warp participate.
* Use :class:`coop.ThreadData` if you want the primitive to infer array size
  and dtype.
