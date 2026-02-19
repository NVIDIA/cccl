.. _cccl-python-coop-two-phase:

Two-Phase Usage (Pre-Created Primitives)
========================================

Two-phase usage means you pre-create a primitive on the host, then invoke that
instance inside a kernel. This is useful when you want to:

* Query temp-storage size/alignment up front.
* Reuse a primitive across kernels or kernel launches.
* Share shared memory across multiple primitives.

Example
-------

.. code-block:: python

   import numba
   from numba import cuda
   from cuda import coop

   @cuda.jit(device=True)
   def max_op(a, b):
       return a if a > b else b

   threads_per_block = 128
   block_reduce = coop.block.reduce(
       numba.int32,
       threads_per_block,
       max_op,
       items_per_thread=1,
   )
   temp_storage_bytes = block_reduce.temp_storage_bytes
   temp_storage_alignment = block_reduce.temp_storage_alignment

   @cuda.jit
   def kernel(input, output):
       temp_storage = coop.TempStorage(
           temp_storage_bytes,
           temp_storage_alignment,
       )
       block_output = block_reduce(
           input[cuda.threadIdx.x],
           temp_storage=temp_storage,
       )
       if cuda.threadIdx.x == 0:
           output[0] = block_output

Notes
-----

* Two-phase primitives are still invoked using the same single-phase call style
  inside kernels. You simply reuse the pre-created instance.
* If you do not need explicit temp storage or shared-memory coordination, prefer
  single-phase usage.
