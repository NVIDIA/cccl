.. _libcudacxx-ptx-instructions-cp-async-bulk-commit_group:

cp.async.bulk.commit_group
==========================

-  PTX ISA:
   `cp.async.bulk.commit_group <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-commit-group>`__

cp.async.bulk.commit_group
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // cp.async.bulk.commit_group; // PTX ISA 80, SM_90
   template <typename=void>
   __device__ static inline void cp_async_bulk_commit_group();
