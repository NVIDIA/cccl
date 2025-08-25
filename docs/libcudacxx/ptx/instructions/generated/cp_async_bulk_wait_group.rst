..
   This file was automatically generated. Do not edit.

cp.async.bulk.wait_group
^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.wait_group N; // PTX ISA 80, SM_90
   template <int N32>
   __device__ static inline void cp_async_bulk_wait_group(
     cuda::ptx::n32_t<N32> N);

cp.async.bulk.wait_group.read
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.wait_group.read N; // PTX ISA 80, SM_90
   template <int N32>
   __device__ static inline void cp_async_bulk_wait_group_read(
     cuda::ptx::n32_t<N32> N);
