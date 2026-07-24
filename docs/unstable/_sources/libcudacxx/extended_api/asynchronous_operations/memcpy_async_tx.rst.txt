.. _libcudacxx-extended-api-asynchronous-operations-memcpy-async-tx:

``cuda::device::memcpy_async_tx``
=================================

Defined in header ``<cuda/barrier>``:

.. code:: cuda

   template <typename T, size_t Alignment>
   inline __device__
   void cuda::device::memcpy_async_tx(
     T* dest,
     const T* src,
     cuda::aligned_size_t<Alignment> size,
     cuda::barrier<cuda::thread_scope_block>& bar);

Copies ``size`` bytes from global memory ``src`` to shared memory ``dest`` and decrements the transaction count of ``bar`` by ``size`` bytes.

Preconditions
-------------

-  ``src``, ``dest`` are 16-byte aligned and ``size`` is a multiple of 16, i.e., ``Alignment >= 16``.
-  ``dest`` points to a shared memory allocation that is at least ``size`` bytes wide.
-  ``src`` points to a global memory allocation that is at least ``size`` bytes wide.
-  ``bar`` is located in shared memory
-  If either ``destination`` or ``source`` is an invalid or null pointer, the behavior is undefined (even if ``count`` is zero).

Requires
--------

-  ``is_trivially_copyable_v<T>`` is true.

Notes
-----

This function can only be used under CUDA Compute Capability 9.0 (Hopper) or higher.

There is no feature flag to check if ``cuda::device::memcpy_async_tx`` is available.

**Comparison to cuda::memcpy_async**: ``memcpy_async_tx`` supports a subset of the operations of ``memcpy_async``.
It gives more control over the synchronization with a barrier than ``memcpy_async``.
Currently, ``memcpy_async_tx`` has no synchronous fallback mechanism., i.e., it currently does not work on older hardware
(pre-CUDA Compute Capability 9.0, i.e., Hopper).

.. _libcudacxx-extended-api-asynchronous-operations-memcpy-async-tx-example:

Example
-------

.. code:: cuda

   #include <cuda/barrier>
   #include <cuda/std/utility> // cuda::std::move

   #if defined(__CUDA_MINIMUM_ARCH__) && __CUDA_MINIMUM_ARCH__ < 900
   static_assert(false, "Insufficient CUDA Compute Capability: cuda::device::memcpy_async_tx is not available.");
   #endif // __CUDA_MINIMUM_ARCH__

   __device__ alignas(16) int gmem_x[2048];

   __device__ inline bool elect_one() {
     const unsigned int tid = threadIdx.x;
     const unsigned int warp_id = tid / 32;
     const unsigned int uniform_warp_id = __shfl_sync(0xFFFFFFFF, warp_id, 0); // broadcast from lane 0
     return (uniform_warp_id == 0 && cuda::ptx::elect_sync(0xFFFFFFFF)); // elect a leader thread among warp 0
   }

   __global__ void example_kernel() {
     alignas(16) __shared__ int smem_x[1024];
     #pragma nv_diag_suppress static_var_with_dynamic_init
     __shared__ cuda::barrier<cuda::thread_scope_block> bar;

     // setup the mbarrier
     if (threadIdx.x == 0) {
       init(&bar, blockDim.x);
     }
     __syncthreads();

     // issue the async copy from a single thread and wait for completion
     const bool is_block_leader = elect_one();
     const int tx_count = is_block_leader ? sizeof(smem_x) : 0;
     if (is_block_leader) {
       cuda::device::memcpy_async_tx(smem_x, gmem_x, cuda::aligned_size_t<16>(tx_count), bar);
     }
     auto token = cuda::device::barrier_arrive_tx(bar, 1, tx_count);
     bar.wait(cuda::std::move(token));

     // smem_x contains the contents of gmem_x[0], ..., gmem_x[1023]
     smem_x[threadIdx.x] += 1;
   }

`See it on Godbolt <https://godbolt.org/z/M8zqnrz9b>`_
