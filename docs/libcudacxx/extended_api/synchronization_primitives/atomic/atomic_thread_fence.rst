.. _libcudacxx-extended-api-synchronization-atomic-atomic-thread-fence:

cuda::atomic::atomic_thread_fence
=====================================

Defined in header ``<cuda/atomic>``:

.. code:: cuda

   __host__ __device__
   void cuda::atomic_thread_fence(cuda::std::memory_order order,
                                  cuda::thread_scope scope = cuda::thread_scope_system);

Establishes memory synchronization ordering of non-atomic and relaxed atomic accesses, as instructed by ``order``,
for all threads within ``scope`` without an associated atomic operation. It has the same semantics as
`cuda::std::atomic_thread_fence <https://en.cppreference.com/w/cpp/atomic/atomic_thread_fence>`_.

Example
-------

The following code is an example of the :ref:`MessagePassing <libcudacxx-extended-api-memory-model-message-passing>` pattern:

.. code:: cuda

   #include <cstdio>
   #include <cuda/atomic>
   #include <cooperative_groups.h>

   namespace cg = cooperative_groups;

   __global__ void example_kernel(int* data, cuda::std::atomic_flag* flag) {
     assert(cg::grid_group::size() == 2);
     assert(cg::thread_block::size() == 1);

     if (blockIdx.x == 0) {
       *data = 42;
       cuda::atomic_thread_fence(cuda::memory_order_release,
                                 cuda::thread_scope_device);
       flag->test_and_set(cuda::std::memory_order_relaxed);
       flag->notify_one();
     }
     else {
       // an atomic operation is required to set up the synchronization
       flag->wait(false, cuda::std::memory_order_relaxed);
       cuda::atomic_thread_fence(cuda::memory_order_acquire,
                                 cuda::thread_scope_device);
       std::printf("%d\n", *data); // Prints 42
     }
   }

`See it on Godbolt <https://godbolt.org/z/aG37o5qxx>`_
