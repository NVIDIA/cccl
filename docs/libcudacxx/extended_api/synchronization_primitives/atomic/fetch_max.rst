.. _libcudacxx-extended-api-synchronization-atomic-atomic-fetch-max:

cuda::atomic::fetch_max
===========================

Defined in header ``<cuda/atomic>``:

.. code:: cuda

   template <typename T, cuda::thread_scope Scope>
   __host__ __device__
   T cuda::atomic<T, Scope>::fetch_max(T const& val,
                                       cuda::std::memory_order order
                                         = cuda::std::memory_order_seq_cst);

Atomically find the maximum of the value stored in the ``cuda::atomic``
and ``val``. The maximum is found using
`cuda::std::max <https://en.cppreference.com/w/cpp/algorithm/max>`_.

Example
-------

.. code:: cuda

   #include <cuda/atomic>

   __global__ void example_kernel() {
     cuda::atomic<int> a(0);
     auto x = a.fetch_max(1);
     auto y = a.load();
     assert(x == 0 && y == 1);
   }

`See it on Godbolt <https://godbolt.org/z/rexn5T78G>`_
