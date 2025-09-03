.. _libcudacxx-extended-api-synchronization-atomic-atomic-fetch-min:

cuda::atomic::fetch_min
===========================

Defined in header ``<cuda/atomic>``:

.. code:: cuda

   template <typename T, cuda::thread_scope Scope>
   __host__ __device__
   T cuda::atomic<T, Scope>::fetch_min(T const& val,
                                       cuda::std::memory_order order
                                         = cuda::std::memory_order_seq_cst);

Atomically find the minimum of the value stored in the ``cuda::atomic``
and ``val``. The minimum is found using
`cuda::std::min <https://en.cppreference.com/w/cpp/algorithm/min>`_.

Example
-------

.. code:: cuda

   #include <cuda/atomic>

   __global__ void example_kernel() {
     cuda::atomic<int> a(1);
     auto x = a.fetch_min(0);
     auto y = a.load();
     assert(x == 1 && y == 0);
   }

`See it on Godbolt <https://godbolt.org/z/vMj9e5hdv>`_
