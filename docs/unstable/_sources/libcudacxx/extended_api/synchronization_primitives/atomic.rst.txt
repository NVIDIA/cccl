.. _libcudacxx-extended-api-synchronization-atomic:

``cuda::atomic``
================

.. toctree::
   :hidden:
   :maxdepth: 1

   atomic/atomic_thread_fence
   atomic/fetch_max
   atomic/fetch_min

Defined in header ``<cuda/atomic>``:

.. code:: cuda

   template <typename T, cuda::thread_scope Scope = cuda::thread_scope_system>
   class cuda::atomic;

The class template ``cuda::atomic`` is an extended form of `cuda::std::atomic <https://en.cppreference.com/w/cpp/atomic/atomic>`_
that takes an additional :ref:`cuda::thread_scope <libcudacxx-extended-api-memory-model-thread-scopes>` argument,
defaulted to ``cuda::std::thread_scope_system``.

It has the same interface and semantics as `cuda::std::atomic <https://en.cppreference.com/w/cpp/atomic/atomic>`_,
with the following additional operations.

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - :ref:`cuda::atomic_thread_fence <libcudacxx-extended-api-synchronization-atomic-atomic-thread-fence>`
     - Memory order and scope dependent fence synchronization primitive.
   * - :ref:`cuda::atomic::fetch_min <libcudacxx-extended-api-synchronization-atomic-atomic-fetch-min>`
     - Atomically find the minimum of the stored value and a provided value.
   * - :ref:`cuda::atomic::fetch_max <libcudacxx-extended-api-synchronization-atomic-atomic-fetch-max>`
     - Atomically find the maximum of the stored value and a provided value.

Concurrency Restrictions
------------------------

An object of type ``cuda::atomic`` or `cuda::std::atomic <https://en.cppreference.com/w/cpp/atomic/atomic>`_
shall not be accessed concurrently by CPU and GPU threads unless:

  - it is in unified memory and the `concurrentManagedAccess property <https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_116f9619ccc85e93bc456b8c69c80e78b>`_
    is 1, or
  - it is in CPU memory and the `hostNativeAtomicSupported property <https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_1ef82fd7d1d0413c7d6f33287e5b6306f>`_
    is 1.

Note, for objects of scopes other than ``cuda::thread_scope_system`` this is a data-race, and therefore also
prohibited regardless of memory characteristics.

Under CUDA Compute Capability 6 (Pascal), an object of type ``atomic`` may not be used:

  - with automatic storage duration, or
  - if ``is_always_lock_free()`` is ``false``.

Under CUDA Compute Capability prior to 6 (Pascal), objects of type ``cuda::atomic`` or
`cuda::std::atomic <https://en.cppreference.com/w/cpp/atomic/atomic>`_ may not be used.

Implementation-Defined Behavior
-------------------------------

For each type ``T`` and :ref:`cuda::thread_scope <libcudacxx-extended-api-memory-model-thread-scopes>` ``S``,
the value of ``cuda::atomic<T, S>::is_always_lock_free()`` is as follows:

.. list-table::
   :widths: 25 25 50
   :header-rows: 0

   * - Type ``T``
     - :ref:`cuda::thread_scope <libcudacxx-extended-api-memory-model-thread-scopes>` ``S``
     - ``cuda::atomic<T, S>::is_always_lock_free()``
   * - Any valid type
     - Any thread scope
     - ``sizeof(T) <= 8``

Example
-------

.. code:: cuda

   #include <cuda/atomic>

   __global__ void example_kernel() {
     // This atomic is suitable for all threads in the system.
     cuda::atomic<int, cuda::thread_scope_system> a;

     // This atomic has the same type as the previous one (`a`).
     cuda::atomic<int> b;

     // This atomic is suitable for all threads on the current processor (e.g. GPU).
     cuda::atomic<int, cuda::thread_scope_device> c;

     // This atomic is suitable for threads in the same thread block.
     cuda::atomic<int, cuda::thread_scope_block> d;
   }

`See it on Godbolt <https://godbolt.org/z/avo3Evbee>`_
