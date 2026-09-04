.. _libcudacxx-extended-api-synchronization-binary-semaphore:

``cuda::binary_semaphore``
==========================

Defined in header ``<cuda/semaphore>``:

.. code:: cpp

   namespace cuda {

   template <cuda::thread_scope Scope>
   using binary_semaphore = cuda::std::counting_semaphore<Scope, 1>;

   }

The class template ``cuda::binary_semaphore`` is an extended form of `cuda::std::binary_semaphore <https://en.cppreference.com/w/cpp/thread/counting_semaphore>`_
that takes an additional :ref:`cuda::thread_scope <libcudacxx-extended-api-memory-model-thread-scopes>` argument.
``cuda::binary_semaphore`` has the same interface and semantics as
`cuda::std::binary_semaphore <https://en.cppreference.com/w/cpp/thread/counting_semaphore>`_, but
``cuda::binary_semaphore`` is a class template.

Concurrency Restrictions
------------------------

An object of type ``cuda::binary_semaphore`` or ``cuda::std::binary_semaphore``, shall not be accessed concurrently by
CPU and GPU threads unless:

   - it is in unified memory and the `concurrentManagedAccess property <https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_116f9619ccc85e93bc456b8c69c80e78b>`_
     is 1, or
   - it is in CPU memory and the `hostNativeAtomicSupported property <https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_1ef82fd7d1d0413c7d6f33287e5b6306f>`_
     is 1.

Note, for objects of scopes other than ``cuda::thread_scope_system`` this is a data-race, and therefore also prohibited
regardless of memory characteristics.

Under CUDA Compute Capability 6 (Pascal) or prior, an object of type ``cuda::binary_semaphore`` or
``cuda::std::binary_semaphore`` may not be used.

Implementation-Defined Behavior
-------------------------------

For each :ref:`cuda::thread_scope <libcudacxx-extended-api-memory-model-thread-scopes>` ``S``,
``cuda::binary_semaphore<S>::max()`` is as follows:

.. list-table::
   :widths: 50 50
   :header-rows: 0

   * - :ref:`cuda::thread_scope <libcudacxx-extended-api-memory-model-thread-scopes>` ``S``
     - ``cuda::binary_semaphore<S>::max()``
   * - Any thread scope
     - ``1``

Example
-------

.. code:: cuda

   #include <cuda/semaphore>

   __global__ void example_kernel() {
     // This semaphore is suitable for all threads in the system.
     cuda::binary_semaphore<cuda::thread_scope_system> a;

     // This semaphore has the same type as the previous one (`a`).
     cuda::std::binary_semaphore<> b;

     // This semaphore is suitable for all threads on the current processor (e.g. GPU).
     cuda::binary_semaphore<cuda::thread_scope_device> c;

     // This semaphore is suitable for all threads in the same thread block.
     cuda::binary_semaphore<cuda::thread_scope_block> d;
   }

`See it on Godbolt <https://godbolt.org/z/eKfjYYz58>`_
