.. _libcudacxx-extended-api-synchronization-atomic-ref:

``cuda::atomic_ref``
====================

.. toctree::
   :hidden:
   :maxdepth: 1

   atomic/atomic_thread_fence
   atomic/fetch_max
   atomic/fetch_min

Defined in header ``<cuda/atomic>``:

.. code:: cuda

   template <typename T, cuda::thread_scope Scope = cuda::thread_scope_system>
   class cuda::atomic_ref;

The class template ``cuda::atomic_ref`` is an extended form of `cuda::std::atomic_ref <https://en.cppreference.com/w/cpp/atomic/atomic_ref>`_
that takes an additional :ref:`cuda::thread_scope <libcudacxx-extended-api-memory-model-thread-scopes>` argument, defaulted to
``cuda::std::thread_scope_system``.

It has the same interface and semantics as `cuda::std::atomic_ref <https://en.cppreference.com/w/cpp/atomic/atomic_ref>`_,
with the following additional operations. This class additionally deviates from the standard by being backported to C++11.

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - :ref:`cuda::atomic_ref::fetch_min <libcudacxx-extended-api-synchronization-atomic-atomic-fetch-min>`
     - Atomically find the minimum of the stored value and a provided value.
   * - :ref:`cuda::atomic_ref::fetch_max <libcudacxx-extended-api-synchronization-atomic-atomic-fetch-max>`
     - Atomically find the maximum of the stored value and a provided value.


Limitations
-----------

``cuda::atomic_ref<T>`` and ``cuda::std::atomic_ref<T>`` may only be instantiated with a T that are either 4 or 8 bytes.

No object or subobject of an object referenced by an ``atomic_­ref`` shall be concurrently referenced by any other
``atomic_­ref`` that has a different ``Scope``.

For ``cuda::atomic_ref<T>`` and ``cuda::std::atomic_ref<T>`` the type ``T`` must satisfy the following:
  - ``sizeof(T) <= 8``.
  - ``T`` must not have “padding bits”, i.e., T's `object representation <https://en.cppreference.com/w/cpp/language/object#Object_representation_and_value_representation>`_
    must not have bits that do not participate in it's value representation.

Concurrency Restrictions
------------------------

See :ref:`memory model <libcudacxx-extended-api-memory-model>` documentation for general restrictions on atomicity.

With CUDA Compute Capability 6 (Pascal), an object of type ``atomic_ref`` may not be used:
  - with a reference to an object with a automatic storage duration in a GPU thread, or
  - if ``is_always_lock_free()`` is ``false``.

For CUDA Compute Capability prior to 6 (Pascal), objects of type ``cuda::atomic_ref`` or
`cuda::std::atomic_ref <https://en.cppreference.com/w/cpp/atomic/atomic_ref>`_ may not be used.

Implementation-Defined Behavior
-------------------------------

For each type ``T`` and :ref:`cuda::thread_scope <libcudacxx-extended-api-memory-model-thread-scopes>` ``S``, the value of
``cuda::atomic_ref<T, S>::is_always_lock_free()`` and ``cuda::std::atomic_ref<T>::is_always_lock_free()`` is as follows:

.. list-table::
   :widths: 25 25 50
   :header-rows: 0

   * - Type ``T``
     - :ref:`cuda::thread_scope <libcudacxx-extended-api-memory-model-thread-scopes>` ``S``
     - ``cuda::atomic_ref<T, S>::is_always_lock_free()``
   * - Any valid type
     - Any thread scope
     - ``sizeof(T) <= 8``

Types of ``T``, where ``sizeof(T) < 4``, are not natively supported by the underlying hardware. For these types atomic
operations are emulated and will be drastically slower. Contention with contiguous memory in the current 4 byte boundary
will be exacerbated. In these situations it is advisable to perform a hierarchical reduction to non-adjacent memory first.

Example
-------

.. code:: cuda

   #include <cuda/atomic>

   __global__ void example_kernel(int *gmem, int *pinned_mem) {
     // This atomic is suitable for all threads in the system.
     cuda::atomic_ref<int, cuda::thread_scope_system> a(*pinned_mem);

     // This atomic has the same type as the previous one (`a`).
     cuda::atomic_ref<int> b(*pinned_mem);

     // This atomic is suitable for all threads on the current processor (e.g. GPU).
     cuda::atomic_ref<int, cuda::thread_scope_device> c(*gmem);

     __shared__ int shared_v;
     // This atomic is suitable for threads in the same thread block.
     cuda::atomic_ref<int, cuda::thread_scope_block> d(shared_v);
   }

`See it on Godbolt <https://godbolt.org/z/fr4K7ErEh>`_
