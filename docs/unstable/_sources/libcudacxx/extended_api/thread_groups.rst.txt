.. _libcudacxx-extended-api-thread-groups:

Thread Groups
=============

.. code:: cuda

   struct ThreadGroup {
     static constexpr cuda::thread_scope thread_scope;
     Integral size() const;
     Integral thread_rank() const;
     void sync() const;
   };

The *ThreadGroup concept* defines the requirements of a type that represents a group of cooperating threads.

The `CUDA Cooperative Groups Library <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#group-collectives>`_
provides a number of types that satisfy this concept.

Data Members
------------

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - ``thread_scope``
     - The scope at which ``ThreadGroup::sync()`` synchronizes memory operations and thread execution.

Member Functions
----------------

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - ``size``
     - Returns the number of participating threads.
   * - ``thread_rank``
     - Returns a unique value for each participating thread (``0 <= ThreadGroup::thread_rank() < ThreadGroup::size()``).
   * - ``sync``
     - Synchronizes the participating threads.

Notes
-----

This concept is defined for documentation purposes but is not materialized in the library.

Example
-------

.. code:: cuda

   #include <cuda/atomic>
   #include <cuda/std/cstddef>

   struct single_thread_group {
     static constexpr cuda::thread_scope thread_scope = cuda::thread_scope::thread_scope_thread;
     cuda::std::size_t size() const { return 1; }
     cuda::std::size_t thread_rank() const { return 0; }
     void sync() const {}
   };

`See it on Godbolt <https://godbolt.org/z/6c16KxqY7>`_
