.. _libcudacxx-standard-api-synchronization:

Synchronization Library
=======================

Any Standard C++ header not listed below is omitted.

.. list-table::
   :widths: 25 45 30
   :header-rows: 1

   * - Header
     - Content
     - Availability
   * - `\<cuda/std/atomic\> <https://en.cppreference.com/w/cpp/header/atomic>`_
     - Atomic objects and operations. See also :ref:`Extended API <libcudacxx-extended-api-synchronization-atomic>`
     - libcu++ 1.0.0 / CCCL 2.0.0 / CUDA 10.2
   * - `\<cuda/std/latch\> <https://en.cppreference.com/w/cpp/header/latch>`_
     - Single-phase asynchronous thread-coordination mechanism. See also :ref:`Extended API <libcudacxx-extended-api-synchronization-latch>`
     - libcu++ 1.1.0 / CCCL 2.0.0 / CUDA 11.0
   * - `\<cuda/std/barrier\> <https://en.cppreference.com/w/cpp/header/barrier>`_
     - Multi-phase asynchronous thread-coordination mechanism. See also :ref:`Extended API <libcudacxx-extended-api-synchronization-barrier>`
     - libcu++ 1.1.0 / CCCL 2.0.0 / CUDA 11.0
   * - `\<cuda/std/semaphore\> <https://en.cppreference.com/w/cpp/header/semaphore>`_
     - Primitives for constraining concurrent access. See also :ref:`Extended API <libcudacxx-extended-api-synchronization-counting-semaphore>`
     - libcu++ 1.1.0 / CCCL 2.0.0 / CUDA 11.0
