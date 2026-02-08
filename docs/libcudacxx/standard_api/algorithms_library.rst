.. _libcudacxx-standard-api-algorithms:

Algorithms Library
===================

.. list-table::
   :widths: 25 45 30
   :header-rows: 1

   * - Header
     - Content
     - Availability
   * - `\<cuda/std/algorithm\> <https://en.cppreference.com/w/cpp/header/algorithm>`_
     - Fundamental library algorithms
     - CCCL 3.2.0 / CUDA 13.2

Extensions
----------

  - All supported algorithms are available from C++17 onwards.
  - All supported algorithms are constexpr, except the allocating ones.

Restrictions
------------

  - Algorithms in namespace `ranges` are not yet supported.
  - Some sorting algorithms are not yet supported:

    * ``inplace_merge``
    * ``nth_element``
    * ``sort``
    * ``stable_partiin``
    * ``stable_sort``
