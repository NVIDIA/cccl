.. _libcudacxx-standard-api-ranges:

Ranges Library
=======================

See the documentation of the standard headers `\<iterator\> <https://en.cppreference.com/w/cpp/header/iterator>`_ and
`\<ranges\> <https://en.cppreference.com/w/cpp/header/ranges>`_

.. list-table::
   :widths: 25 45 30
   :header-rows: 1

   * - Header
     - Content
     - Availability
   * - `\<cuda/std/iterator\> <https://en.cppreference.com/w/cpp/header/iterator>`_
     - Iterator related concepts and machinery such as ``cuda::std::forward_iterator``
     - CCCL 2.3.0 / CUDA 12.4
   * - `\<cuda/std/ranges\> <https://en.cppreference.com/w/cpp/header/ranges>`_
     - Range related concepts and machinery such as ``cuda::std::ranges::forward_range`` and ``cuda::std::ranges::subrange``
     - CCCL 2.4.0 / CUDA 12.5

Extensions
----------

-  All library features are available from C++17 onwards. The concepts can be used like type traits prior to C++20.

  .. code:: cpp

    template<cuda::std::contiguos_range Range>
    void do_something_with_ranges_in_cpp20(Range&& range) {...}

    template<class Range, cuda::std::enable_if_t<cuda::std::contiguos_range<Range>, int> = 0>
    void do_something_with_ranges_in_cpp17(Range&& range) {...}

Restrictions
------------

-  Subsumption does not work prior to C++20
-  Subsumption is only partially implemented in the compiler until nvcc 12.4

Omissions
---------

-  Range based algorithms have *not* been implemented
-  Views have *not* been implemented
