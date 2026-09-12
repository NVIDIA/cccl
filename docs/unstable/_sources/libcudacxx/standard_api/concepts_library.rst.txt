.. _libcudacxx-standard-api-concepts:

Concepts Library
=======================

.. list-table::
   :widths: 25 45 30
   :header-rows: 1

   * - Header
     - Content
     - Availability
   * - `\<cuda/std/concepts\> <https://en.cppreference.com/w/cpp/header/concepts>`_
     - Fundamental library concepts
     - CCCL 2.1.0 / CUDA 12.2

Extensions
----------

-  All library features are available from C++14 onwards. The concepts
   can be used like type traits prior to C++20.

  .. code:: cpp

    template<cuda::std::integral Integer>
    void do_something_with_integers_in_cpp20(Integer&& i) {...}

    template<class Integer, cuda::std::enable_if_t<cuda::std::integral<Integer>, int> = 0>
    void do_something_with_integers_in_cpp17(Integer&& i) {...}

    template<class Integer, cuda::std::enable_if_t<cuda::std::integral<Integer>, int> = 0>
    void do_something_with_integers_in_cpp14(Integer&& i) {...}

Restrictions
------------

-  Subsumption does not work prior to C++20

  .. code:: cpp

    template<class Integer, cuda::std::enable_if_t<subsuming_concept<Integer> && true, int> = 0>
    void would_be_preferred_overload_in_cpp20(Integer&& i) {...}

    template<class Integer, cuda::std::enable_if_t<cuda::std::integral<Integer>, int> = 0>
    void is_always_ambiguous_in_cpp17(Integer&& i) {...}

-  Subsumption is only partially implemented in the compiler until nvcc 12.4

  nvcc has issues detecting subsumption of concepts that are composed of multiple concepts
