.. _libcudacxx-standard-api-container-mdspan:

``<cuda/std/mdspan>``
======================

Provided functionalities
------------------------

-  All features of ``<mdspan>`` are made available in C++17 onwards
-  C++26 ``std::dims`` is made available in C++17 onwards
-  C++26 ``std::aligned_accessor`` is made available in C++17 onwards

Extensions
----------

-  The C++23 multidimensional ``operator[]`` is replaced with ``operator()`` in previous C++ standards
-  Detection of out-of-bounds accesses is available in debug mode

Restrictions
------------

-  On device no exceptions are thrown in case of a bad access.
