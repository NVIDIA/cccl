.. _libcudacxx-standard-api-container-mdspan:

``<cuda/std/mdspan>``
======================

Extensions
----------

-  All features of ``<mdspan>`` are made available in C++14 onwards
-  The C++23 multidimensional ``operator[]`` is replaced with ``operator()``
-  C++26 ``std::dims`` is made available in C++14 onwards

Restrictions
------------

-  On device no exceptions are thrown in case of a bad access.
-  MSVC is only supported with C++20
