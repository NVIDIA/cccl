.. _libcudacxx-standard-api-container-span:

``<cuda/std/span>``
======================

Extensions
----------

-  All features of ``<span>`` are made available in C++14 onwards
-  All features of ``<span>`` are made constexpr in C++14 onwards

Restrictions
------------

-  On device no exceptions are thrown in case of a bad access.
-  The range based constructors are emulated but not 100% equivalent.
