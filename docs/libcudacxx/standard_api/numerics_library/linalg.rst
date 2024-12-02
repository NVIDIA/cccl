.. _libcudacxx-standard-api-numerics-linalg:

``<cuda/std/__linalg/scaled>``
``<cuda/std/__linalg/conjugated>``
``<cuda/std/__linalg/transposed>``
``<cuda/std/__linalg/conjugate_transposed>``
============================================

Extensions
----------

-  C++26 ``std::linalg`` accessors, transposed layout, and related functions are available in C++17

   - ``scaled()`` and ``scaled_accessor``
   - ``conjugated()`` and ``conjugated_accessor``
   - ``transposed()`` and ``layout_transpose``
   - ``conjugate_transposed()``

Omissions
---------

-  Currently we do not expose any BLAS functions and layouts.

Restrictions
------------

-  On device no exceptions are thrown in case of a bad access.
-  MSVC is only supported with C++20
