.. _libcudacxx-standard-api-numerics-linalg:

``<cuda/std/linalg>``
============================================

Provided functionalities
------------------------

- ``scaled()`` `std::linalg::scaled <https://en.cppreference.com/w/cpp/numeric/linalg/scaled>`_
- ``scaled_accessor`` `std::linalg::scaled_accessor <https://en.cppreference.com/w/cpp/numeric/linalg/scaled_accessor>`_
- ``conjugated()`` `std::linalg::conjugated <https://en.cppreference.com/w/cpp/numeric/linalg/conjugated>`_
- ``conjugated_accessor`` `std::linalg::conjugated_accessor <https://en.cppreference.com/w/cpp/numeric/linalg/conjugated_accessor>`_
- ``transposed()`` `std::linalg::transposed <https://en.cppreference.com/w/cpp/numeric/linalg/transposed>`_
- ``layout_transpose`` `std::linalg::layout_transpose <https://en.cppreference.com/w/cpp/numeric/linalg/layout_transpose>`_
- ``conjugate_transposed()`` `std::linalg::conjugate_transposed <https://en.cppreference.com/w/cpp/numeric/linalg/conjugate_transposed>`_

Extensions
----------

-  C++26 ``std::linalg`` accessors, transposed layout, and related functions are available in C++17

Omissions
---------

-  Currently we do not expose any BLAS functions and layouts.

Restrictions
------------

-  On device no exceptions are thrown in case of a bad access.
-  MSVC is only supported with C++20
