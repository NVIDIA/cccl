.. _libcudacxx-standard-api-utility-memory:

<cuda/std/memory>
===================

Provided functionalities
------------------------

- ``cuda::std::addressof``. See the C++ documentation of `std::addressof <https://en.cppreference.com/w/cpp/memory/addressof>`_
- ``cuda::std::align``. See the C++ documentation of `std::align <https://en.cppreference.com/w/cpp/memory/align>`_
- ``cuda::std::assume_aligned``. See the C++ documentation of `std::assume_aligned <https://en.cppreference.com/w/cpp/memory/assume_aligned>`_
- Uninitialized memory algorithms. See the C++ documentation `<https://en.cppreference.com/w/cpp/memory>`_

Extensions
----------

-  Most features are available from C++11 onwards.
-  ``cuda::std::addressof`` is constexpr from C++11 on if compiler support is available
-  ``cuda::std::assume_aligned`` is constexpr from C++14 on

Restrictions
------------

- `construct_at` and is only available in C++20 as that is explicitly mentioned in the standard
- The specialized memory algorithms are not parallel
