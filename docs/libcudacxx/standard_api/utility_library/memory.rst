.. _libcudacxx-standard-api-utility-memory:

<cuda/std/memory>
===================

Provided functionalities
------------------------

- ``cuda::std::addressof``. See the C++ documentation of `std::addressof <https://en.cppreference.com/w/cpp/memory/addressof>`_
- ``cuda::std::align``. See the C++ documentation of `std::align <https://en.cppreference.com/w/cpp/memory/align>`_
- ``cuda::std::assume_aligned``. See the C++ documentation of `std::assume_aligned <https://en.cppreference.com/w/cpp/memory/assume_aligned>`_
- ``cuda::std::start_lifetime_as`` and ``cuda::std::start_lifetime_as_array``. See the C++ documentation of `std::start_lifetime_as <https://en.cppreference.com/w/cpp/memory/start_lifetime_as>`_
- Uninitialized memory algorithms. See the C++ documentation `<https://en.cppreference.com/w/cpp/memory>`_

Extensions
----------

-  Most features are available from C++11 onwards.
-  ``cuda::std::addressof`` is constexpr from C++11 on if compiler support is available
-  ``cuda::std::assume_aligned`` is constexpr from C++14 on

Restrictions
------------

- `construct_at` and is only available in C++20 as that is explicitly mentioned in the standard
- The implicit-lifetime requirement of `start_lifetime_as` is only diagnosed if the compiler provides `__builtin_is_implicit_lifetime`
- The specialized memory algorithms are not parallel
