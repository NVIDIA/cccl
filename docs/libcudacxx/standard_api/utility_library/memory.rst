.. _libcudacxx-standard-api-utility-memory:

<cuda/std/memory>
======================

Provided functionalities
------------------------

- ``cuda::std::addressof``. See the C++ documentation of `std::assume_aligned()<https://en.cppreference.com/w/cpp/memory/addressof>`_
- ``cuda::std::align``. See the C++ documentation of `std::assume_aligned()<https://en.cppreference.com/w/cpp/memory/align>`_
- ``cuda::std::assume_aligned``. See the C++ documentation of `std::assume_aligned()<https://en.cppreference.com/w/cpp/memory/assume_aligned>`_
- ``cuda::std::destroy_n``. See the C++ documentation of `std::assume_aligned()<https://en.cppreference.com/w/cpp/memory/destroy_n>`_
- ``cuda::std::destroy``. See the C++ documentation of `std::assume_aligned()<https://en.cppreference.com/w/cpp/memory/destroy>`_
- Uninitialized memory algorithms. See the C++ documentation of `std::assume_aligned()<https://en.cppreference.com/w/cpp/memory>`_

Extensions
----------

-  Most features are available from C++11 onwards.
-  All features are available at compile time if compiler support is sufficient.

Restrictions
------------

- The features that are explicitly named in the standard `construct_at` and `destroy_at` are only available in C++20
- The specialized memory algorithms are not parallel
