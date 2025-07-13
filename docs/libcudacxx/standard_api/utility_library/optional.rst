.. _libcudacxx-standard-api-utility-optional:

<cuda/std/optional>
=======================

See the documentation of the standard header `\<optional\> <https://en.cppreference.com/w/cpp/header/optional>`_

Extensions
----------

- All features are available from C++14 onwards.
- All features are available at compile time if the value type supports it.
- The implementation of ``optional<T&>`` is available in C++17 and later.
Restrictions
------------

-  On device no exceptions are thrown in case of a bad access.
