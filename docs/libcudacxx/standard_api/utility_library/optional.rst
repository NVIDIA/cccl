.. _libcudacxx-standard-api-utility-optional:

<cuda/std/optional>
=======================

See the documentation of the standard header `\<optional\> <https://en.cppreference.com/w/cpp/header/optional>`_

Extensions
----------

- All features are available from C++14 onwards.
- All features are available at compile time if the value type supports it.

- An implementation of `P2988 optional\<T&\> <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2025/p2988r9.pdf>`_ is available by
  defining ``CCCL_ENABLE_OPTIONAL_REF``

Restrictions
------------

-  On device no exceptions are thrown in case of a bad access.
