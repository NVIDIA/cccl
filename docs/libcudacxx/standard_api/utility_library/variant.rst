.. _libcudacxx-standard-api-utility-variant:

<cuda/std/variant>
======================

See the documentation of the standard header `\<variant\> <https://en.cppreference.com/w/cpp/header/variant>`_

Extensions
----------

-  All features are available from C++14 onwards.
-  All features are available at compile time if the different value types support it.

Restrictions
------------

-  On device no exceptions are thrown in case of a bad access.

Cuda specific changes
---------------------

-  ``cuda::std::visit`` utilizes recursion instead of the usual function
   pointer array. This greatly improves runtime behavior, but comes at
   the cost of increased compile times.
