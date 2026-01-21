.. _libcudacxx-standard-api-numerics-numeric:

``<cuda/std/numeric>``
======================

Omissions
---------

-  Currently we do not expose any parallel algorithms.
-  Saturation arithmetics have not been implemented yet

Extensions
----------

-  All features of ``<numeric>`` are made available in C++11 onwards
-  All features of ``<numeric>`` are made constexpr in C++14 onwards
-  Algorithms that return a value and not an iterator have been marked ``[[nodiscard]]``
