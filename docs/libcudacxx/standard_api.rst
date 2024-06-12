.. _libcudacxx-standard-api:

Standard API
============

.. toctree::
   :hidden:
   :maxdepth: 2

   standard_api/c_library
   standard_api/concepts_library
   standard_api/container_library
   standard_api/numerics_library
   standard_api/ranges_library
   standard_api/synchronization_library
   standard_api/time_library
   standard_api/utility_library

Standard Library Backports
--------------------------

C++ Standard versions include new language features and new library features. As the name implies, language features
are new features of the language the require compiler support. Library features are simply new additions to the
Standard Library that typically do not rely on new language features nor require compiler support and could conceivably
be implemented in an older C++ Standard.

Typically, library features are only available in the particular C++ Standard version (or newer) in which they were
introduced, even if the library features do not depend on any particular language features.

In effort to make library features available to a broader set of users, the NVIDIA C++ Standard Library relaxes this
restriction. libcu++ makes a best-effort to provide access to C++ Standard Library features in older C++ Standard
versions than they were introduced. For example, the calendar functionality added to ``<chrono>`` in C++20 is made
available in C++14.

Feature availability:

-  C++17 and C++20 features of\ ``<chrono>`` available in C++14:

   -  calendar functionality, e.g.,
      ``year``,\ ``month``,\ ``day``,\ ``year_month_day``
   -  duration functions, e.g., ``floor``, ``ceil``, ``round``
   -  Note: Timezone and clocks added in C++20 are not available

-  C++17 features from ``<type_traits>`` available in C++14:

   -  Convenience ``_v`` aliases such as ``is_same_v``
   -  ``void_t``
   -  Trait operations: ``conjunction``,\ ``negation``,\ ``disjunction``
   -  ``invoke_result``

-  C++17 ``<optional>`` is available in C++14

   -  all features are available in C++14 and at compile time if the
      payload type permits it

-  C++17 ``<variant>`` is available in C++14

   -  all features are available in C++14 and at compile time if the
      payload types permit it

-  C++17/20 constexpr ``<array>`` is available in C++14.

   -  all operation on array are made constexpr in C++14 with exception
      of ``{c}rbegin`` and ``{c}rend``, which requires C++17.

-  C++20 constexpr ``<complex>`` is available in C++14.

   -  all operation on complex are made constexpr if
      ``is_constant_evaluated`` is supported.

-  C++20 ``<concepts>`` are available in C++14.

   -  all standard concepts are available in C++14 and C++17. However,
      they need to be used similar to type traits as language concepts
      are not available.

-  C++20 ``<ranges>`` are available in C++17.

   -  all ``<ranges>`` concepts are available in C++17. However, they
      need to be used similar to type traits as language concepts are
      not available.
   -  range algorithms are not implemented.
   -  views are not implemented.

-  C++20 ``<span>`` is mostly available in C++14.

   -  With the exception of the range based constructors all features
      are available in C++14 and C++17. The range based constructors are
      emulated but not 100% equivalent.

-  C++20 features of ``<functional>`` have been partially ported to
   C++17.

   -  ``bind_front`` is available in C++17.

-  C++23 ``<expected>`` is available in C++14.

   -  all features are available in C++14

-  C++23 ``<mdspan>`` is available in C++17.

   -  mdspan is feature complete in C++17 onwards.
   -  mdspan on msvc is only supported in C++20 and onwards.
