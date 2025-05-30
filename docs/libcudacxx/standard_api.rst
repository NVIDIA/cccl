.. _libcudacxx-standard-api:

Standard API
============

.. toctree::
   :hidden:
   :maxdepth: 2

   standard_api/c_library
   standard_api/concepts_library
   standard_api/container_library
   standard_api/execution_library
   standard_api/numerics_library
   standard_api/ranges_library
   standard_api/synchronization_library
   standard_api/time_library
   standard_api/type_support
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

----

.. |V| unicode:: U+2705

C++ Feature Availability Summary
--------------------------------

.. table::
    :widths: 25 25 8 8 8 8 20

    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+----------------+-------------+-------------+-------------+----------------------------------------------------------------------------------------------------------------+
    | **Library**                                                                        | ``Libcu++``                                                                          |  **≤ C++17**   +   **C++20** |   **C++23** |   **C++26** | **Notes**                                                                                                      |
    +====================================================================================+======================================================================================+================+=============+=============+=============+================================================================================================================+
    | `<array> <https://en.cppreference.com/w/cpp/header/array>`_                        | :ref:`<cuda/std/array> <libcudacxx-standard-api-container-array>`                    |   |V|          |  |V|        |             |             |                                                                                                                |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+----------------+-------------+-------------+-------------+----------------------------------------------------------------------------------------------------------------+
    | `<atomic> <https://en.cppreference.com/w/cpp/header/atomic>`_                      | ``<cuda/std/atomic>``                                                                |   |V|          |  |V|        |             |             |                                                                                                                |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+----------------+-------------+-------------+-------------+----------------------------------------------------------------------------------------------------------------+
    | `<barrier> <https://en.cppreference.com/w/cpp/header/barrier>`_                    | ``<cuda/std/barrier>``                                                               |                |  |V|        |             |             |                                                                                                                |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+----------------+-------------+-------------+-------------+----------------------------------------------------------------------------------------------------------------+
    | `<bit> <https://en.cppreference.com/w/cpp/header/bit>`_                            | :ref:`<cuda/std/bit> <libcudacxx-standard-api-numerics-bit>`                         |                |  |V|        |             |             |                                                                                                                |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+----------------+-------------+-------------+-------------+----------------------------------------------------------------------------------------------------------------+
    | `<bitset> <https://en.cppreference.com/w/cpp/header/bitset>`_                      | :ref:`<cuda/std/bitset> <libcudacxx-standard-api-utility-bitset>`                    |   |V|          |             |             |             |                                                                                                                |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+----------------+-------------+-------------+-------------+----------------------------------------------------------------------------------------------------------------+
    | `<chrono> <https://en.cppreference.com/w/cpp/header/chrono>`_                      | :ref:`<cuda/std/chrono> <libcudacxx-standard-api-time>`                              |   |V|          |  |V|        |             |             | Timezone and clocks added in C++20 are not available                                                           |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+----------------+-------------+-------------+-------------+----------------------------------------------------------------------------------------------------------------+
    | `<complex> <https://en.cppreference.com/w/cpp/header/complex>`_                    | :ref:`<cuda/std/complex> <libcudacxx-standard-api-numerics-complex>`                 |   |V|          |  |V|        |             |  |V|        | ``constexpr`` if ``is_constant_evaluated`` is supported                                                        |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+----------------+-------------+-------------+-------------+----------------------------------------------------------------------------------------------------------------+
    | `<concepts> <https://en.cppreference.com/w/cpp/header/concepts>`_                  | :ref:`<cuda/std/concepts> <libcudacxx-standard-api-concepts>`                        |                |  |V|        |             |             |                                                                                                                |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+----------------+-------------+-------------+-------------+----------------------------------------------------------------------------------------------------------------+
    | `<execution> <https://en.cppreference.com/w/cpp/header/execution>`_                | :ref:`<cuda/std/execution> <libcudacxx-standard-api-execution>`                      |                |             |             |  |V|        | Only ``cuda::std::execution::prop``, ``cuda::std::execution::env``, ``cuda::std::execution::get_env`` for now  |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+----------------+-------------+-------------+-------------+----------------------------------------------------------------------------------------------------------------+
    | `<expected> <https://en.cppreference.com/w/cpp/header/expected>`_                  | :ref:`<cuda/std/expected> <libcudacxx-standard-api-utility-expected>`                |                |             |  |V|        |             |                                                                                                                |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+----------------+-------------+-------------+-------------+----------------------------------------------------------------------------------------------------------------+
    | `<functional> <https://en.cppreference.com/w/cpp/header/functional>`_              | :ref:`<cuda/std/functional> <libcudacxx-standard-api-utility-functional>`            |   |V|          |             |             |             |                                                                                                                |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+----------------+-------------+-------------+-------------+----------------------------------------------------------------------------------------------------------------+
    | `<initializer_list> <https://en.cppreference.com/w/cpp/header/initializer_list>`_  | ``<cuda/std/initializer_list>``                                                      |   |V|          |             |             |             |                                                                                                                |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+----------------+-------------+-------------+-------------+----------------------------------------------------------------------------------------------------------------+
    | `<inplace_vector> <https://en.cppreference.com/w/cpp/header/inplace_vector>`_      | :ref:`<cuda/std/inplace_vector> <libcudacxx-standard-api-container-inplace-vector>`  |                |             |             |  |V|        |                                                                                                                |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+----------------+-------------+-------------+-------------+----------------------------------------------------------------------------------------------------------------+
    | `<iterator> <https://en.cppreference.com/w/cpp/header/iterator>`_                  | ``<cuda/std/iterator>``                                                              |   |V|          |             |             |             |                                                                                                                |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+----------------+-------------+-------------+-------------+----------------------------------------------------------------------------------------------------------------+
    | `<latch> <https://en.cppreference.com/w/cpp/header/latch>`_                        | ``<cuda/std/latch>``                                                                 |                |  |V|        |             |             |                                                                                                                |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+----------------+-------------+-------------+-------------+----------------------------------------------------------------------------------------------------------------+
    | `<limits> <https://en.cppreference.com/w/cpp/header/limits>`_                      | ``<cuda/std/limits>``                                                                |   |V|          |             |             |             |                                                                                                                |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+----------------+-------------+-------------+-------------+----------------------------------------------------------------------------------------------------------------+
    | `<linalg> <https://en.cppreference.com/w/cpp/header/linalg>`_                      | :ref:`<cuda/std/linalg> <libcudacxx-standard-api-numerics-linalg>`                   |                |             |             |  |V|        | Accessors, transposed layout, and related functions                                                            |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+----------------+-------------+-------------+-------------+----------------------------------------------------------------------------------------------------------------+
    | `<mdspan> <https://en.cppreference.com/w/cpp/header/mdspan>`_                      | :ref:`<cuda/std/mdspan> <libcudacxx-standard-api-container-mdspan>`                  |                |             |  |V|        |  |V|        |                                                                                                                |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+----------------+-------------+-------------+-------------+----------------------------------------------------------------------------------------------------------------+
    | `<memory> <https://en.cppreference.com/w/cpp/header/memory>`_                      | :ref:`<cuda/std/memory> <libcudacxx-standard-api-utility-memory>`                    |   |V|          |  |V|        |             |             | ``cuda::std::addressof``, ``cuda::std::align``, ``cuda::std::assume_aligned``, Uninitialized memory algorithms |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+----------------+-------------+-------------+-------------+----------------------------------------------------------------------------------------------------------------+
    | `<numbers> <https://en.cppreference.com/w/cpp/header/numbers>`_                    | :ref:`<cuda/std/numbers> <libcudacxx-standard-api-numerics-numbers>`                 |                |  |V|        |             |             |                                                                                                                |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+----------------+-------------+-------------+-------------+----------------------------------------------------------------------------------------------------------------+
    | `<numeric> <https://en.cppreference.com/w/cpp/header/numeric>`_                    | :ref:`<cuda/std/numeric> <libcudacxx-standard-api-numerics-numeric>`                 |   |V|          |             |             |             |                                                                                                                |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+----------------+-------------+-------------+-------------+----------------------------------------------------------------------------------------------------------------+
    | `<optional> <https://en.cppreference.com/w/cpp/header/optional>`_                  | :ref:`<cuda/std/optional> <libcudacxx-standard-api-utility-optional>`                |   |V|          |             |             |             |                                                                                                                |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+----------------+-------------+-------------+-------------+----------------------------------------------------------------------------------------------------------------+
    | `<ranges> <https://en.cppreference.com/w/cpp/header/ranges>`_                      | :ref:`<cuda/std/ranges> <libcudacxx-standard-api-ranges>`                            |   |V|          |  |V|        |             |             | Requires C++20, Range based algorithms and views are not provided                                              |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+----------------+-------------+-------------+-------------+----------------------------------------------------------------------------------------------------------------+
    | `<ratio> <https://en.cppreference.com/w/cpp/header/ratio>`_                        | ``<cuda/std/ratio>``                                                                 |   |V|          |             |             |             |                                                                                                                |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+----------------+-------------+-------------+-------------+----------------------------------------------------------------------------------------------------------------+
    | `<semaphore> <https://en.cppreference.com/w/cpp/header/semaphore>`_                | ``<cuda/std/semaphore>``                                                             |                |  |V|        |             |             |                                                                                                                |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+----------------+-------------+-------------+-------------+----------------------------------------------------------------------------------------------------------------+
    | `<source_location> <https://en.cppreference.com/w/cpp/header/source_location>`_    | ``<cuda/std/source_location>``                                                       |                |  |V|        |             |             |                                                                                                                |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+----------------+-------------+-------------+-------------+----------------------------------------------------------------------------------------------------------------+
    | `<span> <https://en.cppreference.com/w/cpp/header/span>`_                          | :ref:`<cuda/std/span> <libcudacxx-standard-api-container-span>`                      |                |  |V|        |             |             |                                                                                                                |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+----------------+-------------+-------------+-------------+----------------------------------------------------------------------------------------------------------------+
    | `<tuple> <https://en.cppreference.com/w/cpp/header/tuple>`_                        | :ref:`<cuda/std/tuple> <libcudacxx-standard-api-utility-tuple>`                      |   |V|          |             |  |V|        |             |                                                                                                                |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+----------------+-------------+-------------+-------------+----------------------------------------------------------------------------------------------------------------+
    | `<type_traits> <https://en.cppreference.com/w/cpp/header/type_traits>`_            | :ref:`<cuda/std/type_traits> <libcudacxx-standard-api-utility-type-traits>`          |   |V|          |  |V|        |             |             |                                                                                                                |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+----------------+-------------+-------------+-------------+----------------------------------------------------------------------------------------------------------------+
    | `<utility> <https://en.cppreference.com/w/cpp/header/utility>`_                    | :ref:`<cuda/std/utility> <libcudacxx-standard-api-utility-utility>`                  |   |V|          |  |V|        |             |             |                                                                                                                |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+----------------+-------------+-------------+-------------+----------------------------------------------------------------------------------------------------------------+
    | `<variant> <https://en.cppreference.com/w/cpp/header/variant>`_                    | :ref:`<cuda/std/variant> <libcudacxx-standard-api-utility-variant>`                  |   |V|          |             |             |             |                                                                                                                |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+----------------+-------------+-------------+-------------+----------------------------------------------------------------------------------------------------------------+
    | `<version> <https://en.cppreference.com/w/cpp/header/version>`_                    | :ref:`<cuda/std/version> <libcudacxx-standard-api-utility-version>`                  |   |V|          |             |             |             |                                                                                                                |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+----------------+-------------+-------------+-------------+----------------------------------------------------------------------------------------------------------------+

----

C Feature Availability Summary
------------------------------

.. table::
    :widths: 25 25 10

    +--------------------------------------------------------------------+--------------------------+---------------+
    | **Library**                                                        |  ``Libcu++``             | **≤ C++17**   |
    +====================================================================+==========================+===============+
    | `<cassert> <https://en.cppreference.com/w/cpp/header/cassert>`_    | ``<cuda/std/cassert>``   |  |V|          |
    +--------------------------------------------------------------------+--------------------------+---------------+
    | `<ccomplex> <https://en.cppreference.com/w/cpp/header/ccomplex>`_  | ``<cuda/std/ccomplex>``  |  |V|          |
    +--------------------------------------------------------------------+--------------------------+---------------+
    | `<cfloat> <https://en.cppreference.com/w/cpp/header/cfloat>`_      | ``<cuda/std/cfloat>``    |  |V|          |
    +--------------------------------------------------------------------+--------------------------+---------------+
    | `<climits> <https://en.cppreference.com/w/cpp/header/climits>`_    | ``<cuda/std/climits>``   |  |V|          |
    +--------------------------------------------------------------------+--------------------------+---------------+
    | `<cmath> <https://en.cppreference.com/w/cpp/header/cmath>`_        |  ``<cuda/std/cmath>``    |  |V|          |
    +--------------------------------------------------------------------+--------------------------+---------------+
    | `<cstddef> <https://en.cppreference.com/w/cpp/header/cstddef>`_    | ``<cuda/std/cstddef>``   |  |V|          |
    +--------------------------------------------------------------------+--------------------------+---------------+
    | `<cstdint> <https://en.cppreference.com/w/cpp/header/cstdint>`_    | ``<cuda/std/cstdint>``   |  |V|          |
    +--------------------------------------------------------------------+--------------------------+---------------+
    | `<cstdlib> <https://en.cppreference.com/w/cpp/header/cstdlib>`_    | ``<cuda/std/cstdlib>``   |  |V|          |
    +--------------------------------------------------------------------+--------------------------+---------------+
    | `<cstring> <https://en.cppreference.com/w/cpp/header/cstring>`_    | ``<cuda/std/cstring>``   |  |V|          |
    +--------------------------------------------------------------------+--------------------------+---------------+
    | `<ctime> <https://en.cppreference.com/w/cpp/header/ctime>`_        | ``<cuda/std/ctime>``     |  |V|          |
    +--------------------------------------------------------------------+--------------------------+---------------+

see :ref:`C Libraries <libcudacxx-standard-api-c-compat>` for more details
