.. _libcudacxx-tile-api:

CUDA Tile support
==================

`CUDA Tile <https://developer.nvidia.com/cuda/tile#section-more-resources>`_ introduces a new way to program GPUs at a higher level than SIMT.

Restrictions
------------

With the compiler taking more control over memory and threading there are a number of restrictions in a tile program:


C++ Concurrency support
~~~~~~~~~~~~~~~~~~~~~~

Currently the use of inline ptx / assembly is not allowed in a tile program.
All of our threading features rely on inline assembly in some capacity.
Consequently, the following headers are not supported in tile mode:

    - <cuda/atomic>
    - <cuda/barrier>
    - <cuda/latch>
    - <cuda/pipeline>
    - <cuda/semaphore>
    - <cuda/std/atomic>
    - <cuda/std/barrier>
    - <cuda/std/execution>
    - <cuda/std/latch>
    - <cuda/std/semaphore>

This also affects

    - <cuda/cmath>
    - <cuda/discard_memory>
    - <cuda/ptx>


C++ mathematical operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We rely heavily on compiler builtins or cuda runtime functions to implement C++ standard math functions such
as ``cuda::std::exp``. Those compiler builtins are not currently supported in tile mode, so the following header
is mostly unsupported in a tile kernel:

    - <cuda/std/cmath>

``cuda::std::complex`` is supported except for the various math function overloads that are specialized for complex.


C++ customization point objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The standard library uses __ Customization Point Objects __ to enable user-customization of the behavior of many
algorithms and ranges. We rely heavily on those for most of our iterator machinery such as e.g `cuda::std::begin`.

Those CPOs are currently not accessible in a tile kernel. A potential workaround is to construct an instance of the
empty type, e.g. ``decltype(cuda::std::begin){}(container);``

C++ return statements in loops and switches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tile currently does not support return statements inside of a ``switch`` or a loop. We can work around most places but
not all. This mainly affects algorithms and arch traits.

In ``<cuda/std/algorithm>`` the following algorithms are not supported in a tile kernel

    - ``cuda::std::equal_range``
    - ``cuda::std::find_end``
    - ``cuda::std::find_first_of``
    - ``cuda::std::partition``
    - ``cuda::std::search``
    - ``cuda::std::search_n``

Besides that the content of the following headers is unsupported in a tile program

    - ``<cuda/std/devices>``


CUDA device intrinsics
~~~~~~~~~~~~~~~~~~~~~~~

In tile mode the compiler handles threads, warps and blocks. Consequently, the access of CUDA device intrinsics such as
``threadIdx`` is currently not allowed in a tile program. Therefore the following headers are not supported in tile mode:

    - <cuda/access_property>
    - <cuda/annotated_ptr>
    - <cuda/discard_memory>
    - <cuda/hierarchy>
    - <cuda/ptx>


CUDA extended floating point types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tile programs treat the CUDA extended floating point types as compiler builtin types.
This disallows accessing their internals which we require internally.
Support for extended floating point types such as ``__half``, ``__nv_bfloat16`` is only partial in libcu++.
Support for extended floating point types of size 8, 6 and 4-bit is disabled.


Taking the address of a function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is currently not supported to take the address of a function in a tile program.
This affects our memory resource machinery, so the following headers are unsupported in tile mode:

    - <cuda/memory>
    - <cuda/memory_resource>

----

Standard C++ Tile Availability Summary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. |V| unicode:: U+2705
.. |X| unicode:: U+274C
.. |P| unicode:: U+1F7E8

.. table::
    :widths: 25 25 10 40

    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | **Library**                                                                        | ``Libcu++``                                                                          | Supported since | **Notes**                                                                                                 |
    +====================================================================================+======================================================================================+=================+===========================================================================================================+
    | `<algorithm> <https://en.cppreference.com/w/cpp/header/algorithm>`_                | :ref:`<cuda/std/algorithm> <libcudacxx-standard-api-algorithms>`                     |   |P| 3.4       | Partial support. Some algorithms rely on return statements inside of loops                                |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<array> <https://en.cppreference.com/w/cpp/header/array>`_                        | :ref:`<cuda/std/array> <libcudacxx-standard-api-container-array>`                    |   |V| 3.4       |                                                                                                           |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<atomic> <https://en.cppreference.com/w/cpp/header/atomic>`_                      | ``<cuda/std/atomic>``                                                                |   |X|           | Requires ptx                                                                                              |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<barrier> <https://en.cppreference.com/w/cpp/header/barrier>`_                    | ``<cuda/std/barrier>``                                                               |   |X|           | Requires ptx                                                                                              |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<bit> <https://en.cppreference.com/w/cpp/header/bit>`_                            | :ref:`<cuda/std/bit> <libcudacxx-standard-api-numerics-bit>`                         |   |V| 3.4       |                                                                                                           |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<bitset> <https://en.cppreference.com/w/cpp/header/bitset>`_                      | :ref:`<cuda/std/bitset> <libcudacxx-standard-api-utility-bitset>`                    |   |V| 3.4       |                                                                                                           |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<cassert> <https://en.cppreference.com/w/cpp/header/cassert>`_                    | ``<cuda/std/cassert>``                                                               |   |V| 3.4       |                                                                                                           |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<ccomplex> <https://en.cppreference.com/w/cpp/header/ccomplex>`_                  | ``<cuda/std/ccomplex>``                                                              |   |P| 3.4       |                                                                                                           |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<cfloat> <https://en.cppreference.com/w/cpp/header/cfloat>`_                      | ``<cuda/std/cfloat>``                                                                |   |V| 3.4       |                                                                                                           |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<charconv> <https://en.cppreference.com/w/cpp/header/charconv>`_                  | ``<cuda/std/charconv>``                                                              |   |V| 3.4       |                                                                                                           |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<chrono> <https://en.cppreference.com/w/cpp/header/chrono>`_                      | :ref:`<cuda/std/chrono> <libcudacxx-standard-api-time>`                              |   |P| 3.4       | Partial support, requires ptx for clock timers                                                            |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<climits> <https://en.cppreference.com/w/cpp/header/climits>`_                    | ``<cuda/std/climits>``                                                               |   |V| 3.4       |                                                                                                           |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<cmath> <https://en.cppreference.com/w/cpp/header/cmath>`_                        |  ``<cuda/std/cmath>``                                                                |   |X|           | Requires unimplemented compiler builtins for math functions.                                              |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<complex> <https://en.cppreference.com/w/cpp/header/complex>`_                    | :ref:`<cuda/std/complex> <libcudacxx-standard-api-numerics-complex>`                 |   |P| 3.4       | Partial support, requires unimplemented compiler builtins for math functions                              |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<concepts> <https://en.cppreference.com/w/cpp/header/concepts>`_                  | :ref:`<cuda/std/concepts> <libcudacxx-standard-api-concepts>`                        |   |V| 3.4       |                                                                                                           |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<cstddef> <https://en.cppreference.com/w/cpp/header/cstddef>`_                    | ``<cuda/std/cstddef>``                                                               |   |V| 3.4       |                                                                                                           |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<cstdint> <https://en.cppreference.com/w/cpp/header/cstdint>`_                    | ``<cuda/std/cstdint>``                                                               |   |V| 3.4       |                                                                                                           |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<cstdlib> <https://en.cppreference.com/w/cpp/header/cstdlib>`_                    | ``<cuda/std/cstdlib>``                                                               |   |V| 3.4       |                                                                                                           |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<cstring> <https://en.cppreference.com/w/cpp/header/cstring>`_                    | ``<cuda/std/cstring>``                                                               |   |V| 3.4       |                                                                                                           |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<ctime> <https://en.cppreference.com/w/cpp/header/ctime>`_                        | ``<cuda/std/ctime>``                                                                 |   |X|           | Requires ptx for clock timers                                                                             |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<execution> <https://en.cppreference.com/w/cpp/header/execution>`_                | :ref:`<cuda/std/execution> <libcudacxx-standard-api-execution>`                      |   |X|           | Requires taking the address of member functions                                                           |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<expected> <https://en.cppreference.com/w/cpp/header/expected>`_                  | :ref:`<cuda/std/expected> <libcudacxx-standard-api-utility-expected>`                |   |V| 3.4       |                                                                                                           |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<functional> <https://en.cppreference.com/w/cpp/header/functional>`_              | :ref:`<cuda/std/functional> <libcudacxx-standard-api-utility-functional>`            |   |P| 3.4       | Partial support. Tile C++ disallows taking the address of functions, so ``mem_fn`` et al are not available|
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<initializer_list> <https://en.cppreference.com/w/cpp/header/initializer_list>`_  | ``<cuda/std/initializer_list>``                                                      |   |V| 3.4       |                                                                                                           |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<inplace_vector> <https://en.cppreference.com/w/cpp/header/inplace_vector>`_      | :ref:`<cuda/std/inplace_vector> <libcudacxx-standard-api-container-inplace-vector>`  |   |V| 3.4       |                                                                                                           |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<iterator> <https://en.cppreference.com/w/cpp/header/iterator>`_                  | ``<cuda/std/iterator>``                                                              |   |P| 3.4       | Partial support. We rely heavily on CPO's. This affects e.g ``cuda::std::begin``                          |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<latch> <https://en.cppreference.com/w/cpp/header/latch>`_                        | ``<cuda/std/latch>``                                                                 |   |X|           | Requires ptx                                                                                              |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<limits> <https://en.cppreference.com/w/cpp/header/limits>`_                      | ``<cuda/std/limits>``                                                                |   |V| 3.4       |                                                                                                           |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<linalg> <https://en.cppreference.com/w/cpp/header/linalg>`_                      | :ref:`<cuda/std/linalg> <libcudacxx-standard-api-numerics-linalg>`                   |   |V| 3.4       | Accessors, transposed layout, and related functions                                                       |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<mdspan> <https://en.cppreference.com/w/cpp/header/mdspan>`_                      | :ref:`<cuda/std/mdspan> <libcudacxx-standard-api-container-mdspan>`                  |   |V| 3.4       |                                                                                                           |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<memory> <https://en.cppreference.com/w/cpp/header/memory>`_                      | :ref:`<cuda/std/memory> <libcudacxx-standard-api-utility-memory>`                    |   |V| 3.4       | ``cuda::std::addressof``, ``cuda::std::align``, ``cuda::std::assume_aligned``                             |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<numbers> <https://en.cppreference.com/w/cpp/header/numbers>`_                    | :ref:`<cuda/std/numbers> <libcudacxx-standard-api-numerics-numbers>`                 |   |P| 3.4       | Partial support. ``double`` on windows is not supported. Extended floating point types are not supported. |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<numeric> <https://en.cppreference.com/w/cpp/header/numeric>`_                    | :ref:`<cuda/std/numeric> <libcudacxx-standard-api-numerics-numeric>`                 |   |V| 3.4       |                                                                                                           |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<optional> <https://en.cppreference.com/w/cpp/header/optional>`_                  | :ref:`<cuda/std/optional> <libcudacxx-standard-api-utility-optional>`                |   |V| 3.4       |                                                                                                           |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<ranges> <https://en.cppreference.com/w/cpp/header/ranges>`_                      | :ref:`<cuda/std/ranges> <libcudacxx-standard-api-ranges>`                            |   |P| 3.4       | Partial support. We rely heavily on CPO's. This affects e.g ``cuda::std::ranges::begin``                  |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<ratio> <https://en.cppreference.com/w/cpp/header/ratio>`_                        | ``<cuda/std/ratio>``                                                                 |   |V| 3.4       |                                                                                                           |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<random> <https://en.cppreference.com/w/cpp/header/random>`_                      | :ref:`<cuda/std/random> <libcudacxx-standard-api-numerics-random>`                   |   |P| 3.4       | Partial support. ``seed_seq`` relies on dynamic memory allocations, so it is not available                |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<semaphore> <https://en.cppreference.com/w/cpp/header/semaphore>`_                | ``<cuda/std/semaphore>``                                                             |   |X|           | Requires ptx                                                                                              |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<source_location> <https://en.cppreference.com/w/cpp/header/source_location>`_    | ``<cuda/std/source_location>``                                                       |   |X|           | Requires compiler support                                                                                 |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<span> <https://en.cppreference.com/w/cpp/header/span>`_                          | :ref:`<cuda/std/span> <libcudacxx-standard-api-container-span>`                      |   |V| 3.4       |                                                                                                           |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<tuple> <https://en.cppreference.com/w/cpp/header/tuple>`_                        | :ref:`<cuda/std/tuple> <libcudacxx-standard-api-utility-tuple>`                      |   |V| 3.4       |                                                                                                           |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<type_traits> <https://en.cppreference.com/w/cpp/header/type_traits>`_            | :ref:`<cuda/std/type_traits> <libcudacxx-standard-api-utility-type-traits>`          |   |V| 3.4       |                                                                                                           |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<utility> <https://en.cppreference.com/w/cpp/header/utility>`_                    | :ref:`<cuda/std/utility> <libcudacxx-standard-api-utility-utility>`                  |   |V| 3.4       |                                                                                                           |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<variant> <https://en.cppreference.com/w/cpp/header/variant>`_                    | :ref:`<cuda/std/variant> <libcudacxx-standard-api-utility-variant>`                  |   |V| 3.4       |                                                                                                           |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
    | `<version> <https://en.cppreference.com/w/cpp/header/version>`_                    | ``<cuda/std/version>``                                                               |   |V| 3.4       |                                                                                                           |
    +------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------+
