.. _libcudacxx-tile-api:

CUDA Tile support
==================

`CUDA Tile <https://developer.nvidia.com/cuda/tile#section-more-resources>`_ introduces a new way to program GPUs at a higher level than SIMT.

We generally support most features in tile mode such as

    - ``cuda::std::array``
    - ``cuda::std::expected``
    - ``cuda::std::initializer_list``
    - ``cuda::std::optional``
    - ``cuda::std::pair``
    - ``cuda::std::span``
    - ``cuda::std::tuple``
    - ``cuda::std::variant``

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
as ``cuda::std::exp``. Those compiler builtins are not currently supported in tile mode, so the following headers
are mostly unsupported in a tile kernel:

    - <cuda/std/cmath>
    - <cuda/std/complex>


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
Support for extended floating point types such as ``__half``, ``__nv_bfloat16`` is disabled in tile mode.


Taking the address of a function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is currently not supported to take the address of a function in a tile program.
This affects our memory resource machinery, so the following headers are unsupported in tile mode:

    - <cuda/memory>
    - <cuda/memory_resource>
