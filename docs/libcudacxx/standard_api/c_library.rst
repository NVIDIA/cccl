.. _libcudacxx-standard-api-c-compat:

C Library
=========

.. toctree::
   :hidden:
   :maxdepth: 1

   c_library/cstring


Any Standard C++ header not listed below is omitted.

.. list-table::
   :widths: 25 45 30 30 20
   :header-rows: 1

   * - **Header**
     - **Content**
     - **CCCL Availability**
     - **CUDA Toolkit Availability**
     - **C++ Reference**

   * - ``<cuda/std/cassert>``
     - Lightweight assumption testing
     - CCCL 2.0.0
     - CUDA 10.2
     - `\<cassert\> <https://en.cppreference.com/w/cpp/header/cassert>`_

   * - ``<cuda/std/ccomplex>``
     - C complex number arithmetic
     - CCCL 2.0.0
     - CUDA 11.4
     - `\<ccomplex\> <https://en.cppreference.com/w/cpp/header/ccomplex>`_

   * - ``<cuda/std/cfloat>``
     - Type support library
     - CCCL 2.2.0
     - CUDA 12.3
     - `\<cfloat\> <https://en.cppreference.com/w/cpp/header/cfloat>`_

   * - ``<cuda/std/cfloat>``
     - Limits of floating point types
     - CCCL 2.0.0
     - CUDA 10.2
     - `\<cfloat\> <https://en.cppreference.com/w/cpp/header/cfloat>`_

   * - ``<cuda/std/climits>``
     - Limits of integral types
     - CCCL 2.0.0
     - CUDA 10.2
     - `\<climits\> <https://en.cppreference.com/w/cpp/header/climits>`_

   * - ``<cuda/std/cmath>``
     - Common math functions
     - CCCL 2.2.0
     - CUDA 12.3
     - `\<cmath\> <https://en.cppreference.com/w/cpp/header/cmath>`_

   * - ``<cuda/std/cstddef>``
     - Fundamental types
     - CCCL 2.0.0
     - CUDA 10.2
     - `\<cstddef\> <https://en.cppreference.com/w/cpp/header/cstddef>`_

   * - ``<cuda/std/cstdint>``
     - Fundamental integer types
     - CCCL 2.2.0
     - CUDA 12.3
     - `\<cstdint\> <https://en.cppreference.com/w/cpp/header/cstdint>`_

   * - ``<cuda/std/cstdint>``
     - Fixed-width integer types
     - CCCL 2.0.0
     - CUDA 10.2
     - `\<cstdint\> <https://en.cppreference.com/w/cpp/header/cstdint>`_

   * - ``<cuda/std/cstdlib>``
     - Common utilities
     - CCCL 2.2.0
     - CUDA 12.3
     - `\<cstdlib\> <https://en.cppreference.com/w/cpp/header/cstdlib>`_

   * - :ref:`\<cuda/std/cstring\> <libcudacxx-standard-api-cstring>`
     - Provides array manipulation functions such as ``memcpy``, ``memset`` and ``memcmp``
     - CCCL 3.0.0
     - CUDA 13.0
     - `\<cstring\> <https://en.cppreference.com/w/cpp/header/cstring>`_
