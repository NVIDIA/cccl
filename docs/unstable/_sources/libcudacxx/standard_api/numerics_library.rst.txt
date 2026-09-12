.. _libcudacxx-standard-api-numerics:

Numerics Library
================

.. toctree::
   :hidden:
   :maxdepth: 1

   numerics_library/bit
   numerics_library/complex
   numerics_library/linalg
   numerics_library/numbers
   numerics_library/numeric
   numerics_library/random

Any Standard C++ header not listed below is omitted.

.. list-table::
   :widths: 25 45 30 30 20
   :header-rows: 1

   * - **Header**
     - **Content**
     - **CCCL Availability**
     - **CUDA Toolkit Availability**
     - **C++ Reference**

   * - ``<cuda/std/ratio>``
     - Compile-time rational arithmetic
     - CCCL 2.0.0
     - CUDA 11.0
     - `\<ratio\> <https://en.cppreference.com/w/cpp/header/ratio>`_

   * - :ref:`\<cuda/std/bit\> <libcudacxx-standard-api-numerics-bit>`
     - Access, manipulate, and process individual bits and bit sequences.
     - CCCL 2.0.0
     - CUDA 11.7
     - `\<bit\> <https://en.cppreference.com/w/cpp/header/bit>`_

   * - :ref:`\<cuda/std/complex\> <libcudacxx-standard-api-numerics-complex>`
     - Complex number type
     - CCCL 2.0.0
     - CUDA 11.4
     - `\<complex\> <https://en.cppreference.com/w/cpp/header/complex>`_

   * - :ref:`\<cuda/std/linalg\> <libcudacxx-standard-api-numerics-linalg>`
     - Linear algebra layouts and accessors
     - CCCL 3.0.0
     - CUDA 13.0
     - `\<linalg\> <https://en.cppreference.com/w/cpp/header/linalg>`_

   * - :ref:`\<cuda/std/numbers\> <libcudacxx-standard-api-numerics-numbers>`
     - Numeric constants
     - CCCL 3.0.0
     - CUDA 13.0
     - `\<numbers\> <https://en.cppreference.com/w/cpp/header/numbers>`_

   * - :ref:`\<cuda/std/numeric\> <libcudacxx-standard-api-numerics-numeric>`
     - Numeric algorithms
     - CCCL 2.5.0
     - CUDA 12.6
     - `\<numeric\> <https://en.cppreference.com/w/cpp/header/numeric>`_

   * - :ref:`\<cuda/std/random\> <libcudacxx-standard-api-numerics-random>`
     - Random number generation
     - CCCL 3.3.0
     - CUDA 13.3
     - `\<random\> <https://en.cppreference.com/w/cpp/header/random>`_
