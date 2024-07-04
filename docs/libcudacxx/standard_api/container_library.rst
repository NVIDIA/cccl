.. _libcudacxx-standard-api-container:

Container Library
=======================

.. toctree::
   :hidden:
   :maxdepth: 1

   container_library/array
   container_library/inplace_vector
   container_library/mdspan
   container_library/span

Any Standard C++ header not listed below is omitted.

.. list-table::
   :widths: 25 45 30
   :header-rows: 1

   * - Header
     - Content
     - Availability
   * - `\<cuda/std/array\> <https://en.cppreference.com/w/cpp/header/array>`_
     - Fixed size array
     - libcu++ 1.8.0 / CCCL 2.0.0 / CUDA 11.7
   * - `\<cuda/std/inplace_vector\> <https://en.cppreference.com/w/cpp/header/inplace_vector>`_
     - Flexible size container with fixed capacity
     - libcu++ 2.6.0 / CCCL 2.6.0
   * - `\<cuda/std/span\> <https://en.cppreference.com/w/cpp/header/span>`_
     - Non - owning view into a contiguous sequence of objects
     - libcu++ 2.1.0 / CCCL 2.1.0 / CUDA 12.2
   * - `\<cuda/std/mdspan\> <https://en.cppreference.com/w/cpp/header/mdspan>`_
     - Non - owning view into a multidimensional contiguous sequence of objects
     - libcu++ 2.1.0 / CCCL 2.1.0 / CUDA 12.2
