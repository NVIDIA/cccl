.. _libcudacxx-standard-api-container:

Container Library
=================

.. toctree::
   :hidden:
   :maxdepth: 1

   container_library/array
   container_library/inplace_vector
   container_library/mdspan
   container_library/span

Any Standard C++ header not listed below is omitted.

.. list-table::
   :widths: 25 45 30 30 20
   :header-rows: 1

   * - **Header**
     - **Content**
     - **CCCL Availability**
     - **CUDA Toolkit Availability**
     - **C++ Reference**

   * - :ref:`\<cuda/std/array\> <libcudacxx-standard-api-container-array>`
     - Fixed size array
     - CCCL 2.0.0
     - CUDA 11.7
     - `\<array\> <https://en.cppreference.com/w/cpp/header/array>`_

   * - :ref:`\<cuda/std/inplace_vector\> <libcudacxx-standard-api-container-inplace-vector>`
     - Flexible size container with fixed capacity
     - CCCL 2.6.0
     - CUDA 12.8
     - `\<inplace_vector\> <https://en.cppreference.com/w/cpp/header/inplace_vector>`_

   * - :ref:`\<cuda/std/mdspan\> <libcudacxx-standard-api-container-mdspan>`
     - Non - owning view into a multidimensional contiguous sequence of objects
     - CCCL 2.1.0
     - CUDA 12.2
     - `\<mdspan\> <https://en.cppreference.com/w/cpp/header/mdspan>`_

   * - :ref:`\<cuda/std/span\> <libcudacxx-standard-api-container-span>`
     - Non - owning view into a contiguous sequence of objects
     - CCCL 2.1.0
     - CUDA 12.2
     - `\<span\> <https://en.cppreference.com/w/cpp/header/span>`_
