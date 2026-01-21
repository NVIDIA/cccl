.. _libcudacxx-extended-api-numeric:

Numeric
=======

.. toctree::
   :hidden:
   :maxdepth: 1

   numeric/add_overflow
   numeric/narrow
   numeric/overflow_result

.. list-table::
   :widths: 25 45 30 30
   :header-rows: 1

   * - **Header**
     - **Content**
     - **CCCL Availability**
     - **CUDA Toolkit Availability**

   * - :ref:`cuda::narrow <libcudacxx-extended-api-numeric-narrow>`
     - Casts a value and checks whether the value has changed
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`cuda::overflow_result <libcudacxx-extended-api-numeric-overflow_result>`
     - Represents the result of arithmetic operations that may overflow
     - CCCL 3.0.0
     - CUDA 13.0

   * - :ref:`cuda::add_overflow <libcudacxx-extended-api-numeric-add_overflow>`
     - Performs addition with overflow checking
     - CCCL 3.2.0
     - CUDA 13.2
