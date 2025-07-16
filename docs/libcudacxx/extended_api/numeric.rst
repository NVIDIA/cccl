.. _libcudacxx-extended-api-numeric:

Numeric
=======

.. toctree::
   :hidden:
   :maxdepth: 1

   numeric/add_overflow
   numeric/div_overflow
   numeric/mul_overflow
   numeric/narrow
   numeric/overflow_cast
   numeric/overflow_result
   numeric/sub_overflow


.. list-table::
   :widths: 25 45 30 30
   :header-rows: 1

   * - **Header**
     - **Content**
     - **CCCL Availability**
     - **CUDA Toolkit Availability**

   * - :ref:`cuda::overflow_cast <libcudacxx-extended-api-numeric-overflow_cast>`
     - Casts a value with overflow checking
     - CCCL 3.0.0
     - CUDA 13.0

   * - :ref:`cuda::narrow <libcudacxx-extended-api-numeric-narrow>`
     - Casts a value and checks whether the value has changed
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`cuda::add_overflow <libcudacxx-extended-api-numeric-add_overflow>`
     - Performs addition with overflow checking
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`cuda::sub_overflow <libcudacxx-extended-api-numeric-sub_overflow>`
     - Performs subtraction with overflow checking
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`cuda::mul_overflow <libcudacxx-extended-api-numeric-mul_overflow>`
     - Performs multiplication with overflow checking
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`cuda::div_overflow <libcudacxx-extended-api-numeric-div_overflow>`
     - Performs division with overflow checking
     - CCCL 3.1.0
     - CUDA 13.1
