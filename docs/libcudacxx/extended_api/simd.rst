.. _libcudacxx-extended-api-simd:

SIMD
====

.. toctree::
   :hidden:
   :maxdepth: 1

   simd/dot
   simd/saturating_add
   simd/abs_diff

.. list-table::
   :widths: 25 45 30 30
   :header-rows: 1

   * - **Header**
     - **Content**
     - **CCCL Availability**
     - **CUDA Toolkit Availability**

   * - :ref:`cuda::simd::saturating_add <libcudacxx-extended-api-simd-saturating-add>`
     - Performs element-wise saturating addition of two ``basic_vec`` objects
     - CCCL 3.5.0
     - CUDA 13.5

   * - :ref:`cuda::simd::dot <libcudacxx-extended-api-simd-dot>`
     - Computes the dot product of two ``basic_vec`` objects and an accumulator
     - CCCL 3.6.0
     - CUDA 13.6

   * - :ref:`cuda::simd::abs_diff <libcudacxx-extended-api-simd-abs-diff>`
     - Performs element-wise absolute difference of two integer ``basic_vec`` objects
     - CCCL 3.6.0
     - CUDA 13.6
