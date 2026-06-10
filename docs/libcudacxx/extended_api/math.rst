.. _libcudacxx-extended-api-math:

Math
====

.. toctree::
   :hidden:
   :maxdepth: 1

   math/ceil_div
   math/round_up
   math/round_down
   math/ilog
   math/ipow
   math/pow2
   math/isqrt
   math/neg
   math/uabs
   math/fast_mod_div
   math/mul_hi
   math/sincos

.. list-table::
   :widths: 25 45 30 30
   :header-rows: 1

   * - **Header**
     - **Content**
     - **CCCL Availability**
     - **CUDA Toolkit Availability**

   * - :ref:`ceil_div <libcudacxx-extended-api-math-ceil-div>`
     - Ceiling division
     - CCCL 2.7.0
     - CUDA 12.8

   * - :ref:`round_up <libcudacxx-extended-api-math-round-up>`
     - Round up to the next multiple
     - CCCL 3.0.0
     - CUDA 13.0

   * - :ref:`round_down <libcudacxx-extended-api-math-round-down>`
     - Round down to the previous multiple
     - CCCL 3.0.0
     - CUDA 13.0

   * - :ref:`ilog2 <libcudacxx-extended-api-math-ilog>`
     - Integer logarithm to the base 2
     - CCCL 3.0.0
     - CUDA 13.0

   * - :ref:`ilog10 <libcudacxx-extended-api-math-ilog>`
     - Integer logarithm to the base 10
     - CCCL 3.0.0
     - CUDA 13.0

   * - :ref:`ipow <libcudacxx-extended-api-math-ipow>`
     - Integer power
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`is_power_of_two <libcudacxx-extended-api-math-pow2>`
     - If the value is a power of two
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`isqrt <libcudacxx-extended-api-math-isqrt>`
     - Integer square root
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`neg <libcudacxx-extended-api-math-neg>`
     - Integer negation
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`next_power_of_two <libcudacxx-extended-api-math-pow2>`
     - Next power of two
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`prev_power_of_two <libcudacxx-extended-api-math-pow2>`
     - Previous power of two
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`uabs <libcudacxx-extended-api-math-uabs>`
     - Unsigned absolute value
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`fast_mod_div <libcudacxx-extended-api-math-fast-mod-div>`
     - Fast Modulo/Division
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`mul_hi <libcudacxx-extended-api-math-mul-hi>`
     - Most significant half of the product
     - CCCL 3.2.0
     - CUDA 13.2

   * - :ref:`sincos <libcudacxx-extended-api-math-sincos>`
     - Computes sine and cosine of a value at the same time.
     - CCCL 3.3.0
     - CUDA 13.3
