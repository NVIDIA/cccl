.. _libcudacxx-standard-api-numerics-bit:

``<cuda/std/bit>``
==================

CUDA Performance Considerations
-------------------------------

Given an unsigned integer with ``N`` bits and ``N <= 32``, the ``<bit>`` functions translate into the following SASS instructions. For some functions, the results is decorated with a compile-time assumption to restrict its range and allowing further optimizations.

- ``bit_width()`` translates into a single ``FLO`` SASS instruction. The result is assumed to be in the range ``[0, N]``.
- ``bit_ceil()`` translates into ``ADD, FLO, SHL, IMINMAX`` SASS instructions. The result is assumed to be greater than or equal to the input.
- ``bit_floor()`` translates into ``FLO, SHL`` SASS instructions. The result is assumed to be less than or equal to the input.
- ``byteswap()`` translates into a single ``PRMT`` SASS instruction.
- ``popcount()`` translates into a single ``POPC`` SASS instruction. The result is assumed to be in the range ``[0, N]``.
- ``has_single_bit()`` translates into ``POPC + ISETP`` SASS instructions.
- ``rotl()/rotr()`` translate into a single ``SHF`` (funned shift) SASS instruction.
- ``countl_zero()`` translates into ``FLO, IMINMAX`` SASS instructions. The result is assumed to be in the range ``[0, N]``.
- ``countl_one()`` translates into ``LOP3, FLO, IMINMAX`` SASS instructions. The result is assumed to be in the range ``[0, N]``.
- ``countr_zero()`` translates into ``BREV, FLO, IMINMAX`` SASS instructions. The result is assumed to be in the range ``[0, N]``.
- ``countr_one()`` translates into ``LOP3, BREV, FLO, IMINMAX`` SASS instructions. The result is assumed to be in the range ``[0, N]``.

Additional Notes
----------------

- All functions are marked ``[[nodiscard]]`` and ``noexcept``
- All functions support 128-bit integer types
- ``bit_ceil()`` checks for overflow in debug mode
- ``rotl()/rotr()`` checks for invalid count value (``INT_MIN``) in debug mode

.. note::

    When the input values are run-time values that the compiler can resolve at compile-time, e.g. an index of a loop with a fixed number of iterations, using the functions could not be optimal.

.. note::

    GCC <= 8 uses a slow path with more instructions even in CUDA
