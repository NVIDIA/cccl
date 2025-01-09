.. _libcudacxx-standard-api-numerics-bit:

``<cuda/std/bit>``
======================

CUDA Performance Considerations
-------------------------------

- ``bit_width()`` translates into a single ``FLO`` SASS instruction. The result is assumed to be in the range ``[0, N-bit]``.
- ``bit_ceil()`` translates into ``FLO, SHL`` SASS instructions. The result is assumed to be greater than or equal to the input.
- ``bit_floor()`` translates into ``ADD, FLO, SHL, IMINMAX`` SASS instructions. The result is assumed to be less than or equal to the input.
- ``popcount()`` translates into a single ``POPC`` SASS instruction. The result is assumed to be in the range ``[0, N-bit]``.
- ``has_single_bit()`` translates into ``POPC + ISETP`` SASS instructions.
- ``rotl()/rotr()`` translate into a single ``SHF`` (funned shift) SASS instruction.
- ``countl_zero()`` translates into ``FLO, IMINMAX`` SASS instructions. The result is assumed to be in the range ``[0, N-bit]``.
- ``countl_one()`` translates into ``LOP3, FLO, IMINMAX`` SASS instructions. The result is assumed to be in the range ``[0, N-bit]``.
- ``countr_zero()`` translates into ``BREV, FLO, IMINMAX`` SASS instructions. The result is assumed to be in the range ``[0, N-bit]``.
- ``countr_one()`` translates into ``LOP3, BREV, FLO, IMINMAX`` SASS instructions. The result is assumed to be in the range ``[0, N-bit]``.

Additional Notes
----------------

- All functions are marked ``[[nodiscard]]`` and ``noexcept``
- All functions support ``__uint128_t``
- ``bit_ceil()`` checks for overflow in debug mode
