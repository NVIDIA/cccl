# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest


class _DynamicInt:
    """Stand-in for a runtime DSL integer operand."""


def test_radix_bit_range_defaults_preserve_runtime_operands():
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Int32, Uint64

    from cuda.coop.cutlass._dsl._provider import validate_radix_bit_range

    begin_bit = _DynamicInt()
    end_bit = _DynamicInt()

    assert validate_radix_bit_range(0, None, Int32) == 32
    assert validate_radix_bit_range(0, None, Uint64) == 64
    assert validate_radix_bit_range(3, 13, Int32) == 13
    assert validate_radix_bit_range(begin_bit, end_bit, Int32) is end_bit
    assert validate_radix_bit_range(begin_bit, None, Int32) == 32


@pytest.mark.parametrize(
    ("begin_bit", "end_bit", "message"),
    [
        pytest.param(32, None, "begin_bit must be < 32", id="begin-at-width"),
        pytest.param(0, 33, "end_bit must be <= 32", id="end-past-width"),
        pytest.param(16, 16, "end_bit must be greater", id="empty-range"),
    ],
)
def test_radix_bit_range_rejects_invalid_static_bounds(
    begin_bit,
    end_bit,
    message,
):
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Int32

    from cuda.coop.cutlass._dsl._provider import validate_radix_bit_range

    with pytest.raises(ValueError, match=message):
        validate_radix_bit_range(begin_bit, end_bit, Int32)
