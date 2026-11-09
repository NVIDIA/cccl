# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest

from cuda.coop._core import (
    INT32,
    CxxFunction,
    RuntimeValue,
    Value,
    binding,
    i32_parameter,
)


def test_i32_binding_selects_omitted_constant_or_runtime_parameter():
    assert i32_parameter(binding(None), name="value") is None
    assert i32_parameter(
        binding(None), name="value", omitted_value=3
    ) == CxxFunction("3", dtype=INT32, name="value")
    assert i32_parameter(binding(np.int32(5)), name="value") == CxxFunction(
        "5", dtype=INT32, name="value"
    )
    assert i32_parameter(
        binding(RuntimeValue("payload")), name="value"
    ) == Value(INT32, name="value")


@pytest.mark.parametrize("value", [-(1 << 31), (1 << 31) - 1])
def test_i32_binding_accepts_signed_boundaries(value):
    assert i32_parameter(binding(value), name="value") == CxxFunction(
        str(value), dtype=INT32, name="value"
    )


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        (True, TypeError, "must be an integer"),
        (1.5, TypeError, "must be an integer"),
        (1 << 31, ValueError, "fit a signed 32-bit integer"),
    ],
)
def test_i32_binding_rejects_unrepresentable_constants(value, error, message):
    with pytest.raises(error, match=message):
        i32_parameter(binding(value), name="value")


@pytest.mark.parametrize(("left", "right"), [(True, 1), (0.0, -0.0)])
def test_static_bindings_do_not_alias_in_caches(left, right):
    assert len({binding(left), binding(right)}) == 2
