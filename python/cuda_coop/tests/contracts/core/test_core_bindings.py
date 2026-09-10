# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402


import numpy as np
import pytest

from cuda.coop._core import (
    INT32,
    ArgumentBinding,
    ArgumentKind,
    BindingKind,
    CxxFunction,
    RuntimeValue,
    Value,
    binding,
    i32_parameter,
)
from cuda.coop._core.block import ArgumentBinding as BlockArgumentBinding


def test_argument_binding_classifies_omitted_static_and_runtime_values():
    omitted = binding(None)
    static = binding(7)
    runtime = binding(RuntimeValue("payload"))

    assert omitted.kind is BindingKind.OMITTED
    assert omitted.argument_kind is None
    assert static == ArgumentBinding.static(7)
    assert static.argument_kind is ArgumentKind.STATIC
    assert runtime == ArgumentBinding.runtime()
    assert runtime.argument_kind is ArgumentKind.RUNTIME
    assert BlockArgumentBinding is ArgumentBinding


def test_i32_parameter_materializes_each_binding_kind():
    assert i32_parameter(ArgumentBinding.omitted(), name="value") is None
    assert i32_parameter(
        ArgumentBinding.omitted(), name="value", omitted_value=3
    ) == CxxFunction("3", dtype=INT32, name="value")
    assert i32_parameter(ArgumentBinding.static(5), name="value") == CxxFunction(
        "5", dtype=INT32, name="value"
    )
    assert i32_parameter(ArgumentBinding.runtime(), name="value") == Value(
        INT32, name="value"
    )


@pytest.mark.parametrize("value", [True, 1.5, "4"])
def test_i32_parameter_rejects_non_integral_static_values(value):
    with pytest.raises(TypeError, match="static value must be an integer"):
        i32_parameter(ArgumentBinding.static(value), name="value")


def test_i32_parameter_normalizes_numpy_integers():
    assert i32_parameter(
        ArgumentBinding.static(np.int32(4)),
        name="value",
    ) == CxxFunction("4", dtype=INT32, name="value")


@pytest.mark.parametrize("value", [-(1 << 31), (1 << 31) - 1])
def test_i32_parameter_accepts_signed_i32_boundaries(value):
    expected = CxxFunction(str(value), dtype=INT32, name="value")

    assert i32_parameter(ArgumentBinding.static(value), name="value") == expected
    assert (
        i32_parameter(
            ArgumentBinding.omitted(),
            name="value",
            omitted_value=np.int64(value),
        )
        == expected
    )


@pytest.mark.parametrize("value", [-(1 << 31) - 1, 1 << 31])
@pytest.mark.parametrize("binding_kind", ["static", "omitted"])
def test_i32_parameter_rejects_values_outside_signed_i32(value, binding_kind):
    with pytest.raises(ValueError, match="fit a signed 32-bit integer"):
        if binding_kind == "static":
            i32_parameter(ArgumentBinding.static(value), name="value")
        else:
            i32_parameter(
                ArgumentBinding.omitted(),
                name="value",
                omitted_value=value,
            )


def test_non_static_binding_rejects_payload_data():
    with pytest.raises(ValueError, match="only static"):
        ArgumentBinding(BindingKind.RUNTIME, 7)


@pytest.mark.parametrize(
    ("left", "right"),
    (
        (True, 1),
        (1, np.int32(1)),
        (0, 0.0),
        (0.0, -0.0),
    ),
)
def test_static_binding_identity_preserves_type_and_representation(left, right):
    left_binding = ArgumentBinding.static(left)
    right_binding = ArgumentBinding.static(right)

    assert left_binding != right_binding
    assert left_binding.semantic_key != right_binding.semantic_key
