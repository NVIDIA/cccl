# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest

from cuda.coop._core import (
    ArgumentBinding,
    ArgumentKind,
    CxxOperator,
    Dependency,
    ParameterRole,
    PythonOperator,
    WarpReduceOperation,
    make_warp_reduce_spec,
)


@pytest.mark.parametrize(
    ("operation", "method_name"),
    [("sum", "Sum"), ("min", "Min"), ("max", "Max")],
)
def test_warp_reduce_selects_builtin_entry_point(operation, method_name):
    spec = make_warp_reduce_spec(
        dtype="int",
        threads_in_warp=16,
        operation=operation,
    )

    assert spec.operation is WarpReduceOperation(operation)
    assert spec.method_name == method_name
    assert spec.specialization.template_arguments == {
        "T": "int",
        "VIRTUAL_WARP_THREADS": 16,
    }
    assert [item.name for item in spec.specialization.parameters[0]] == [
        "temp_storage",
        "input",
        "output",
    ]


def test_warp_reduce_custom_operator_and_runtime_prefix_signature():
    def multiply(left, right):
        return left * right

    spec = make_warp_reduce_spec(
        dtype="int",
        threads_in_warp=8,
        operation="reduce",
        reduce_operator=PythonOperator(
            ret_dtype=Dependency("T"),
            arg_dtypes=(Dependency("T"), Dependency("T")),
            op=multiply,
            name="binary_op",
        ),
        valid_items=ArgumentBinding.runtime(),
    )

    assert [
        (item.name, item.kind, item.role)
        for item in spec.specialization.classify_method()
    ] == [
        ("temp_storage", ArgumentKind.RUNTIME, ParameterRole.TEMP_STORAGE),
        ("input", ArgumentKind.RUNTIME, ParameterRole.INPUT),
        ("binary_op", ArgumentKind.STATIC, ParameterRole.OPERATOR),
        ("valid_items", ArgumentKind.RUNTIME, ParameterRole.INPUT),
        ("output", ArgumentKind.RUNTIME, ParameterRole.OUTPUT),
    ]
    assert spec.specialization.parameters[0][-2].dtype.name == "int32"


def test_warp_reduce_canonicalizes_and_bounds_static_prefix():
    numpy = make_warp_reduce_spec(
        dtype="int",
        threads_in_warp=8,
        operation="sum",
        valid_items=ArgumentBinding.static(np.int32(5)),
    )
    plain = make_warp_reduce_spec(
        dtype="int",
        threads_in_warp=8,
        operation="sum",
        valid_items=ArgumentBinding.static(5),
    )

    assert numpy.valid_items == ArgumentBinding.static(5)
    assert numpy.semantic_key == plain.semantic_key
    assert numpy.specialization.parameters[0][-2].cpp == "5"
    with pytest.raises(ValueError, match="exceeds warp size 8"):
        make_warp_reduce_spec(
            dtype="int",
            threads_in_warp=8,
            operation="sum",
            valid_items=ArgumentBinding.static(9),
        )


def test_warp_reduce_validates_operator_and_builtin_prefix_contracts():
    operator = CxxOperator(
        "::cuda::std::plus<>",
        Dependency("T"),
        name="binary_op",
    )

    with pytest.raises(TypeError, match="requires a reduce operator"):
        make_warp_reduce_spec(
            dtype="int",
            threads_in_warp=32,
            operation="reduce",
        )
    with pytest.raises(ValueError, match="does not accept"):
        make_warp_reduce_spec(
            dtype="int",
            threads_in_warp=32,
            operation="sum",
            reduce_operator=operator,
        )
    with pytest.raises(ValueError, match="does not accept valid_items"):
        make_warp_reduce_spec(
            dtype="int",
            threads_in_warp=32,
            operation="min",
            valid_items=True,
        )


@pytest.mark.parametrize("threads_in_warp", [True, 0, 3, 33])
def test_warp_reduce_rejects_invalid_logical_warp_width(threads_in_warp):
    with pytest.raises(ValueError, match="power of two"):
        make_warp_reduce_spec(
            dtype="int",
            threads_in_warp=threads_in_warp,
            operation="sum",
        )
