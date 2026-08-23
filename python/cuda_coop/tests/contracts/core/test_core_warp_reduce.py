# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402


import pytest

from cuda.coop._core import (
    ArgumentBinding,
    ArgumentKind,
    CxxOperator,
    Dependency,
    ParameterRole,
    PythonOperator,
    lower_method_parameters,
)
from cuda.coop._core.block import BlockReduceAlgorithm, make_block_reduce_spec
from cuda.coop._core.warp import WarpReduceOperation, make_warp_reduce_spec


@pytest.mark.parametrize(
    ("operation", "method_name"),
    [
        ("sum", "Sum"),
        ("min", "Min"),
        ("max", "Max"),
    ],
)
def test_warp_reduce_selects_builtin_entry_point(operation, method_name):
    spec = make_warp_reduce_spec(
        dtype="int",
        threads_in_warp=16,
        operation=operation,
    )

    assert spec.operation is WarpReduceOperation(operation)
    assert spec.method_name == method_name
    assert spec.threads_in_warp == 16
    assert spec.specialization.template_arguments == {
        "T": "int",
        "VIRTUAL_WARP_THREADS": 16,
    }
    assert [item.name for item in spec.specialization.parameters[0]] == [
        "temp_storage",
        "input",
        "output",
    ]
    if operation in {"min", "max"}:
        assert spec.call.reduce_operator.cpp == f"::cuda::{operation}imum<>"
        assert f"{spec.call.reduce_operator.cpp}{{}}" == (
            f"::cuda::{operation}imum<>{{}}"
        )


@pytest.mark.parametrize(
    ("operation", "materialized"),
    [
        ("min", "::cuda::minimum<>{}"),
        ("max", "::cuda::maximum<>{}"),
    ],
)
def test_warp_builtin_semantics_materialize_once_when_reused(operation, materialized):
    class BraceAppendingAdapter:
        def lower_parameter(self, parameter, *, specialization):
            del specialization
            return parameter

        def lower_cxx_operator(self, operator, *, specialization):
            del specialization
            return f"{operator.cpp}{{}}"

    warp = make_warp_reduce_spec(
        dtype="int",
        threads_in_warp=32,
        operation=operation,
    )
    block = make_block_reduce_spec(
        dtype="int",
        block_dim=(64, 1, 1),
        items_per_thread=1,
        operation=warp.call.operation,
        algorithm=BlockReduceAlgorithm.WARP_REDUCTIONS,
        value_kind=warp.call.value_kind,
        reduce_operator=warp.call.reduce_operator,
    )

    lowered = lower_method_parameters(
        BraceAppendingAdapter(),
        block.specialization,
        block.specialization.parameters[0],
        include_temp_storage=False,
    )

    assert materialized in lowered


def test_warp_reduce_custom_operator_and_partial_signature():
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
            name="reduce_op",
        ),
        valid_items=True,
    )

    assert spec.method_name == "Reduce"
    assert spec.has_valid_items
    assert spec.specialization.parameters[0][3].is_return is True
    assert [
        (item.name, item.kind, item.role)
        for item in spec.specialization.classify_method()
    ] == [
        ("temp_storage", ArgumentKind.RUNTIME, ParameterRole.TEMP_STORAGE),
        ("input", ArgumentKind.RUNTIME, ParameterRole.INPUT),
        ("reduce_op", ArgumentKind.STATIC, ParameterRole.OPERATOR),
        ("output", ArgumentKind.RUNTIME, ParameterRole.OUTPUT),
        ("valid_items", ArgumentKind.RUNTIME, ParameterRole.INPUT),
    ]


def test_warp_reduce_semantic_identity_includes_operator_width_and_partial_mode():
    def add(left, right):
        return left + right

    def multiply(left, right):
        return left * right

    def make(op, *, threads=16, valid=False):
        return make_warp_reduce_spec(
            dtype="int",
            threads_in_warp=threads,
            operation="reduce",
            reduce_operator=PythonOperator(
                Dependency("T"),
                (Dependency("T"), Dependency("T")),
                op,
            ),
            valid_items=valid,
        )

    assert make(add).semantic_key == make(add).semantic_key
    assert make(add).semantic_key != make(multiply).semantic_key
    assert make(add).semantic_key != make(add, threads=8).semantic_key
    assert make(add).semantic_key != make(add, valid=True).semantic_key


def test_warp_reduce_can_retain_full_signature_with_partial_support():
    spec = make_warp_reduce_spec(
        dtype="int",
        threads_in_warp=16,
        operation="sum",
        valid_items=True,
        include_full_warp=True,
    )

    assert spec.has_full_warp
    assert spec.has_valid_items
    assert [
        [parameter.name for parameter in method]
        for method in spec.specialization.parameters
    ] == [
        ["temp_storage", "input", "output"],
        ["temp_storage", "input", "output", "valid_items"],
    ]


def test_warp_reduce_rejects_redundant_full_signature_request():
    with pytest.raises(ValueError, match="requires a valid_items"):
        make_warp_reduce_spec(
            dtype="int",
            threads_in_warp=16,
            operation="sum",
            include_full_warp=True,
        )


def test_warp_reduce_rejects_static_valid_items_larger_than_logical_warp():
    with pytest.raises(ValueError, match="exceeds warp size 16"):
        make_warp_reduce_spec(
            dtype="int",
            threads_in_warp=16,
            operation="sum",
            valid_items=ArgumentBinding.static(17),
        )


def test_warp_reduce_validates_operator_contract():
    operator = CxxOperator(
        "::cuda::std::plus<>",
        Dependency("T"),
        name="reduce_op",
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


@pytest.mark.parametrize("threads_in_warp", [True, 0, 3, 33])
def test_warp_reduce_rejects_invalid_logical_warp_width(threads_in_warp):
    with pytest.raises(ValueError, match="power of two"):
        make_warp_reduce_spec(
            dtype="int",
            threads_in_warp=threads_in_warp,
            operation="sum",
        )
