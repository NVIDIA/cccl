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
    ReduceOperation,
    ReduceValueKind,
    make_block_reduce_semantics,
    make_block_reduce_spec,
)
from cuda.coop._core.block import BlockReduceAlgorithm


def test_block_sum_scalar_signature_and_specialization():
    spec = make_block_reduce_spec(
        dtype="int",
        block_dim=(32, 2, 1),
        items_per_thread=1,
        operation="sum",
        algorithm="warp_reductions",
        value_kind="scalar",
    )

    assert spec.operation is ReduceOperation.SUM
    assert spec.value_kind is ReduceValueKind.SCALAR
    assert spec.method_name == "Sum"
    assert spec.block_dim == (32, 2, 1)
    assert spec.specialization.template_arguments == {
        "T": "int",
        "BLOCK_DIM_X": 32,
        "ALGORITHM": "::cub::BLOCK_REDUCE_WARP_REDUCTIONS",
        "BLOCK_DIM_Y": 2,
        "BLOCK_DIM_Z": 1,
    }
    assert [
        (item.name, item.kind, item.role)
        for item in spec.specialization.classify_method()
    ] == [
        ("temp_storage", ArgumentKind.RUNTIME, ParameterRole.TEMP_STORAGE),
        ("src", ArgumentKind.RUNTIME, ParameterRole.INPUT),
        ("output", ArgumentKind.RUNTIME, ParameterRole.OUTPUT),
    ]
    assert spec.specialization.parameters[0][-1].is_return is True


def test_block_reduce_array_tracks_item_count_and_custom_operator():
    def multiply(left, right):
        return left * right

    spec = make_block_reduce_spec(
        dtype="int",
        block_dim=(64, 1, 1),
        items_per_thread=4,
        operation="reduce",
        algorithm=BlockReduceAlgorithm.RAKING,
        value_kind="array",
        reduce_operator=PythonOperator(
            ret_dtype=Dependency("T"),
            arg_dtypes=(Dependency("T"), Dependency("T")),
            op=multiply,
            name="binary_op",
        ),
    )

    assert spec.specialization.ordered_specialization_arguments[-1] == (
        "ITEMS_PER_THREAD",
        4,
    )
    assert [item.name for item in spec.specialization.parameters[0]] == [
        "temp_storage",
        "src",
        "binary_op",
        "output",
    ]
    operator = spec.specialization.classify_method()[2]
    assert operator.kind is ArgumentKind.STATIC
    assert operator.role is ParameterRole.OPERATOR


def test_block_reduce_prefix_is_i32_and_bounded_by_the_block():
    runtime = make_block_reduce_spec(
        dtype="int",
        block_dim=(64, 1, 1),
        items_per_thread=1,
        operation="sum",
        algorithm="raking",
        value_kind="scalar",
        valid_items=ArgumentBinding.runtime(),
    )
    static = make_block_reduce_spec(
        dtype="int",
        block_dim=(64, 1, 1),
        items_per_thread=1,
        operation="sum",
        algorithm="raking",
        value_kind="scalar",
        valid_items=ArgumentBinding.static(np.int32(17)),
    )

    assert runtime.specialization.parameters[0][-2].dtype.name == "int32"
    assert static.call.valid_items == ArgumentBinding.static(17)
    assert static.specialization.parameters[0][-2].cpp == "17"
    with pytest.raises(ValueError, match="exceeds block size 64"):
        make_block_reduce_spec(
            dtype="int",
            block_dim=(64, 1, 1),
            items_per_thread=1,
            operation="sum",
            algorithm="raking",
            value_kind="scalar",
            valid_items=ArgumentBinding.static(65),
        )


def test_reduce_semantics_validate_payload_and_operator_contracts():
    plus = CxxOperator(
        "::cuda::std::plus<T>",
        Dependency("T"),
        name="binary_op",
    )
    first = make_block_reduce_semantics(
        dtype="int",
        items_per_thread=1,
        operation="reduce",
        value_kind="scalar",
        reduce_operator=plus,
        valid_items=ArgumentBinding.static(np.int32(5)),
    )
    second = make_block_reduce_semantics(
        dtype="int",
        items_per_thread=1,
        operation="reduce",
        value_kind="scalar",
        reduce_operator=plus,
        valid_items=ArgumentBinding.static(5),
    )

    assert first.semantic_key == second.semantic_key
    with pytest.raises(TypeError, match="requires a reduce operator"):
        make_block_reduce_semantics(
            dtype="int",
            items_per_thread=1,
            operation="reduce",
            value_kind="scalar",
        )
    with pytest.raises(ValueError, match="does not accept"):
        make_block_reduce_semantics(
            dtype="int",
            items_per_thread=1,
            operation="sum",
            value_kind="scalar",
            reduce_operator=plus,
        )
    with pytest.raises(ValueError, match="not supported for array"):
        make_block_reduce_semantics(
            dtype="int",
            items_per_thread=2,
            operation="sum",
            value_kind="array",
            valid_items=True,
        )


@pytest.mark.parametrize("items_per_thread", [True, 0, -1, 1.5])
def test_reduce_semantics_reject_invalid_item_count(items_per_thread):
    with pytest.raises(ValueError, match="positive integer"):
        make_block_reduce_semantics(
            dtype="int",
            items_per_thread=items_per_thread,
            operation="sum",
            value_kind="scalar",
        )
