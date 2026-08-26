# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402


import numpy as np
import pytest

from cuda.coop._core import (
    ArgumentBinding,
    ArgumentKind,
    CxxOperator,
    Dependency,
    ParameterRole,
    PythonOperator,
)
from cuda.coop._core.block import (
    BlockReduceAlgorithm,
    BlockReduceOperation,
    BlockReduceValueKind,
    make_block_reduce_semantics,
    make_block_reduce_spec,
)


def test_block_sum_scalar_signature_and_specialization():
    spec = make_block_reduce_spec(
        dtype="int",
        block_dim=(32, 2, 1),
        items_per_thread=1,
        operation="sum",
        algorithm="::cub::BLOCK_REDUCE_WARP_REDUCTIONS",
        value_kind="scalar",
    )

    assert spec.operation is BlockReduceOperation.SUM
    assert spec.value_kind is BlockReduceValueKind.SCALAR
    assert spec.method_name == "Sum"
    assert spec.block_dim == (32, 2, 1)
    assert not spec.specialization.fake_return
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
    assert spec.specialization.symbol_mangling_inputs[:2] == (
        "block_reduce",
        "Sum",
    )


def test_block_reduce_array_tracks_item_count_and_operator():
    def multiply(left, right):
        return left * right

    spec = make_block_reduce_spec(
        dtype="int",
        block_dim=(64, 1, 1),
        items_per_thread=4,
        operation="reduce",
        algorithm="::cub::BLOCK_REDUCE_RAKING",
        value_kind="array",
        reduce_operator=PythonOperator(
            ret_dtype=Dependency("T"),
            arg_dtypes=(Dependency("T"), Dependency("T")),
            op=multiply,
            name="binary_op",
        ),
    )

    assert spec.method_name == "Reduce"
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
    assert spec.specialization.symbol_mangling_inputs[:2] == (
        "block_reduce",
        "Reduce",
    )


def test_block_reduce_partial_scalar_preserves_cub_argument_order():
    spec = make_block_reduce_spec(
        dtype="int",
        block_dim=(128, 1, 1),
        items_per_thread=1,
        operation="reduce",
        algorithm="::cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY",
        value_kind="scalar",
        reduce_operator=CxxOperator(
            "::cuda::maximum<T>",
            Dependency("T"),
            name="binary_op",
        ),
        valid_items=True,
    )

    assert spec.has_valid_items
    assert [item.name for item in spec.specialization.parameters[0]] == [
        "temp_storage",
        "src",
        "binary_op",
        "num_valid",
        "output",
    ]


def test_block_reduce_call_semantics_support_dynamic_provider_translation():
    call = make_block_reduce_semantics(
        dtype="int",
        items_per_thread=1,
        operation="reduce",
        value_kind="scalar",
        reduce_operator=CxxOperator(
            "::cuda::std::bit_xor<T>",
            Dependency("T"),
            name="binary_op",
        ),
        valid_items=True,
    )

    assert call.method_name == "Reduce"
    assert call.has_valid_items
    assert call.valid_items.argument_kind is ArgumentKind.RUNTIME
    assert call.semantic_key[0] == "reduce"
    assert not hasattr(call, "parameters")
    assert (
        call.semantic_key
        == make_block_reduce_semantics(
            dtype="int",
            items_per_thread=1,
            operation="reduce",
            value_kind="scalar",
            reduce_operator=CxxOperator(
                "::cuda::std::bit_xor<T>",
                Dependency("T"),
                name="binary_op",
            ),
            valid_items=True,
        ).semantic_key
    )


def test_block_reduce_semantic_identity_tracks_shape_algorithm_and_operator():
    def add(left, right):
        return left + right

    def multiply(left, right):
        return left * right

    def make(
        op,
        *,
        block=(32, 1, 1),
        algorithm=BlockReduceAlgorithm.WARP_REDUCTIONS,
        valid=False,
    ):
        return make_block_reduce_spec(
            dtype="int",
            block_dim=block,
            items_per_thread=1,
            operation="reduce",
            algorithm=algorithm,
            value_kind="scalar",
            reduce_operator=PythonOperator(
                Dependency("T"),
                (Dependency("T"), Dependency("T")),
                op,
            ),
            valid_items=valid,
        )

    assert make(add).semantic_key == make(add).semantic_key
    assert make(add).semantic_key != make(multiply).semantic_key
    assert make(add).semantic_key != make(add, block=(64, 1, 1)).semantic_key
    assert (
        make(add).semantic_key
        != make(
            add,
            algorithm=BlockReduceAlgorithm.RAKING,
        ).semantic_key
    )
    assert make(add).semantic_key != make(add, valid=True).semantic_key


def test_block_reduce_validates_operator_and_value_form_contracts():
    operator = CxxOperator(
        "::cuda::std::plus<T>",
        Dependency("T"),
        name="binary_op",
    )

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
            reduce_operator=operator,
        )
    with pytest.raises(ValueError, match="not supported for array"):
        make_block_reduce_semantics(
            dtype="int",
            items_per_thread=2,
            operation="sum",
            value_kind="array",
            valid_items=True,
        )
    with pytest.raises(ValueError, match="requires items_per_thread == 1"):
        make_block_reduce_semantics(
            dtype="int",
            items_per_thread=2,
            operation="sum",
            value_kind="scalar",
        )


@pytest.mark.parametrize("items_per_thread", [True, 0, -1, 1.5])
def test_block_reduce_rejects_invalid_item_count(items_per_thread):
    with pytest.raises(ValueError, match="positive integer"):
        make_block_reduce_semantics(
            dtype="int",
            items_per_thread=items_per_thread,
            operation="sum",
            value_kind="scalar",
        )


def test_block_reduce_rejects_static_valid_items_larger_than_block():
    with pytest.raises(ValueError, match="exceeds block size 32"):
        make_block_reduce_spec(
            dtype="int",
            block_dim=(8, 4, 1),
            items_per_thread=1,
            operation="sum",
            algorithm=BlockReduceAlgorithm.WARP_REDUCTIONS,
            value_kind="scalar",
            valid_items=ArgumentBinding.static(33),
        )


def test_block_reduce_canonicalizes_static_valid_items_identity():
    def build(valid_items):
        return make_block_reduce_spec(
            dtype="int",
            block_dim=(32, 1, 1),
            items_per_thread=1,
            operation="sum",
            algorithm=BlockReduceAlgorithm.WARP_REDUCTIONS,
            value_kind="scalar",
            valid_items=ArgumentBinding.static(valid_items),
        )

    plain = build(5)
    numpy = build(np.int32(5))

    assert plain.call.semantic_key == numpy.call.semantic_key
    assert plain.semantic_key == numpy.semantic_key
    assert numpy.call.valid_items == ArgumentBinding.static(5)


@pytest.mark.parametrize(
    "block_dim",
    [
        (True, 1, 1),
        (1.5, 1, 1),
        "abc",
    ],
)
def test_block_reduce_rejects_non_integral_block_dimensions(block_dim):
    with pytest.raises(ValueError, match="three positive dimensions"):
        make_block_reduce_spec(
            dtype="int",
            block_dim=block_dim,
            items_per_thread=1,
            operation="sum",
            algorithm=BlockReduceAlgorithm.WARP_REDUCTIONS,
            value_kind="scalar",
        )
