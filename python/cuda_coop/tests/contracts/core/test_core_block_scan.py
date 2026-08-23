# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402


import pytest

from cuda.coop._core import (
    ArgumentKind,
    CxxFunction,
    CxxOperator,
    Dependency,
    ParameterRole,
    PythonOperator,
    StatefulOperator,
)
from cuda.coop._core.block import ScanValueKind, make_block_scan_spec


def test_block_scan_sum_array_signature_and_auxiliary_item_count():
    spec = make_block_scan_spec(
        dtype="int",
        block_dim=(64, 1, 1),
        items_per_thread=4,
        mode="exclusive",
        algorithm="::cub::BLOCK_SCAN_RAKING",
        value_kind="array",
    )

    assert spec.method_name == "ExclusiveSum"
    assert spec.value_kind is ScanValueKind.ARRAY
    assert spec.specialization.template_parameter_names == (
        "T",
        "BLOCK_DIM_X",
        "ALGORITHM",
        "BLOCK_DIM_Y",
        "BLOCK_DIM_Z",
    )
    assert spec.specialization.ordered_specialization_arguments[-1] == (
        "ITEMS_PER_THREAD",
        4,
    )
    assert spec.specialization.symbol_mangling_inputs[-1][-1] == (
        "ITEMS_PER_THREAD",
        4,
    )
    assert [
        (item.name, item.role) for item in spec.specialization.classify_method()
    ] == [
        ("temp_storage", ParameterRole.TEMP_STORAGE),
        ("input", ParameterRole.INPUT),
        ("output", ParameterRole.OUTPUT),
    ]


def test_block_scan_cxx_operator_and_initial_value_are_static():
    spec = make_block_scan_spec(
        dtype="int",
        block_dim=(32, 2, 1),
        items_per_thread=1,
        mode="exclusive",
        algorithm="::cub::BLOCK_SCAN_WARP_SCANS",
        value_kind="scalar",
        initial_value=CxxFunction("7", Dependency("T"), name="initial_value"),
        scan_operator=CxxOperator(
            "::cuda::std::multiplies<T>",
            Dependency("T"),
            name="scan_op",
        ),
    )

    assert spec.method_name == "ExclusiveScan"
    assert spec.specialization.fake_return
    assert [
        (item.kind, item.role) for item in spec.specialization.classify_method()
    ] == [
        (ArgumentKind.RUNTIME, ParameterRole.TEMP_STORAGE),
        (ArgumentKind.RUNTIME, ParameterRole.INPUT),
        (ArgumentKind.RUNTIME, ParameterRole.OUTPUT),
        (ArgumentKind.STATIC, ParameterRole.CONSTANT),
        (ArgumentKind.STATIC, ParameterRole.OPERATOR),
    ]


def test_block_scan_distinguishes_stateless_and_stateful_operators():
    def add(left, right):
        return left + right

    class Prefix:
        pass

    spec = make_block_scan_spec(
        dtype="int",
        block_dim=(32, 1, 1),
        items_per_thread=2,
        mode="exclusive",
        algorithm="::cub::BLOCK_SCAN_RAKING",
        value_kind="array",
        scan_operator=PythonOperator(
            Dependency("T"),
            (Dependency("T"), Dependency("T")),
            add,
            name="scan_op",
        ),
        prefix_operator=StatefulOperator(
            Prefix(),
            state_dtype="prefix_state",
            ret_dtype=Dependency("T"),
            arg_dtypes=(Dependency("T"),),
            name="prefix_op",
        ),
    )

    classifications = spec.specialization.classify_method()
    assert classifications[-2].kind is ArgumentKind.STATIC
    assert classifications[-2].role is ParameterRole.OPERATOR
    assert classifications[-1].kind is ArgumentKind.RUNTIME
    assert classifications[-1].role is ParameterRole.STATE


def test_block_scan_aggregate_is_semantic_output_not_return_value():
    spec = make_block_scan_spec(
        dtype="int",
        block_dim=(32, 1, 1),
        items_per_thread=1,
        mode="inclusive",
        algorithm="::cub::BLOCK_SCAN_RAKING",
        value_kind="array",
        block_aggregate=True,
    )

    aggregate = spec.specialization.parameters[0][-1]
    assert aggregate.is_output
    assert aggregate.is_return is False
    assert aggregate.deref_on_call
    assert spec.specialization.classify_method()[-1].role is ParameterRole.OUTPUT


def test_block_scan_rejects_prefix_with_aggregate():
    with pytest.raises(ValueError, match="mutually exclusive"):
        make_block_scan_spec(
            dtype="int",
            block_dim=(32, 1, 1),
            items_per_thread=1,
            mode="exclusive",
            algorithm="::cub::BLOCK_SCAN_RAKING",
            value_kind="array",
            prefix_operator=PythonOperator(
                Dependency("T"),
                (Dependency("T"),),
                lambda value: value,
            ),
            block_aggregate=True,
        )


def test_block_scan_rejects_nonexistent_initial_value_overloads():
    initial = CxxFunction("0", Dependency("T"), name="initial_value")
    operator = CxxOperator(
        "::cuda::maximum<>",
        Dependency("T"),
        name="scan_op",
    )

    with pytest.raises(ValueError, match="sum overloads"):
        make_block_scan_spec(
            dtype="int",
            block_dim=(32, 1, 1),
            items_per_thread=1,
            mode="exclusive",
            algorithm="::cub::BLOCK_SCAN_RAKING",
            value_kind="scalar",
            initial_value=initial,
        )
    with pytest.raises(ValueError, match="no initial-value overload"):
        make_block_scan_spec(
            dtype="int",
            block_dim=(32, 1, 1),
            items_per_thread=1,
            mode="inclusive",
            algorithm="::cub::BLOCK_SCAN_RAKING",
            value_kind="scalar",
            initial_value=initial,
            scan_operator=operator,
        )
    with pytest.raises(ValueError, match="unsupported CUB BlockScan algorithm"):
        make_block_scan_spec(
            dtype="int",
            block_dim=(32, 1, 1),
            items_per_thread=1,
            mode="exclusive",
            algorithm="::cub::BLOCK_REDUCE_RAKING",
            value_kind="scalar",
        )
