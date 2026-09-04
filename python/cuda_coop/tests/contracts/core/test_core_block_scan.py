# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from cuda.coop._core import (
    ArgumentKind,
    CxxFunction,
    CxxOperator,
    Dependency,
    ParameterRole,
    PythonOperator,
    Reference,
    ScanMode,
    ScanValueKind,
    make_block_scan_spec,
    make_scan_semantics,
)


def test_scan_semantics_are_out_of_place_shape_and_operator_neutral():
    operation = make_scan_semantics(
        dtype="int32",
        mode="inclusive",
        value_kind="array",
        items_per_thread=4,
        aggregate=True,
    )

    assert operation.mode is ScanMode.INCLUSIVE
    assert operation.value_kind is ScanValueKind.ARRAY
    assert operation.items_per_thread == 4
    assert operation.aggregate
    assert operation.scan_operator is None


def test_scan_semantics_require_exact_initial_value_dtype():
    matching = make_scan_semantics(
        dtype="int32",
        mode="exclusive",
        value_kind="scalar",
        items_per_thread=1,
        initial_value=Reference("int32", name="initial_value"),
    )
    dependent = make_scan_semantics(
        dtype="int32",
        mode="exclusive",
        value_kind="scalar",
        items_per_thread=1,
        initial_value=CxxFunction("7", Dependency("T"), name="initial_value"),
    )

    assert matching.initial_value.dtype == "int32"
    assert dependent.initial_value.dtype == Dependency("T")
    with pytest.raises(TypeError, match="exactly match"):
        make_scan_semantics(
            dtype="int32",
            mode="exclusive",
            value_kind="scalar",
            items_per_thread=1,
            initial_value=Reference("int64", name="initial_value"),
        )


def test_scan_semantics_validate_invalid_forms():
    with pytest.raises(ValueError, match="items_per_thread == 1"):
        make_scan_semantics(
            dtype="int32",
            mode="exclusive",
            value_kind="scalar",
            items_per_thread=2,
        )
    with pytest.raises(ValueError, match="inclusive scans"):
        make_scan_semantics(
            dtype="int32",
            mode="inclusive",
            value_kind="scalar",
            items_per_thread=1,
            initial_value=CxxFunction("0", Dependency("T")),
        )
    with pytest.raises(TypeError, match="unsupported scan operator"):
        make_scan_semantics(
            dtype="int32",
            mode="inclusive",
            value_kind="scalar",
            items_per_thread=1,
            scan_operator=object(),
        )


def test_block_scan_sum_array_signature_is_out_of_place():
    spec = make_block_scan_spec(
        dtype="int32",
        block_dim=(64, 1, 1),
        items_per_thread=4,
        mode="exclusive",
        algorithm="raking",
        value_kind="array",
    )

    assert spec.method_name == "ExclusiveSum"
    assert spec.value_kind is ScanValueKind.ARRAY
    assert spec.specialization.ordered_specialization_arguments[-1] == (
        "ITEMS_PER_THREAD",
        4,
    )
    method = spec.specialization.parameters[0]
    assert [item.name for item in method] == ["temp_storage", "input", "output"]
    assert method[1].is_output is False
    assert method[2].is_output is True
    assert method[2].is_inout is False


def test_block_scan_custom_operator_and_initial_value_signature():
    spec = make_block_scan_spec(
        dtype="int32",
        block_dim=(32, 2, 1),
        items_per_thread=1,
        mode="exclusive",
        algorithm="warp_scans",
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


def test_block_scan_accepts_stateless_python_operator():
    def maximum(left, right):
        return left if left > right else right

    spec = make_block_scan_spec(
        dtype="int32",
        block_dim=(32, 1, 1),
        items_per_thread=2,
        mode="inclusive",
        algorithm="raking_memoize",
        value_kind="array",
        scan_operator=PythonOperator(
            Dependency("T"),
            (Dependency("T"), Dependency("T")),
            maximum,
            name="scan_op",
        ),
    )

    operator = spec.specialization.classify_method()[-1]
    assert operator.kind is ArgumentKind.STATIC
    assert operator.role is ParameterRole.OPERATOR


def test_block_scan_aggregate_is_an_explicit_side_output():
    spec = make_block_scan_spec(
        dtype="int32",
        block_dim=(32, 1, 1),
        items_per_thread=1,
        mode="inclusive",
        algorithm="raking",
        value_kind="scalar",
        block_aggregate=True,
    )

    aggregate = spec.specialization.parameters[0][-1]
    assert aggregate.name == "block_aggregate"
    assert aggregate.is_output
    assert aggregate.is_return is False
    assert aggregate.deref_on_call
    assert spec.specialization.metadata["aggregate_excludes_initial"]


def test_block_scan_rejects_noncanonical_sum_initial_and_bad_algorithm():
    with pytest.raises(ValueError, match="sum overloads"):
        make_block_scan_spec(
            dtype="int32",
            block_dim=(32, 1, 1),
            items_per_thread=1,
            mode="exclusive",
            algorithm="raking",
            value_kind="scalar",
            initial_value=CxxFunction("0", Dependency("T")),
        )
    with pytest.raises(ValueError, match="unsupported CUB BlockScan algorithm"):
        make_block_scan_spec(
            dtype="int32",
            block_dim=(32, 1, 1),
            items_per_thread=1,
            mode="exclusive",
            algorithm="::cub::BLOCK_REDUCE_RAKING",
            value_kind="scalar",
        )
    with pytest.raises(ValueError, match="multiple of 32"):
        make_block_scan_spec(
            dtype="int32",
            block_dim=(48, 1, 1),
            items_per_thread=1,
            mode="inclusive",
            algorithm="warp_scans",
            value_kind="scalar",
        )
