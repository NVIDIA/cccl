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
    StatefulOperator,
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
    assert operation.prefix_callback is None


def test_scan_semantic_identity_includes_prefix_callback():
    def identity(value):
        return value

    def increment(value):
        return value + 1

    prefix = PythonOperator(
        Dependency("T"),
        (Dependency("T"),),
        identity,
        name="prefix_op",
    )
    different_prefix = PythonOperator(
        Dependency("T"),
        (Dependency("T"),),
        increment,
        name="prefix_op",
    )
    with_prefix = make_scan_semantics(
        dtype="int32",
        mode="exclusive",
        value_kind="scalar",
        items_per_thread=1,
        prefix_callback=prefix,
    )
    equivalent = make_scan_semantics(
        dtype="int32",
        mode="exclusive",
        value_kind="scalar",
        items_per_thread=1,
        prefix_callback=prefix,
    )
    without_prefix = make_scan_semantics(
        dtype="int32",
        mode="exclusive",
        value_kind="scalar",
        items_per_thread=1,
    )
    with_different_prefix = make_scan_semantics(
        dtype="int32",
        mode="exclusive",
        value_kind="scalar",
        items_per_thread=1,
        prefix_callback=different_prefix,
    )

    assert with_prefix.prefix_callback is prefix
    assert with_prefix == equivalent
    assert hash(with_prefix) == hash(equivalent)
    assert with_prefix != without_prefix
    assert with_prefix != with_different_prefix


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
    with pytest.raises(TypeError, match="unsupported scan prefix callback"):
        make_scan_semantics(
            dtype="int32",
            mode="exclusive",
            value_kind="scalar",
            items_per_thread=1,
            prefix_callback=object(),
        )


def test_scan_semantics_reject_prefix_callback_with_aggregate():
    prefix = PythonOperator(
        Dependency("T"),
        (Dependency("T"),),
        lambda value: value,
        name="prefix_op",
    )

    with pytest.raises(ValueError, match="aggregate and prefix callback"):
        make_scan_semantics(
            dtype="int32",
            mode="exclusive",
            value_kind="scalar",
            items_per_thread=1,
            aggregate=True,
            prefix_callback=prefix,
        )


def test_scan_semantics_reject_prefix_callback_with_initial_value():
    prefix = PythonOperator(
        Dependency("T"),
        (Dependency("T"),),
        lambda aggregate: aggregate,
        name="prefix_op",
    )

    with pytest.raises(ValueError, match="initial value and prefix callback"):
        make_scan_semantics(
            dtype="int32",
            mode="exclusive",
            value_kind="scalar",
            items_per_thread=1,
            initial_value=CxxFunction(
                "0",
                Dependency("T"),
                name="initial_value",
            ),
            prefix_callback=prefix,
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


def test_block_scan_sum_accepts_stateless_prefix_callback():
    prefix = PythonOperator(
        Dependency("T"),
        (Dependency("T"),),
        lambda value: value,
        name="prefix_op",
    )
    spec = make_block_scan_spec(
        dtype="int32",
        block_dim=(32, 1, 1),
        items_per_thread=1,
        mode="inclusive",
        algorithm="raking",
        value_kind="scalar",
        prefix_operator=prefix,
    )

    assert spec.method_name == "InclusiveSum"
    assert spec.has_prefix_callback
    assert [item.name for item in spec.specialization.parameters[0]] == [
        "temp_storage",
        "input",
        "output",
        "prefix_op",
    ]
    classification = spec.specialization.classify_method()[-1]
    assert classification.kind is ArgumentKind.STATIC
    assert classification.role is ParameterRole.OPERATOR


def test_block_scan_prefix_callback_follows_scan_operator_in_cub_signature():
    def maximum(left, right):
        return left if left > right else right

    def running_prefix(state, aggregate):
        previous = state[0]
        state[0] = maximum(previous, aggregate)
        return previous

    prefix = StatefulOperator(
        running_prefix,
        state_dtype="int64",
        ret_dtype=Dependency("T"),
        arg_dtypes=(Dependency("T"),),
        name="prefix_op",
    )
    spec = make_block_scan_spec(
        dtype="int32",
        block_dim=(32, 1, 1),
        items_per_thread=2,
        mode="exclusive",
        algorithm="raking",
        value_kind="array",
        scan_operator=PythonOperator(
            Dependency("T"),
            (Dependency("T"), Dependency("T")),
            maximum,
            name="scan_op",
        ),
        prefix_operator=prefix,
    )

    assert spec.method_name == "ExclusiveScan"
    assert spec.has_prefix_callback
    assert not spec.has_initial_value
    assert not spec.has_block_aggregate
    method = spec.specialization.parameters[0]
    assert [item.name for item in method] == [
        "temp_storage",
        "input",
        "output",
        "scan_op",
        "prefix_op",
    ]
    assert [
        (item.kind, item.role) for item in spec.specialization.classify_method()
    ] == [
        (ArgumentKind.RUNTIME, ParameterRole.TEMP_STORAGE),
        (ArgumentKind.RUNTIME, ParameterRole.INPUT),
        (ArgumentKind.RUNTIME, ParameterRole.OUTPUT),
        (ArgumentKind.STATIC, ParameterRole.OPERATOR),
        (ArgumentKind.RUNTIME, ParameterRole.STATE),
    ]
    assert spec.specialization.metadata["prefix_callback"] == "StatefulOperator"


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


def test_block_scan_rejects_prefix_callback_with_initial_value():
    prefix = PythonOperator(
        Dependency("T"),
        (Dependency("T"),),
        lambda value: value,
        name="prefix_op",
    )

    with pytest.raises(ValueError, match="initial value and prefix callback"):
        make_block_scan_spec(
            dtype="int32",
            block_dim=(32, 1, 1),
            items_per_thread=1,
            mode="exclusive",
            algorithm="raking",
            value_kind="scalar",
            scan_operator=CxxOperator(
                "::cuda::std::plus<T>",
                Dependency("T"),
                name="scan_op",
            ),
            initial_value=CxxFunction(
                "0",
                Dependency("T"),
                name="initial_value",
            ),
            prefix_operator=prefix,
        )
