# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402


import numpy as np
import pytest

from cuda.coop._core import (
    ArgumentBinding,
    ArgumentKind,
    CxxFunction,
    CxxOperator,
    Dependency,
    ParameterRole,
    PythonOperator,
    Reference,
)
from cuda.coop._core.warp import WarpScanMode, make_warp_scan_spec


@pytest.mark.parametrize(
    ("mode", "method_name"),
    [
        ("exclusive", "ExclusiveSum"),
        ("inclusive", "InclusiveSum"),
    ],
)
def test_warp_scan_selects_default_sum_entry_point(mode, method_name):
    spec = make_warp_scan_spec(
        dtype="int",
        threads_in_warp=16,
        mode=mode,
    )

    assert spec.mode is WarpScanMode(mode)
    assert spec.method_name == method_name
    assert spec.uses_sum_method
    assert spec.specialization.fake_return
    assert spec.specialization.parameters[0][2].is_return is True
    assert [item.name for item in spec.specialization.parameters[0]] == [
        "temp_storage",
        "input",
        "output",
    ]


@pytest.mark.parametrize(
    ("mode", "method_name"),
    [
        ("exclusive", "ExclusiveScanPartial"),
        ("inclusive", "InclusiveScanPartial"),
    ],
)
def test_warp_scan_partial_signature_and_aggregate(mode, method_name):
    spec = make_warp_scan_spec(
        dtype="int",
        threads_in_warp=8,
        mode=mode,
        scan_operator=CxxOperator(
            "::cuda::maximum<>",
            Dependency("T"),
            name="scan_op",
        ),
        initial_value=Reference(Dependency("T"), name="initial_value"),
        valid_items=True,
        warp_aggregate=True,
    )

    assert spec.method_name == method_name
    assert spec.has_initial_value
    assert spec.has_valid_items
    assert spec.has_warp_aggregate
    assert [
        (item.name, item.kind, item.role)
        for item in spec.specialization.classify_method()
    ] == [
        ("temp_storage", ArgumentKind.RUNTIME, ParameterRole.TEMP_STORAGE),
        ("input", ArgumentKind.RUNTIME, ParameterRole.INPUT),
        ("output", ArgumentKind.RUNTIME, ParameterRole.OUTPUT),
        ("initial_value", ArgumentKind.RUNTIME, ParameterRole.INPUT),
        ("scan_op", ArgumentKind.STATIC, ParameterRole.OPERATOR),
        ("valid_items", ArgumentKind.RUNTIME, ParameterRole.INPUT),
        ("warp_aggregate", ArgumentKind.RUNTIME, ParameterRole.OUTPUT),
    ]
    aggregate = spec.specialization.parameters[0][-1]
    assert aggregate.is_return is False
    assert aggregate.deref_on_call
    assert aggregate.is_array_pointer


def test_warp_scan_static_initial_and_python_operator_are_distinct_semantics():
    def add(left, right):
        return left + right

    first = make_warp_scan_spec(
        dtype="int",
        threads_in_warp=32,
        mode="exclusive",
        scan_operator=PythonOperator(
            Dependency("T"),
            (Dependency("T"), Dependency("T")),
            add,
            name="scan_op",
        ),
        initial_value=CxxFunction("0", Dependency("T"), name="initial_value"),
    )
    second = make_warp_scan_spec(
        dtype="int",
        threads_in_warp=32,
        mode="exclusive",
        scan_operator=PythonOperator(
            Dependency("T"),
            (Dependency("T"), Dependency("T")),
            add,
            name="scan_op",
        ),
        initial_value=CxxFunction("1", Dependency("T"), name="initial_value"),
    )

    assert first.method_name == "ExclusiveScan"
    assert first.specialization.classify_method()[3].kind is ArgumentKind.STATIC
    assert first.specialization.classify_method()[3].role is ParameterRole.CONSTANT
    assert first.semantic_key != second.semantic_key


def test_warp_scan_accepts_numpy_static_valid_items():
    spec = make_warp_scan_spec(
        dtype="int",
        threads_in_warp=8,
        mode="inclusive",
        valid_items=ArgumentBinding.static(np.int32(5)),
    )

    valid_items = spec.specialization.parameters[0][-1]
    assert valid_items.name == "valid_items"
    assert valid_items.cpp == "5"
    plain = make_warp_scan_spec(
        dtype="int",
        threads_in_warp=8,
        mode="inclusive",
        valid_items=ArgumentBinding.static(5),
    )
    assert spec.valid_items == ArgumentBinding.static(5)
    assert spec.semantic_key == plain.semantic_key


@pytest.mark.parametrize(
    "scan_operator",
    [
        None,
        CxxOperator("::cuda::maximum<>", Dependency("T"), name="scan_op"),
    ],
)
def test_warp_scan_rejects_partial_exclusive_without_initial_value(scan_operator):
    with pytest.raises(ValueError, match="leaves lane zero undefined"):
        make_warp_scan_spec(
            dtype="int",
            threads_in_warp=8,
            mode="exclusive",
            scan_operator=scan_operator,
            valid_items=True,
        )


def test_warp_scan_semantic_identity_includes_mode_width_and_outputs():
    base = make_warp_scan_spec(
        dtype="int",
        threads_in_warp=16,
        mode="exclusive",
    )
    inclusive = make_warp_scan_spec(
        dtype="int",
        threads_in_warp=16,
        mode="inclusive",
    )
    narrower = make_warp_scan_spec(
        dtype="int",
        threads_in_warp=8,
        mode="exclusive",
    )
    aggregate = make_warp_scan_spec(
        dtype="int",
        threads_in_warp=16,
        mode="exclusive",
        warp_aggregate=True,
    )

    assert base.semantic_key != inclusive.semantic_key
    assert base.semantic_key != narrower.semantic_key
    assert base.semantic_key != aggregate.semantic_key


def test_warp_scan_rejects_non_default_call_without_operator():
    with pytest.raises(ValueError, match="requires a scan operator"):
        make_warp_scan_spec(
            dtype="int",
            threads_in_warp=32,
            mode="exclusive",
            initial_value=CxxFunction("0", "int"),
        )
    with pytest.raises(TypeError, match="unsupported scan operator"):
        make_warp_scan_spec(
            dtype="int",
            threads_in_warp=32,
            mode="inclusive",
            scan_operator=object(),
        )


@pytest.mark.parametrize("threads_in_warp", [True, 0, 6, 64])
def test_warp_scan_rejects_invalid_logical_warp_width(threads_in_warp):
    with pytest.raises(ValueError, match="power of two"):
        make_warp_scan_spec(
            dtype="int",
            threads_in_warp=threads_in_warp,
            mode="inclusive",
        )
