# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
    WarpScanMode,
    make_warp_scan_spec,
)


@pytest.mark.parametrize(
    ("mode", "method_name"),
    [
        ("exclusive", "ExclusiveSum"),
        ("inclusive", "InclusiveSum"),
    ],
)
def test_warp_scan_selects_default_sum_entry_point(mode, method_name):
    spec = make_warp_scan_spec(
        dtype="int32",
        threads_in_warp=16,
        mode=mode,
    )

    assert spec.mode is WarpScanMode(mode)
    assert spec.method_name == method_name
    assert spec.uses_sum_method
    assert spec.specialization.fake_return
    assert [item.name for item in spec.specialization.parameters[0]] == [
        "temp_storage",
        "input",
        "output",
    ]


def test_partial_exclusive_sum_uses_plus_and_an_explicitly_typed_zero():
    spec = make_warp_scan_spec(
        dtype="int32",
        threads_in_warp=8,
        mode="exclusive",
        valid_items=ArgumentBinding.static(np.int32(5)),
    )

    assert spec.method_name == "ExclusiveScanPartial"
    assert not spec.uses_sum_method
    assert spec.call.scan_operator == CxxOperator(
        "::cuda::std::plus<T>",
        Dependency("T"),
        name="scan_op",
    )
    assert spec.call.initial_value == CxxFunction(
        "{T}{0}",
        Dependency("T"),
        name="initial_value",
    )
    assert spec.valid_items == ArgumentBinding.static(5)
    assert [item.name for item in spec.specialization.parameters[0]] == [
        "temp_storage",
        "input",
        "output",
        "initial_value",
        "scan_op",
        "valid_items",
    ]


def test_warp_scan_partial_signature_and_aggregate_output():
    spec = make_warp_scan_spec(
        dtype="int32",
        threads_in_warp=8,
        mode="inclusive",
        scan_operator=CxxOperator(
            "::cuda::maximum<T>",
            Dependency("T"),
            name="scan_op",
        ),
        valid_items=True,
        warp_aggregate=True,
    )

    assert spec.method_name == "InclusiveScanPartial"
    assert spec.has_valid_items
    assert spec.has_warp_aggregate
    assert [
        (item.name, item.kind, item.role)
        for item in spec.specialization.classify_method()
    ] == [
        ("temp_storage", ArgumentKind.RUNTIME, ParameterRole.TEMP_STORAGE),
        ("input", ArgumentKind.RUNTIME, ParameterRole.INPUT),
        ("output", ArgumentKind.RUNTIME, ParameterRole.OUTPUT),
        ("scan_op", ArgumentKind.STATIC, ParameterRole.OPERATOR),
        ("valid_items", ArgumentKind.RUNTIME, ParameterRole.INPUT),
        ("warp_aggregate", ArgumentKind.RUNTIME, ParameterRole.OUTPUT),
    ]
    aggregate = spec.specialization.parameters[0][-1]
    assert aggregate.is_return is False
    assert aggregate.deref_on_call
    assert spec.specialization.metadata["aggregate_excludes_initial"]


def test_warp_scan_accepts_runtime_initial_value_and_python_operator():
    def maximum(left, right):
        return left if left > right else right

    spec = make_warp_scan_spec(
        dtype="int32",
        threads_in_warp=32,
        mode="exclusive",
        scan_operator=PythonOperator(
            Dependency("T"),
            (Dependency("T"), Dependency("T")),
            maximum,
            name="scan_op",
        ),
        initial_value=Reference("int32", name="initial_value"),
    )

    assert spec.method_name == "ExclusiveScan"
    classifications = spec.specialization.classify_method()
    assert classifications[3].kind is ArgumentKind.RUNTIME
    assert classifications[3].role is ParameterRole.INPUT
    assert classifications[4].kind is ArgumentKind.STATIC
    assert classifications[4].role is ParameterRole.OPERATOR


def test_warp_scan_semantic_identity_includes_width_prefix_and_outputs():
    base = make_warp_scan_spec(
        dtype="int32",
        threads_in_warp=16,
        mode="inclusive",
    )
    narrower = make_warp_scan_spec(
        dtype="int32",
        threads_in_warp=8,
        mode="inclusive",
    )
    partial = make_warp_scan_spec(
        dtype="int32",
        threads_in_warp=16,
        mode="inclusive",
        valid_items=ArgumentBinding.runtime(),
    )
    aggregate = make_warp_scan_spec(
        dtype="int32",
        threads_in_warp=16,
        mode="inclusive",
        warp_aggregate=True,
    )

    assert base.semantic_key != narrower.semantic_key
    assert base.semantic_key != partial.semantic_key
    assert base.semantic_key != aggregate.semantic_key


@pytest.mark.parametrize("valid_items", [0, -1, 9])
def test_warp_scan_rejects_static_prefix_outside_logical_width(valid_items):
    with pytest.raises(ValueError, match="between 1 and the logical warp size"):
        make_warp_scan_spec(
            dtype="int32",
            threads_in_warp=8,
            mode="inclusive",
            valid_items=ArgumentBinding.static(valid_items),
        )


@pytest.mark.parametrize("threads_in_warp", [True, 0, 6, 64])
def test_warp_scan_rejects_invalid_logical_warp_width(threads_in_warp):
    with pytest.raises(ValueError, match="power of two"):
        make_warp_scan_spec(
            dtype="int32",
            threads_in_warp=threads_in_warp,
            mode="inclusive",
        )
