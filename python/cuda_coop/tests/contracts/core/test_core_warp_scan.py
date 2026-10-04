# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check full and partial WarpScan signatures without a backend compiler.

Logical width, prefix binding, and aggregate outputs affect both the CUB
entry point and specialization identity. Initial values and custom
operators must keep their distinct runtime or static argument roles.
"""

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
    classify_parameter,
    make_warp_scan_specialization,
)


@pytest.mark.parametrize(
    ("mode", "method_name"),
    [
        ("exclusive", "ExclusiveSum"),
        ("inclusive", "InclusiveSum"),
    ],
)
def test_warp_scan_selects_default_sum_entry_point(mode, method_name):
    specialization = make_warp_scan_specialization(
        dtype="int32",
        threads_in_warp=16,
        mode=mode,
    )

    assert specialization.mode is WarpScanMode(mode)
    assert specialization.method_name == method_name
    assert specialization.specialization.metadata["operator"] is None
    assert specialization.specialization.fake_return
    assert [
        item.name for item in specialization.specialization.parameters[0]
    ] == [
        "temp_storage",
        "input",
        "output",
    ]


def test_partial_exclusive_sum_uses_plus_and_an_explicitly_typed_zero():
    """Check the general Scan overload used for a partial exclusive sum.

    CUB's partial scan takes an operator, so addition needs an explicit plus
    functor. Without an initial value, rank zero's exclusive result would
    be undefined. A zero gives rank zero the exclusive-sum result, and using
    the payload dtype keeps C++ input, output, and initial types consistent.
    """

    specialization = make_warp_scan_specialization(
        dtype="int32",
        threads_in_warp=8,
        mode="exclusive",
        valid_items=ArgumentBinding.static(np.int32(5)),
    )

    assert specialization.method_name == "ExclusiveScanPartial"
    assert specialization.specialization.metadata["operator"] is not None
    assert specialization.call.scan_operator == CxxOperator(
        "::cuda::std::plus<T>",
        Dependency("T"),
        name="scan_op",
    )
    assert specialization.call.initial_value == CxxFunction(
        "{T}{0}",
        Dependency("T"),
        name="initial_value",
    )
    assert specialization.valid_items == ArgumentBinding.static(5)
    assert [
        item.name for item in specialization.specialization.parameters[0]
    ] == [
        "temp_storage",
        "input",
        "output",
        "initial_value",
        "scan_op",
        "valid_items",
    ]


def test_warp_scan_partial_signature_and_aggregate_output():
    specialization = make_warp_scan_specialization(
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

    assert specialization.method_name == "InclusiveScanPartial"
    assert specialization.has_valid_items
    assert specialization.call.aggregate
    assert [
        (item.name, item.kind, item.role)
        for item in map(
            classify_parameter, specialization.specialization.parameters[0]
        )
    ] == [
        ("temp_storage", ArgumentKind.RUNTIME, ParameterRole.TEMP_STORAGE),
        ("input", ArgumentKind.RUNTIME, ParameterRole.INPUT),
        ("output", ArgumentKind.RUNTIME, ParameterRole.OUTPUT),
        ("scan_op", ArgumentKind.STATIC, ParameterRole.OPERATOR),
        ("valid_items", ArgumentKind.RUNTIME, ParameterRole.INPUT),
        ("warp_aggregate", ArgumentKind.RUNTIME, ParameterRole.OUTPUT),
    ]
    aggregate = specialization.specialization.parameters[0][-1]
    assert aggregate.is_return is False
    assert aggregate.deref_on_call
    assert specialization.specialization.metadata["aggregate_excludes_initial"]


def test_warp_scan_accepts_runtime_initial_value_and_python_operator():
    def maximum(left, right):
        return max(right, left)

    specialization = make_warp_scan_specialization(
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

    assert specialization.method_name == "ExclusiveScan"
    classifications = tuple(
        map(classify_parameter, specialization.specialization.parameters[0])
    )
    assert classifications[3].kind is ArgumentKind.RUNTIME
    assert classifications[3].role is ParameterRole.INPUT
    assert classifications[4].kind is ArgumentKind.STATIC
    assert classifications[4].role is ParameterRole.OPERATOR


def test_warp_scan_semantic_identity_includes_width_prefix_and_outputs():
    base = make_warp_scan_specialization(
        dtype="int32",
        threads_in_warp=16,
        mode="inclusive",
    )
    narrower = make_warp_scan_specialization(
        dtype="int32",
        threads_in_warp=8,
        mode="inclusive",
    )
    partial = make_warp_scan_specialization(
        dtype="int32",
        threads_in_warp=16,
        mode="inclusive",
        valid_items=ArgumentBinding.runtime(),
    )
    aggregate = make_warp_scan_specialization(
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
        make_warp_scan_specialization(
            dtype="int32",
            threads_in_warp=8,
            mode="inclusive",
            valid_items=ArgumentBinding.static(valid_items),
        )


@pytest.mark.parametrize("threads_in_warp", [True, 0, 6, 64])
def test_warp_scan_rejects_invalid_logical_warp_width(threads_in_warp):
    with pytest.raises(ValueError, match="power of two"):
        make_warp_scan_specialization(
            dtype="int32",
            threads_in_warp=threads_in_warp,
            mode="inclusive",
        )
