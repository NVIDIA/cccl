# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

import operator

import pytest

from cuda.coop._core import (
    INT8,
    INT32,
    ArgumentKind,
    Array,
    CxxOperator,
    Dependency,
    ParameterRole,
    PythonOperator,
    Reference,
    TempStorageParameter,
    Value,
)
from cuda.coop._core.warp import (
    WarpMergeSortPayload,
    WarpMergeSortTilePolicy,
    make_warp_merge_sort_spec,
)


def _python_compare(op=operator.lt):
    return PythonOperator(
        ret_dtype=INT8,
        arg_dtypes=(Dependency("KeyT"), Dependency("KeyT")),
        op=op,
        name="compare_op",
    )


def test_warp_merge_sort_keys_semantics():
    compare = _python_compare()
    spec = make_warp_merge_sort_spec(
        key_dtype="i32",
        items_per_thread=3,
        threads_in_warp=16,
        compare_operator=compare,
    )

    assert spec.payload is WarpMergeSortPayload.KEYS
    assert not spec.has_values
    assert spec.method_name == "Sort"
    assert spec.key_dtype == "i32"
    assert spec.value_dtype is None
    assert spec.items_per_thread == 3
    assert spec.threads_in_warp == 16
    assert spec.compare_operator is compare
    assert spec.specialization.template_arguments == {
        "KeyT": "i32",
        "ITEMS_PER_THREAD": 3,
        "VIRTUAL_WARP_THREADS": 16,
        "ValueT": "::cub::NullType",
    }
    assert spec.specialization.parameters == (
        (
            TempStorageParameter(),
            Array(
                Dependency("KeyT"),
                Dependency("ITEMS_PER_THREAD"),
                name="keys",
                is_inout=True,
                is_return=False,
            ),
            compare,
        ),
    )
    assert [
        (parameter.kind, parameter.role)
        for parameter in spec.specialization.classify_method()
    ] == [
        (ArgumentKind.RUNTIME, ParameterRole.TEMP_STORAGE),
        (ArgumentKind.RUNTIME, ParameterRole.INOUT),
        (ArgumentKind.STATIC, ParameterRole.OPERATOR),
    ]


def test_warp_merge_sort_pairs_semantics():
    spec = make_warp_merge_sort_spec(
        key_dtype="i64",
        value_dtype="f32",
        items_per_thread=2,
        threads_in_warp=8,
        compare_operator=_python_compare(),
    )

    assert spec.payload is WarpMergeSortPayload.PAIRS
    assert spec.has_values
    assert spec.value_dtype == "f32"
    assert spec.specialization.template_arguments["ValueT"] == "f32"
    assert [parameter.name for parameter in spec.specialization.parameters[0]] == [
        "temp_storage",
        "keys",
        "values",
        "compare_op",
    ]
    assert spec.specialization.parameters[0][1].is_inout
    assert spec.specialization.parameters[0][2].is_inout


def test_warp_merge_sort_pairs_partial_tile_semantics():
    spec = make_warp_merge_sort_spec(
        key_dtype="i32",
        value_dtype="f64",
        items_per_thread=2,
        threads_in_warp=32,
        compare_operator=_python_compare(),
        valid_items=59,
        oob_default=999,
    )

    assert spec.tile_policy is WarpMergeSortTilePolicy.PARTIAL
    assert spec.has_partial_tile
    assert [parameter.name for parameter in spec.specialization.parameters[0]] == [
        "temp_storage",
        "keys",
        "values",
        "compare_op",
        "valid_items",
        "oob_default",
    ]
    assert spec.specialization.parameters[0][-2] == Value(INT32, name="valid_items")
    assert spec.specialization.parameters[0][-1] == Reference(
        Dependency("KeyT"), name="oob_default"
    )


def test_warp_merge_sort_accepts_cxx_comparison_operator():
    compare = CxxOperator(
        cpp="::cuda::std::greater<KeyT>",
        dtype=Dependency("KeyT"),
        name="compare_op",
    )
    spec = make_warp_merge_sort_spec(
        key_dtype="i32",
        items_per_thread=1,
        threads_in_warp=32,
        compare_operator=compare,
    )

    assert spec.compare_operator is compare
    assert spec.specialization.parameters[0][-1] is compare


def test_warp_merge_sort_semantic_identity_tracks_shape_payload_tile_and_operator():
    def make(*, items=2, threads=16, value=None, partial=False, op=operator.lt):
        return make_warp_merge_sort_spec(
            key_dtype="i32",
            value_dtype=value,
            items_per_thread=items,
            threads_in_warp=threads,
            compare_operator=_python_compare(op),
            valid_items=17 if partial else None,
            oob_default=999 if partial else None,
        )

    assert make().semantic_key == make().semantic_key
    assert make().semantic_key != make(items=3).semantic_key
    assert make().semantic_key != make(threads=8).semantic_key
    assert make().semantic_key != make(value="f32").semantic_key
    assert make().semantic_key != make(partial=True).semantic_key
    assert make(partial=True).semantic_key == make(partial=True).semantic_key
    assert make().semantic_key != make(op=operator.gt).semantic_key


@pytest.mark.parametrize(
    ("items_per_thread", "threads_in_warp"),
    [(0, 32), (True, 32), (1, 3), (1, 64)],
)
def test_warp_merge_sort_rejects_invalid_shape(items_per_thread, threads_in_warp):
    with pytest.raises(ValueError):
        make_warp_merge_sort_spec(
            key_dtype="i32",
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            compare_operator=_python_compare(),
        )


def test_warp_merge_sort_rejects_missing_dtype_or_comparison_operator():
    with pytest.raises(ValueError, match="key dtype must be provided"):
        make_warp_merge_sort_spec(
            key_dtype=None,
            items_per_thread=1,
            threads_in_warp=32,
            compare_operator=_python_compare(),
        )
    with pytest.raises(TypeError, match="requires a comparison operator"):
        make_warp_merge_sort_spec(
            key_dtype="i32",
            items_per_thread=1,
            threads_in_warp=32,
            compare_operator=None,
        )
    with pytest.raises(ValueError, match="compare_op must be provided"):
        make_warp_merge_sort_spec(
            key_dtype="i32",
            items_per_thread=1,
            threads_in_warp=32,
            compare_operator=_python_compare(None),
        )


@pytest.mark.parametrize(
    ("valid_items", "oob_default"),
    [(17, None), (None, 999)],
)
def test_warp_merge_sort_requires_complete_partial_tile_arguments(
    valid_items, oob_default
):
    with pytest.raises(
        ValueError, match="valid_items and oob_default must be provided together"
    ):
        make_warp_merge_sort_spec(
            key_dtype="i32",
            items_per_thread=2,
            threads_in_warp=32,
            compare_operator=_python_compare(),
            valid_items=valid_items,
            oob_default=oob_default,
        )
