# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from enum import Enum
from inspect import Parameter, signature

import pytest

import cuda.coop.numba_mlir as coop
import cuda.coop.numba_mlir._group_ops as numba_mlir_group_ops
import cuda.coop.numba_mlir._thread_group as numba_mlir_groups
from cuda.coop._core import ThreadHierarchy as CoreThreadHierarchy

from ....support.cases.api_contracts import (
    GROUP_CONSTRUCTOR_SIGNATURES,
    GROUP_METHOD_SIGNATURES,
    PORTABLE_GROUP_PRIMITIVE_SIGNATURES,
    QUALIFIED_GROUP_PRIMITIVE_SUFFIXES,
    REQUIRED_PARAMETER,
    portable_group_primitive_parameter_contract,
)


def _parameter_contract(member):
    result = []
    for parameter in signature(member).parameters.values():
        default = parameter.default
        if default is Parameter.empty:
            default = REQUIRED_PARAMETER
        elif isinstance(default, Enum):
            default = default.value
        result.append((parameter.name, parameter.kind.name, default))
    return tuple(result)


@pytest.mark.parametrize("name", GROUP_CONSTRUCTOR_SIGNATURES)
def test_group_constructor_signatures_match_contract(name):
    assert str(signature(getattr(coop, name))) == GROUP_CONSTRUCTOR_SIGNATURES[name]


@pytest.mark.parametrize("name", GROUP_METHOD_SIGNATURES)
def test_group_method_signatures_match_contract(name):
    assert (
        str(signature(getattr(coop.ThreadGroup, name))) == GROUP_METHOD_SIGNATURES[name]
    )


@pytest.mark.parametrize("name", PORTABLE_GROUP_PRIMITIVE_SIGNATURES)
def test_group_primitive_signatures_match_contract(name):
    suffix = QUALIFIED_GROUP_PRIMITIVE_SUFFIXES["numba_mlir"].get(name, ())

    assert _parameter_contract(getattr(coop, name)) == (
        portable_group_primitive_parameter_contract(name, suffix=suffix)
    )


def test_group_exports_are_runtime_free_and_use_shared_hierarchy():
    assert coop.Hierarchy is coop.ThreadHierarchy
    assert coop.ThreadHierarchy is CoreThreadHierarchy

    for name in (
        "Hierarchy",
        "ThreadGroup",
        "ThreadHierarchy",
        *GROUP_CONSTRUCTOR_SIGNATURES,
        *numba_mlir_group_ops.__all__,
    ):
        assert name in coop.__all__


def test_current_group_construction_preserves_backend_type():
    current = coop.this_block()

    assert type(current) is coop.ThreadGroup
    assert current.is_current
    assert repr(current).startswith("ThreadGroup(kind='block'")
    assert current.block_dim is None
    assert current.static_size is None
    assert coop.this_cluster().static_size is None
    assert coop.this_grid().static_size is None
    assert coop.this_thread().static_size == 1


def test_group_equality_hashing_and_group_by_use_numba_mlir_type():
    first = coop.this_block()
    second = coop.this_block()
    mapped = first.group_by(2)

    assert first == second
    assert hash(first) == hash(second)
    assert type(mapped) is coop.ThreadGroup
    assert type(mapped.parent) is coop.ThreadGroup
    assert mapped.kind == "warps_within_block"
    assert mapped.static_size == 64
    assert mapped.groups_per_parent is None
    assert mapped.complete_membership is None


def test_group_constructors_preserve_shared_validation():
    with pytest.raises(TypeError):
        coop.ThreadHierarchy(block_dim=64)
    with pytest.raises(TypeError):
        coop.this_warp(16)
    with pytest.raises(TypeError):
        coop.this_warp(block_dim=64)
    with pytest.raises(ValueError, match="requires the count to divide"):
        coop.this_warp().group_by(12)
    with pytest.raises(NotImplementedError, match="nested"):
        coop.this_warp().group_by(8).group_by(2)


def test_group_methods_use_one_compile_time_marker(monkeypatch):
    calls = []
    marker_result = object()

    def marker(group, operation, *args):
        calls.append((group, operation, args))
        return marker_result

    monkeypatch.setattr(
        numba_mlir_groups,
        "_thread_group_method_marker",
        marker,
    )
    group = coop.this_block()

    assert group.rank("block") is marker_result
    assert group.count("grid") is marker_result
    assert group.rank_as("uint32", "warp") is marker_result
    assert group.count_as("uint64", "thread") is marker_result
    assert group.sync() is None
    assert group.sync_aligned() is None
    assert group.is_member() is marker_result
    assert calls == [
        (group, "rank", (None, "block")),
        (group, "count", (None, "grid")),
        (group, "rank", ("uint32", "warp")),
        (group, "count", ("uint64", "thread")),
        (group, "sync", ()),
        (group, "sync_aligned", ()),
        (group, "is_member", ()),
    ]


def test_group_method_marker_fails_clearly_outside_compilation():
    with pytest.raises(RuntimeError, match="whole-function planner"):
        coop.this_block().rank()

    with pytest.raises(ValueError, match="level must be one of"):
        coop.this_block().count("tile")


def test_group_primitives_use_one_compile_time_marker(monkeypatch):
    calls = []
    marker_result = object()

    def marker(operation, *args, **kwargs):
        calls.append((operation, args, kwargs))
        return marker_result

    monkeypatch.setattr(
        numba_mlir_group_ops,
        "_group_primitive_marker",
        marker,
    )
    group = coop.this_block()
    invocations = (
        ("load", (group, "source", "output"), {"valid_items": 31}),
        ("store", (group, "destination", "value"), {"offset": 2}),
        (
            "reduce",
            (group, "value"),
            {"binary_op": "plus", "broadcast": False},
        ),
        ("sum", (group, "value"), {"valid_items": 31}),
        ("scan", (group, "value"), {"mode": "inclusive"}),
        ("exclusive_sum", (group, "value"), {}),
        ("inclusive_sum", (group, "value"), {}),
        (
            "exclusive_scan",
            (group, "value"),
            {"scan_op": "max", "initial_value": 0},
        ),
        ("inclusive_scan", (group, "value"), {}),
        ("exchange", (group, "value"), {"mode": "blocked_to_striped"}),
        (
            "adjacent_difference",
            (group, "value"),
            {"direction": "right", "difference_op": "difference"},
        ),
        (
            "discontinuity",
            (group, "value"),
            {"mode": "tails", "flag_op": "flag"},
        ),
        (
            "shuffle",
            (group, "value"),
            {"distance": 2, "block_prefix": "prefix"},
        ),
        (
            "merge_sort_keys",
            (group, "keys"),
            {"descending": True, "compare_op": "compare"},
        ),
        (
            "merge_sort_pairs",
            (group, "keys", "values"),
            {"descending": True, "compare_op": "compare"},
        ),
        ("radix_sort_keys", (group, "keys"), {"begin_bit": 3}),
        (
            "radix_sort_pairs",
            (group, "keys", "values"),
            {"begin_bit": 3},
        ),
        (
            "radix_rank",
            (group, "keys"),
            {"radix_bits": 4, "exclusive_digit_prefix": "prefix"},
        ),
        ("histogram", (group, "samples"), {"bins": 256}),
        (
            "run_length_decode",
            (group, "run_values", "run_lengths"),
            {
                "decoded_items_per_thread": 4,
                "relative_offsets": "offsets",
            },
        ),
        ("topk_max_keys", (group, "keys", 7), {}),
        ("topk_min_keys", (group, "keys", 7), {}),
        ("topk_max_pairs", (group, "keys", "values", 7), {}),
        ("topk_min_pairs", (group, "keys", "values", 7), {}),
    )

    for name, args, kwargs in invocations:
        result = getattr(coop, name)(*args, **kwargs)
        if name == "store":
            assert result is None
        else:
            assert result is marker_result

    assert {operation for operation, _, _ in calls} == set(numba_mlir_group_ops.__all__)
    calls_by_operation = {
        operation: (args, kwargs) for operation, args, kwargs in calls
    }
    assert calls_by_operation["load"] == (
        (group, "source", "output"),
        {
            "algorithm": "direct",
            "valid_items": 31,
            "oob_default": None,
            "offset": None,
            "temp_storage": None,
        },
    )
    assert calls_by_operation["reduce"] == (
        (group, "value"),
        {
            "binary_op": "plus",
            "broadcast": False,
            "valid_items": None,
            "algorithm": None,
        },
    )
    assert calls_by_operation["histogram"][1] == {
        "bins": 256,
        "bins_per_thread": 1,
        "counter_dtype": None,
        "algorithm": "atomic",
    }
    assert calls_by_operation["run_length_decode"][1] == {
        "decoded_items_per_thread": 4,
        "decoded_window_offset": 0,
        "relative_offsets": "offsets",
        "total_decoded_size": None,
        "decoded_offset_dtype": None,
    }


def test_group_primitive_marker_fails_clearly_outside_compilation():
    with pytest.raises(
        RuntimeError,
        match=(
            r"cuda\.coop\.numba_mlir\.reduce is a compile-time kernel "
            "construct.*whole-function planner"
        ),
    ):
        coop.reduce(coop.this_block(), 1)


def test_thread_data_signature_uses_canonical_prefix_and_alignas_suffix():
    parameters = signature(coop.ThreadData).parameters

    assert tuple(parameters) == ("items_per_thread", "dtype", "alignas")
    assert parameters["items_per_thread"].kind is Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["dtype"].kind is Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["alignas"].kind is Parameter.KEYWORD_ONLY
