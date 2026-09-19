# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Histogram type, capacity, ownership, and import-light API contracts."""

from importlib import import_module

import numpy as np
import pytest

from cuda import coop
from cuda.coop._core import (
    INT32,
    UINT64,
    LaunchFacts,
    ResultVisibility,
    make_group_primitive_call,
    plan_group_primitive,
    this_block,
    this_warp,
)
from cuda.coop._core.block.histogram import make_block_histogram_spec
from cuda.coop._core.group.histogram import GroupHistogramSemantics


def _spec(**kwargs):
    options = dict(
        sample_dtype=INT32,
        block_dim=(64, 1, 1),
        items_per_thread=3,
        bins=65,
        bins_per_thread=2,
    )
    options.update(kwargs)
    return make_block_histogram_spec(**options)


@pytest.mark.parametrize(
    "name,value",
    [
        ("bins", 0),
        ("bins", True),
        ("bins", 2.5),
        ("bins_per_thread", 0),
        ("bins_per_thread", True),
        ("items_per_thread", 0),
        ("bins", 129),
        ("bins_per_thread", 2**31),
        ("items_per_thread", 2**31),
        ("block_dim", (32, 2, 1)),
        ("algorithm", "other"),
    ],
)
def test_invalid_static_contracts(name, value):
    with pytest.raises((ValueError, TypeError)):
        _spec(**{name: value})


@pytest.mark.parametrize("name", ["sample_dtype", "counter_dtype"])
@pytest.mark.parametrize("dtype", [np.float32, bool, np.int16, np.complex64])
def test_unsupported_dtype(name, dtype):
    with pytest.raises(TypeError, match="dtype"):
        _spec(**{name: dtype})


def test_result_extent_and_dtype_are_independent_of_samples():
    operation = GroupHistogramSemantics(INT32, 3, 65, 2, UINT64, "sort")
    plan = plan_group_primitive(
        make_group_primitive_call(this_block(), operation), LaunchFacts(64)
    ).require_supported()
    result = plan.result.values[0]
    assert result.dtype == UINT64
    assert result.items_per_member == 2
    assert result.visibility is ResultVisibility.PER_MEMBER
    assert plan.provenance.cpp_class == "cub::BlockHistogram"
    assert operation != GroupHistogramSemantics(INT32, 3, 65, 1, UINT64, "sort")


def test_warp_is_unsupported():
    operation = GroupHistogramSemantics(INT32, 1, 32)
    with pytest.raises((ValueError, NotImplementedError), match="block"):
        plan_group_primitive(
            make_group_primitive_call(this_warp(), operation), LaunchFacts(64)
        ).require_supported()


class _ReadonlySamples:
    items_per_thread = 3
    dtype = np.uint8

    def __len__(self):
        return 3

    def __getitem__(self, index):
        return (1, 2, 1)[index]


def test_common_readonly_inputs_and_typed_controls(monkeypatch):
    api = import_module("cuda.coop._core.api.histogram")
    dispatch = import_module("cuda.coop._core.api._dispatch")
    samples = _ReadonlySamples()
    sentinel = object()
    monkeypatch.setattr(api, "_group_primitive_marker", lambda *a, **k: sentinel)
    with dispatch._compiler_scope("test.backend"):
        assert (
            coop.histogram(this_block(), samples, bins=4, counter_dtype=np.int64)
            is sentinel
        )
        with pytest.raises(TypeError, match="ThreadData"):
            coop.histogram(this_block(), 1, bins=4)
    assert list(samples) == [1, 2, 1]
