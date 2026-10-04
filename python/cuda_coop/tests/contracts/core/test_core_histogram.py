# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check histogram geometry and independent counter-result contracts.

The core specialization builder, which shared planning calls, must reject
Boolean, noninteger, and nonpositive counts, unsupported dtypes, too few
output slots for the bins, sizes beyond signed 32-bit limits, unknown
algorithms, and multidimensional blocks. Planning must also reject warp
groups. A supported plan keeps the counter dtype and bins_per_thread even
when both differ from the sample payload.
"""

import numpy as np
import pytest

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
from cuda.coop._core.block.histogram import make_block_histogram_specialization
from cuda.coop._core.group.histogram import GroupHistogramSemantics


def _specialization(**kwargs):
    options = {
        "sample_dtype": INT32,
        "block_dim": (64, 1, 1),
        "items_per_thread": 3,
        "bins": 65,
        "bins_per_thread": 2,
    }
    options.update(kwargs)
    return make_block_histogram_specialization(**options)


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
        _specialization(**{name: value})


@pytest.mark.parametrize("name", ["sample_dtype", "counter_dtype"])
@pytest.mark.parametrize("dtype", [np.float32, bool, np.int16, np.complex64])
def test_unsupported_dtype(name, dtype):
    with pytest.raises(TypeError, match="dtype"):
        _specialization(**{name: dtype})


@pytest.mark.parametrize("name", ["Int32", "Uint32", "Int64", "Uint64"])
def test_structural_compiler_dtype_names(name):
    """Recognize compiler dtype names without importing a compiler package.

    Named stand-ins exercise the shared specializer's case-insensitive name
    check. The original type objects must remain in its template arguments so
    a backend can resolve them later.
    """

    dtype = type(name, (), {})
    specialization = _specialization(sample_dtype=dtype, counter_dtype=dtype)
    assert specialization.specialization.template_arguments["SampleT"] is dtype
    assert specialization.specialization.template_arguments["CounterT"] is dtype


@pytest.mark.parametrize(
    "name", ["Bool", "Boolean", "Float32", "Float64", "Int16"]
)
@pytest.mark.parametrize("parameter", ["sample_dtype", "counter_dtype"])
def test_unsupported_structural_compiler_dtype_names(name, parameter):
    """Keep named compiler types within each supported sample/counter set."""

    with pytest.raises(TypeError, match="dtype"):
        _specialization(**{parameter: type(name, (), {})})


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
