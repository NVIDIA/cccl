# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check native batched-reduction geometry, method choice, and identity.

Output capacity is the smallest per-lane extent that holds all batch results.
Layout selects a CUB method, and synchronization is restricted to the logical
warp so neighboring groups can skip the call. Operator, layout, and batch
count must remain part of specialization identity. Construction rejects warp
widths that are not a power of two up to 32, batch counts that are not
positive integers, unknown layouts, and a missing dtype.
"""

import pytest

from cuda.coop._core import CxxOperator, Dependency
from cuda.coop._core.warp.reduce_batched import (
    make_warp_reduce_batched_specialization,
)


def _specialization(**kwargs):
    arguments = {
        "dtype": "int",
        "batches": 3,
        "threads_in_warp": 8,
        "reduce_operator": CxxOperator("::cuda::std::plus<T>", Dependency("T")),
    }
    arguments.update(kwargs)
    return make_warp_reduce_batched_specialization(**arguments)


@pytest.mark.parametrize("width", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("batches", [1, 3, 32, 33, 65])
@pytest.mark.parametrize("layout", ["striped", "blocked"])
def test_batched_output_capacity_and_method(width, batches, layout):
    specialization = _specialization(
        threads_in_warp=width, batches=batches, output_layout=layout
    )
    assert specialization.outputs_per_thread * width >= batches
    assert (specialization.outputs_per_thread - 1) * width < batches
    assert specialization.specialization.struct_name == "WarpReduceBatched"
    assert specialization.specialization.method_name == (
        "ReduceToStriped" if layout == "striped" else "ReduceToBlocked"
    )
    assert (
        specialization.specialization.template_arguments["SYNC_PHYSICAL_WARP"]
        == "false"
    )


@pytest.mark.parametrize("width", [0, 3, 33, True, 8.0])
def test_batched_invalid_warp_width(width):
    with pytest.raises(ValueError, match="power of two"):
        _specialization(threads_in_warp=width)


@pytest.mark.parametrize("batches", [0, -1, True, 1.5])
def test_batched_requires_positive_static_batches(batches):
    with pytest.raises((TypeError, ValueError), match="batches"):
        _specialization(batches=batches)


def test_batched_invalid_layout_and_missing_dtype():
    with pytest.raises(ValueError, match="output_layout"):
        _specialization(output_layout="broadcast")
    with pytest.raises(ValueError, match="dtype"):
        _specialization(dtype=None)


def test_layout_and_operator_are_part_of_semantic_identity():
    specialization = _specialization()
    assert specialization.call == _specialization().call
    assert specialization.call != _specialization(output_layout="blocked").call
    assert (
        specialization.call
        != _specialization(
            reduce_operator=CxxOperator("::cuda::maximum<T>", Dependency("T"))
        ).call
    )
    assert (
        len(
            {
                specialization.call,
                _specialization().call,
                _specialization(batches=4).call,
            }
        )
        == 2
    )
