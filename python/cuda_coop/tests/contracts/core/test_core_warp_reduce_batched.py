# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from cuda.coop._core import CxxOperator, Dependency
from cuda.coop._core.warp.reduce_batched import make_warp_reduce_batched_spec


def _spec(**kwargs):
    arguments = dict(
        dtype="int",
        batches=3,
        threads_in_warp=8,
        reduce_operator=CxxOperator("::cuda::std::plus<T>", Dependency("T")),
    )
    arguments.update(kwargs)
    return make_warp_reduce_batched_spec(**arguments)


@pytest.mark.parametrize("width", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("batches", [1, 3, 32, 33, 65])
@pytest.mark.parametrize("layout", ["striped", "blocked"])
def test_batched_output_capacity_and_method(width, batches, layout):
    spec = _spec(threads_in_warp=width, batches=batches, output_layout=layout)
    assert spec.outputs_per_thread * width >= batches
    assert (spec.outputs_per_thread - 1) * width < batches
    assert spec.specialization.struct_name == "WarpReduceBatched"
    assert spec.specialization.method_name == (
        "ReduceToStriped" if layout == "striped" else "ReduceToBlocked"
    )
    assert spec.specialization.template_arguments["SYNC_PHYSICAL_WARP"] == "false"


@pytest.mark.parametrize("width", [0, 3, 33, True, 8.0])
def test_batched_invalid_warp_width(width):
    with pytest.raises(ValueError, match="power of two"):
        _spec(threads_in_warp=width)


@pytest.mark.parametrize("batches", [0, -1, True, 1.5])
def test_batched_requires_positive_static_batches(batches):
    with pytest.raises((TypeError, ValueError), match="batches"):
        _spec(batches=batches)


def test_batched_invalid_layout_and_missing_dtype():
    with pytest.raises(ValueError, match="output_layout"):
        _spec(output_layout="broadcast")
    with pytest.raises(ValueError, match="dtype"):
        _spec(dtype=None)


def test_layout_and_operator_are_part_of_semantic_identity():
    spec = _spec()
    assert spec.call == _spec().call
    assert spec.call != _spec(output_layout="blocked").call
    assert (
        spec.call
        != _spec(
            reduce_operator=CxxOperator("::cuda::maximum<T>", Dependency("T"))
        ).call
    )
    assert len({spec.call, _spec().call, _spec(batches=4).call}) == 2
