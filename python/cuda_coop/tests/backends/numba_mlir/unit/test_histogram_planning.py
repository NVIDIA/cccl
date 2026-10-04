# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check histogram validation before provider compilation or execution.

The planner receives Python IR, argument types, and exact launch facts.
Cases separate sample and counter dtype rules, static bin capacity, and the
one-dimensional complete-block requirement. A dynamic bin-count argument
requests literal specialization instead of becoming a runtime control.
"""

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]


def _plan(function, arg_types=(), block=(64, 1, 1)):
    """Run group planning with explicit types and a known launch shape.

    A minimal compiler state supplies function IR and argument types. Launch
    facts let shared planning check block shape and bin capacity without
    provider compilation or a CUDA device.
    """

    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    from cuda.coop.numba_mlir._compiler._group_planner import _GroupCallPlanner

    return _GroupCallPlanner(
        SimpleNamespace(func_ir=run_frontend(function), args=arg_types),
        {"block": block, "grid": (1, 1, 1), "cluster": None},
    ).run()


@pytest.mark.parametrize("dtype", [np.float32, np.int16, np.bool_])
def test_reject_sample_dtype(dtype):
    from cuda import coop

    def kernel():
        samples = coop.ThreadData(items_per_thread=2, dtype=dtype)
        return coop.histogram(coop.this_block(), samples, bins=16)

    with pytest.raises(TypeError, match="samples dtype"):
        _plan(kernel)


@pytest.mark.parametrize("dtype", [np.float32, np.uint8, np.bool_])
def test_reject_counter_dtype(dtype):
    from cuda import coop

    def kernel():
        samples = coop.ThreadData(items_per_thread=2, dtype=np.int32)
        return coop.histogram(
            coop.this_block(), samples, bins=16, counter_dtype=dtype
        )

    with pytest.raises(TypeError, match="counter_dtype"):
        _plan(kernel)


@pytest.mark.parametrize(
    "bins,bins_per_thread", [(0, 1), (True, 1), (129, 2), (2, 0), (2, True)]
)
def test_reject_invalid_capacity(bins, bins_per_thread):
    from cuda import coop

    def kernel():
        samples = coop.ThreadData(items_per_thread=2, dtype=np.int32)
        return coop.histogram(
            coop.this_block(),
            samples,
            bins=bins,
            bins_per_thread=bins_per_thread,
        )

    with pytest.raises((TypeError, ValueError), match="bins"):
        _plan(kernel)


def test_require_literal_bins():
    """Request a literal bin count when an argument lacks a known value.

    The provider specializes its counter storage on bins. ForceLiteralArg
    requests specialization on its value before planning can continue.
    """

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda.core.errors import ForceLiteralArg

    from cuda import coop

    def kernel(bins):
        samples = coop.ThreadData(items_per_thread=2, dtype=np.int32)
        return coop.histogram(coop.this_block(), samples, bins=bins)

    with pytest.raises(ForceLiteralArg):
        _plan(kernel, (types.int32,))


@pytest.mark.parametrize("warp", [False, True])
def test_reject_other_group_topologies(warp):
    from cuda import coop

    group = coop.this_warp if warp else coop.this_block

    def kernel():
        samples = coop.ThreadData(items_per_thread=2, dtype=np.int32)
        return coop.histogram(group(), samples, bins=16)

    with pytest.raises(
        (ValueError, TypeError, NotImplementedError), match="block|dimensional"
    ):
        _plan(kernel, block=(64, 1, 1) if warp else (32, 2, 1))
