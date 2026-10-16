# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reject decode type and payload-shape errors during group planning.

Run the frontend with explicit types and launch facts, without compiling a
provider. Lengths and runtime offsets need integer types. Run values and
lengths must have the same fixed extent. Qualified auxiliary buffers must
match their required extent and selected ``decoded_offset_dtype``. The
generated driver checks run contents at execution.
"""

from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]


def _plan(function, args=(), block=(32, 1, 1)):
    """Run group planning with argument types and an exact block shape.

    The minimal compiler state reaches public-call validation. Block size
    selects the decode specialization without compiling a provider.
    """

    from numba_cuda_mlir.numba_cuda.compiler import run_frontend

    from cuda.coop.numba_mlir._compiler._group_planner import _GroupCallPlanner

    return _GroupCallPlanner(
        SimpleNamespace(func_ir=run_frontend(function), args=args),
        {"block": block, "grid": (1, 1, 1), "cluster": None},
    ).run()


@pytest.mark.parametrize("dtype_name", ["boolean", "float32", "complex64"])
def test_lengths_require_integer_dtype(dtype_name):
    from numba_cuda_mlir import types

    from cuda import coop

    dtype = getattr(types, dtype_name)

    def kernel():
        values = coop.ThreadData(items_per_thread=2, dtype=types.int32)
        lengths = coop.ThreadData(items_per_thread=2, dtype=dtype)
        return coop.run_length_decode(
            coop.this_block(), values, lengths, decoded_items_per_thread=4
        )

    with pytest.raises(TypeError, match="integer dtype"):
        _plan(kernel)


@pytest.mark.parametrize("dtype_name", ["boolean", "float64"])
def test_runtime_offsets_require_integer_dtype(dtype_name):
    from numba_cuda_mlir import types

    from cuda import coop

    def kernel(offset):
        values = coop.ThreadData(items_per_thread=2, dtype=types.int32)
        lengths = coop.ThreadData(items_per_thread=2, dtype=types.int32)
        return coop.run_length_decode(
            coop.this_block(),
            values,
            lengths,
            decoded_items_per_thread=4,
            decoded_window_offset=offset,
        )

    with pytest.raises(TypeError, match="integer dtype"):
        _plan(kernel, (getattr(types, dtype_name),))


def test_mismatched_run_payload_extents():
    from numba_cuda_mlir import types

    from cuda import coop

    def kernel():
        values = coop.ThreadData(items_per_thread=2, dtype=types.int32)
        lengths = coop.ThreadData(items_per_thread=3, dtype=types.int32)
        return coop.run_length_decode(
            coop.this_block(), values, lengths, decoded_items_per_thread=4
        )

    with pytest.raises(ValueError, match="matching fixed extents"):
        _plan(kernel)


@pytest.mark.parametrize(
    "aux,extent,dtype",
    [("total", 2, "uint32"), ("relative", 3, "uint32"), ("total", 1, "int32")],
)
def test_auxiliary_output_shape_and_dtype(aux, extent, dtype):
    from numba_cuda_mlir import types

    import cuda.coop.numba_mlir as numba_coop

    dtype = getattr(types, dtype)

    def kernel():
        values = numba_coop.ThreadData(items_per_thread=2, dtype=types.int32)
        lengths = numba_coop.ThreadData(items_per_thread=2, dtype=types.int32)
        output = numba_coop.ThreadData(items_per_thread=extent, dtype=dtype)
        if aux == "total":
            return numba_coop.run_length_decode(
                numba_coop.this_block(),
                values,
                lengths,
                decoded_items_per_thread=4,
                total_decoded_size=output,
            )
        return numba_coop.run_length_decode(
            numba_coop.this_block(),
            values,
            lengths,
            decoded_items_per_thread=4,
            relative_offsets=output,
        )

    with pytest.raises(TypeError, match="extent|dtype"):
        _plan(kernel)
