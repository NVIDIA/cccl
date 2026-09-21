# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Independent batch/layout oracles, input preservation, and subgroup calls."""

from contextlib import ExitStack

import numpy as np
import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import cute

from cuda import coop
from cuda.coop import cutlass as cutlass_coop
from tests.backends.cutlass.support import NUMPY_DTYPES, cutlass_dtype, device_array

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


def _run(
    *,
    api=coop,
    width=32,
    batches=3,
    layout="striped",
    dtype=np.int32,
    op="sum",
    block=(8, 4, 2),
    divergent=False,
    payload_kind="thread_data",
    compile_options=(),
):
    threads = int(np.prod(block))
    output_count = (batches + width - 1) // width
    value_type = cutlass_dtype(dtype)

    @cute.kernel
    def kernel(source: cute.Pointer, output: cute.Pointer, preserved: cute.Pointer):
        block_group = api.this_block()
        warp = api.this_warp().group_by(width)
        values = api.ThreadData(batches)
        source_tensor = cute.recast_tensor(
            cute.make_tensor(source, cute.make_layout(threads * batches)),
            dtype=value_type,
        )
        preserved_tensor = cute.recast_tensor(
            cute.make_tensor(preserved, cute.make_layout(threads * batches)),
            dtype=value_type,
        )
        api.load(block_group, source_tensor, values)
        thread = block_group.rank()
        output_tensor = cute.recast_tensor(
            cute.make_tensor(output, cute.make_layout((threads // width) * batches)),
            dtype=value_type,
        )
        if cutlass.const_expr(payload_kind == "rmem"):
            values = values.to_register_tensor()
        elif cutlass.const_expr(payload_kind == "ssa"):
            values = values.to_tensor_ssa()
        # Repeated calls also exercise request deduplication and live registers.
        for _ in range(3):
            if not divergent or thread < width:
                result = api.reduce_batched(
                    warp, values, binary_op=op, output_layout=layout
                )
                assert result.dtype is value_type
                lane = thread % width
                for item in cutlass.range_constexpr(output_count):
                    if cutlass.const_expr(layout == "striped"):
                        batch = lane + item * width
                    else:
                        batch = lane * output_count + item
                    if batch < batches:
                        output_tensor[(thread // width) * batches + batch] = result[
                            item
                        ]
        if cutlass.const_expr(payload_kind != "thread_data"):
            values = api.ThreadData.from_payload(values)
        api.store(block_group, preserved_tensor, values)

    @cute.jit
    def launch(source: cute.Pointer, output: cute.Pointer, preserved: cute.Pointer):
        kernel(source, output, preserved).launch(grid=1, block=block)

    # Small positive integers keep products exact for the floating-point cases.
    source = (np.arange(threads * batches) % 2 + 1).astype(dtype)
    output = np.zeros((threads // width) * batches, dtype=dtype)
    preserved = np.zeros_like(source)
    with ExitStack() as stack:
        pointers = [
            stack.enter_context(device_array(x)) for x in (source, output, preserved)
        ]
        compiled = cute.compile[compile_options](launch, *pointers)
        compiled(*pointers)
        compiled(*pointers)
    reducer = {
        "sum": np.add,
        "multiplies": np.multiply,
        "min": np.minimum,
        "max": np.maximum,
        "bit_and": np.bitwise_and,
        "bit_or": np.bitwise_or,
        "bit_xor": np.bitwise_xor,
    }[op]
    expected = reducer.reduce(
        source.reshape(threads // width, width, batches), axis=1, dtype=dtype
    )
    if divergent:
        expected[1:] = 0
    np.testing.assert_array_equal(output, expected.ravel())
    np.testing.assert_array_equal(preserved, source)
    return compiled


@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
@pytest.mark.parametrize("width", (1, 2, 4, 8, 16, 32))
@pytest.mark.parametrize("batches", (3, 33))
@pytest.mark.parametrize("layout", ("striped", "blocked"))
def test_batch_layouts(api, width, batches, layout):
    _run(api=api, width=width, batches=batches, layout=layout)


@pytest.mark.parametrize("dtype", NUMPY_DTYPES)
def test_numeric_types(dtype):
    _run(dtype=dtype, batches=35, width=8)


@pytest.mark.parametrize(
    "op", ("sum", "multiplies", "min", "max", "bit_and", "bit_or", "bit_xor")
)
def test_builtin_operators(op):
    _run(op=op, width=8, batches=9)


@pytest.mark.parametrize("width", (1, 2, 4, 8, 16))
def test_independent_divergent_subgroup(width):
    _run(width=width, divergent=True)


@pytest.mark.parametrize("payload_kind", ("rmem", "ssa"))
def test_qualified_register_payloads(payload_kind):
    _run(api=cutlass_coop, payload_kind=payload_kind, dtype=np.float64)
