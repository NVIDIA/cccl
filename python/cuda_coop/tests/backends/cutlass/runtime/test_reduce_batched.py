# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check independent batch reductions and result ownership within warps.

The host reshapes inputs into subgroup, member, and batch axes, then reduces
only the member axis. Device stores collect blocked or striped ownership
into batch order. Partial result slots are skipped, and inputs must remain
unchanged across repeated calls and launches.
"""

from contextlib import ExitStack

import numpy as np
import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import cute

from cuda import coop
from cuda.coop import cutlass as cutlass_coop
from tests.backends.cutlass.support import (
    NUMPY_DTYPES,
    cutlass_dtype,
    device_array,
)

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


def _run(
    *,
    api=coop,
    width=32,
    items_per_thread=3,
    layout="striped",
    dtype=np.int32,
    op="sum",
    block=(8, 4, 2),
    divergent=False,
    payload_kind="thread_data",
    compile_options=(),
):
    """Run one reduction profile and compare it with host results.

    Each thread's item index identifies a separate reduction batch. The host
    uses the input dtype so that NumPy does not widen integer arithmetic.
    The kernel skips result slots that hold no batch in the selected layout.

    Only the first whole subgroup participates in the divergent case. Sibling
    groups must keep their zero output sentinels. Three calls per launch and
    two launches check repeated use and that the input stays unchanged.
    """

    threads = int(np.prod(block))
    output_count = (items_per_thread + width - 1) // width
    value_type = cutlass_dtype(dtype)

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        output: cute.Pointer,
        preserved: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        block_group = api.this_block()
        warp = api.this_warp().group_by(width)
        values = api.ThreadData(items_per_thread)
        source_tensor = cute.recast_tensor(
            cute.make_tensor(
                source, cute.make_layout(threads * items_per_thread)
            ),
            dtype=value_type,
        )
        preserved_tensor = cute.recast_tensor(
            cute.make_tensor(
                preserved, cute.make_layout(threads * items_per_thread)
            ),
            dtype=value_type,
        )
        api.load(block_group, source_tensor, values)
        thread = block_group.rank()
        output_tensor = cute.recast_tensor(
            cute.make_tensor(
                output, cute.make_layout((threads // width) * items_per_thread)
            ),
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
                    if batch < items_per_thread:
                        output_tensor[
                            (thread // width) * items_per_thread + batch
                        ] = result[item]
        if cutlass.const_expr(payload_kind != "thread_data"):
            values = api.ThreadData.from_payload(values)
        api.store(block_group, preserved_tensor, values)

    @cute.jit
    def launch(
        source: cute.Pointer,
        output: cute.Pointer,
        preserved: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        kernel(source, output, preserved, items_per_thread).launch(
            grid=1, block=block
        )

    # Small positive integers keep products exact for the floating-point cases.
    source = (np.arange(threads * items_per_thread) % 2 + 1).astype(dtype)
    output = np.zeros((threads // width) * items_per_thread, dtype=dtype)
    preserved = np.zeros_like(source)
    with ExitStack() as stack:
        pointers = [
            stack.enter_context(device_array(x))
            for x in (source, output, preserved)
        ]
        compiled = cute.compile[compile_options](
            launch, *pointers, items_per_thread
        )
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
        source.reshape(threads // width, width, items_per_thread),
        axis=1,
        dtype=dtype,
    )
    if divergent:
        expected[1:] = 0
    np.testing.assert_array_equal(output, expected.ravel())
    np.testing.assert_array_equal(preserved, source)
    return compiled


@pytest.mark.parametrize(
    "api", (coop, cutlass_coop), ids=("common", "qualified")
)
@pytest.mark.parametrize("width", (1, 2, 4, 8, 16, 32))
@pytest.mark.parametrize("items_per_thread", (1, 4, 33))
@pytest.mark.parametrize("layout", ("striped", "blocked"))
def test_batch_layouts(api, width, items_per_thread, layout):
    _run(api=api, width=width, items_per_thread=items_per_thread, layout=layout)


@pytest.mark.parametrize("dtype", NUMPY_DTYPES)
def test_numeric_types(dtype):
    _run(dtype=dtype, items_per_thread=35, width=8)


@pytest.mark.parametrize(
    "op", ("sum", "multiplies", "min", "max", "bit_and", "bit_or", "bit_xor")
)
def test_builtin_operators(op):
    _run(op=op, width=8, items_per_thread=9)


@pytest.mark.parametrize("width", (1, 2, 4, 8, 16))
def test_independent_divergent_subgroup(width):
    """Let one complete subgroup reduce while sibling groups skip the call.

    The participating lanes form the first logical subgroup. The oracle checks
    its batch results and zero sentinels for every nonparticipating subgroup.
    """

    _run(width=width, divergent=True)


@pytest.mark.parametrize("payload_kind", ("rmem", "ssa"))
def test_qualified_register_payloads(payload_kind):
    _run(api=cutlass_coop, payload_kind=payload_kind, dtype=np.float64)
