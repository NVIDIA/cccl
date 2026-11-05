# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile Scan types and controls without executing the generated kernels.

Typed null pointers and an explicit SM80 target exercise specialization and
provider compilation. Unconditional stores and use of results outside a
valid prefix are compile inputs only. Runtime tests separately check which
lanes may consume a result and whether its value is correct.
"""

import cutlass
import numpy as np
import pytest
from cutlass import cute
from cutlass.base_dsl.compiler import GPUArch
from cutlass.cute.runtime import make_ptr

import cuda.coop.cutlass as coop
from cuda.coop.cutlass._compiler._types import ALL_PROVIDER_TYPES

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.compile]


def _pointer(dtype=cutlass.Int32):
    return make_ptr(dtype, 0, cute.AddressSpace.gmem, assumed_align=16)


@pytest.mark.parametrize(
    "algorithm", ("raking", "raking_memoize", "warp_scans")
)
@pytest.mark.parametrize("array", (False, True))
def test_block_forms(algorithm, array):
    @cute.kernel
    def kernel(memory: cute.Pointer, items_per_thread: cutlass.Constexpr):
        value = cutlass.Int32(cute.arch.thread_idx()[0])
        if cutlass.const_expr(array):
            payload = coop.ThreadData(
                items_per_thread, dtype=cutlass.Int32, alignment=64
            )
            payload[0] = value
            payload[1] = value
            value = payload
        result = coop.exclusive_scan(
            coop.this_block(),
            value,
            scan_op="max",
            initial_value=-7,
            algorithm=algorithm,
        )
        if cutlass.const_expr(array):
            result = result[0]
        cute.make_tensor(memory, cute.make_layout(1))[0] = result

    @cute.jit
    def launch(memory: cute.Pointer, items_per_thread: cutlass.Constexpr):
        kernel(memory, items_per_thread).launch(grid=1, block=(8, 4, 2))

    assert cute.compile[(GPUArch("sm_80"),)](launch, _pointer(), 2) is not None


@pytest.mark.parametrize("dtype", tuple(ALL_PROVIDER_TYPES))
def test_typed_zero_partial(dtype):
    """Propagate types from an inferred zero seed and a partial aggregate.

    Each type uses a partial logical-warp ``exclusive_sum``. Its result feeds
    ThreadData and a block ``sum``. The initially untyped aggregate also
    acquires the input dtype, checking type propagation across both outputs.
    """

    @cute.kernel
    def kernel(memory: cute.Pointer, items_per_thread: cutlass.Constexpr):
        aggregate = coop.ThreadData(items_per_thread, alignment=64)
        result = coop.exclusive_sum(
            coop.this_warp().group_by(8),
            dtype(1),
            valid_items=5,
            aggregate_output=aggregate,
        )
        payload = coop.ThreadData(
            items_per_thread, dtype=dtype, values=[result]
        )
        total = coop.sum(coop.this_block(), payload)
        cute.make_tensor(memory, cute.make_layout(1))[0] = total + aggregate[0]

    @cute.jit
    def launch(memory: cute.Pointer, items_per_thread: cutlass.Constexpr):
        kernel(memory, items_per_thread).launch(grid=1, block=(8, 4, 2))

    assert (
        cute.compile[(GPUArch("sm_80"),)](launch, _pointer(dtype), 1)
        is not None
    )


@pytest.mark.parametrize(
    "dtype", (cutlass.Int32, cutlass.Int64, cutlass.Uint32, cutlass.Uint64)
)
def test_dynamic_prefix(dtype):
    """Accept runtime prefix counts in signed and unsigned wide integer types.

    Both 32- and 64-bit operands must compile for the logical-warp path. The
    runtime trap tests separately check values outside the valid range.
    """

    @cute.kernel
    def kernel(memory: cute.Pointer, count: dtype):
        result = coop.exclusive_sum(
            coop.this_warp().group_by(8),
            cutlass.Int32(1),
            valid_items=count,
        )
        cute.make_tensor(memory, cute.make_layout(1))[0] = result

    @cute.jit
    def launch(memory: cute.Pointer, count: dtype):
        kernel(memory, count).launch(grid=1, block=(8, 4, 2))

    assert (
        cute.compile[(GPUArch("sm_80"),)](launch, _pointer(), dtype(5))
        is not None
    )


@pytest.mark.parametrize("seed", (2, np.float32(2)))
def test_float_seed(seed):
    @cute.kernel
    def kernel(memory: cute.Pointer):
        result = coop.exclusive_scan(
            coop.this_block(), cutlass.Float32(1), initial_value=seed
        )
        cute.make_tensor(memory, cute.make_layout(1))[0] = result

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=32)

    assert (
        cute.compile[(GPUArch("sm_80"),)](launch, _pointer(cutlass.Float32))
        is not None
    )


@pytest.mark.parametrize(
    "case, expected",
    (
        ("zero", "valid_items must be between 1"),
        ("too_many", "valid_items must be between 1"),
        ("block_prefix", "valid_items"),
        ("warp_array", "scalar"),
        ("warp_storage", "temp_storage.*blocks"),
        ("warp_algorithm", "algorithm.*block"),
        ("missing_seed", "requires initial_value"),
        ("inclusive_seed", "does not accept initial_value"),
        ("float_seed", "dtype does not match"),
        ("typed_seed", "dtype must match"),
        ("nonfinite_seed", "finite"),
        ("large_seed", "not representable"),
        ("aggregate_extent", "aggregate_output.*one item"),
        ("aggregate_dtype", "aggregate_output.*dtype must match"),
        ("aggregate_scalar", "aggregate_output must be ThreadData"),
        ("bitwise_float", "integer dtype"),
        ("callback", "custom callbacks"),
        ("small_storage", "TempStorage capacity is smaller"),
    ),
)
def test_invalid_controls(case, expected):
    @cute.kernel
    def kernel():
        group = coop.this_warp().group_by(8)
        value = cutlass.Int32(1)
        if cutlass.const_expr(case == "zero"):
            coop.exclusive_sum(group, value, valid_items=0)
        elif cutlass.const_expr(case == "too_many"):
            coop.exclusive_sum(group, value, valid_items=9)
        elif cutlass.const_expr(case == "block_prefix"):
            coop.inclusive_sum(coop.this_block(), value, valid_items=1)
        elif cutlass.const_expr(case == "warp_array"):
            coop.inclusive_sum(
                group,
                coop.ThreadData(
                    items_per_thread=1, dtype=cutlass.Int32, values=[value]
                ),
            )
        elif cutlass.const_expr(case == "warp_storage"):
            coop.exclusive_sum(group, value, temp_storage=coop.TempStorage())
        elif cutlass.const_expr(case == "warp_algorithm"):
            coop.exclusive_sum(group, value, algorithm="raking")
        elif cutlass.const_expr(case == "missing_seed"):
            coop.exclusive_scan(group, value, scan_op="max")
        elif cutlass.const_expr(case == "inclusive_seed"):
            coop.scan(group, value, mode="inclusive", initial_value=1)
        elif cutlass.const_expr(case == "float_seed"):
            coop.exclusive_scan(group, value, initial_value=1.0)
        elif cutlass.const_expr(case == "typed_seed"):
            coop.exclusive_scan(group, value, initial_value=cutlass.Int64(1))
        elif cutlass.const_expr(case == "nonfinite_seed"):
            coop.exclusive_scan(
                group, cutlass.Float32(1), initial_value=np.float32(np.inf)
            )
        elif cutlass.const_expr(case == "large_seed"):
            coop.exclusive_scan(
                group, cutlass.Float32(1), initial_value=1 << 200
            )
        elif cutlass.const_expr(case == "aggregate_extent"):
            coop.exclusive_sum(
                group,
                value,
                aggregate_output=coop.ThreadData(items_per_thread=2),
            )
        elif cutlass.const_expr(case == "aggregate_dtype"):
            coop.exclusive_sum(
                group,
                value,
                aggregate_output=coop.ThreadData(
                    items_per_thread=1, dtype=cutlass.Float32
                ),
            )
        elif cutlass.const_expr(case == "aggregate_scalar"):
            coop.exclusive_sum(group, value, aggregate_output=value)
        elif cutlass.const_expr(case == "bitwise_float"):
            coop.inclusive_scan(group, cutlass.Float32(1), scan_op="bit_and")
        elif cutlass.const_expr(case == "callback"):
            coop.inclusive_scan(group, value, scan_op=lambda a, b: a + b)
        else:
            coop.exclusive_sum(
                coop.this_block(),
                value,
                temp_storage=coop.TempStorage(1),
            )

    @cute.jit
    def launch():
        kernel().launch(grid=1, block=64)

    with pytest.raises(Exception, match=expected):
        cute.compile[(GPUArch("sm_80"),)](launch)


@pytest.mark.parametrize("ssa", (False, True))
def test_register_payload(ssa):
    """Compile indexed Scan results from rmem and TensorSSA inputs."""

    @cute.kernel
    def kernel(memory: cute.Pointer):
        fragment = cute.make_rmem_tensor((2,), cutlass.Int32)
        fragment.fill(cutlass.Int32(1))
        if cutlass.const_expr(ssa):
            value = fragment.load()
        else:
            value = fragment
        result = coop.inclusive_sum(coop.this_block(), value)
        cute.make_tensor(memory, cute.make_layout(1))[0] = result[0]

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=32)

    assert cute.compile[(GPUArch("sm_80"),)](launch, _pointer()) is not None


@pytest.mark.parametrize(
    "case, expected",
    (
        ("warp_scans", "complete|multiple"),
        ("logical", "complete"),
        ("exact", "exact block dimensions"),
    ),
)
def test_launch_requirements(case, expected):
    @cute.kernel
    def kernel():
        if cutlass.const_expr(case == "logical"):
            coop.inclusive_sum(coop.this_warp().group_by(8), cutlass.Int32(1))
        elif cutlass.const_expr(case == "warp_scans"):
            coop.inclusive_sum(
                coop.this_block(),
                cutlass.Int32(1),
                algorithm="warp_scans",
            )
        else:
            coop.inclusive_sum(coop.this_block(), cutlass.Int32(1))

    @cute.jit
    def launch(block_size: cutlass.Int32):
        if cutlass.const_expr(case == "exact"):
            kernel().launch(grid=1, block=block_size)
        else:
            kernel().launch(grid=1, block=48)

    with pytest.raises(Exception, match=expected):
        cute.compile[(GPUArch("sm_80"),)](launch, 48)
