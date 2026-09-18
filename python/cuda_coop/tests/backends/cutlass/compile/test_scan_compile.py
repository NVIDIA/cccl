# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Scan compile gates cover typed ABI, prefixes, and storage controls."""

import numpy as np
import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import cute
from cutlass.base_dsl.compiler import GPUArch
from cutlass.cute.runtime import make_ptr

from cuda import coop
from cuda.coop import cutlass as cutlass_coop
from cuda.coop.cutlass._compiler._types import ALL_PROVIDER_TYPES

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.compile]


def _pointer(dtype=cutlass.Int32):
    return make_ptr(dtype, 0, cute.AddressSpace.gmem, assumed_align=16)


@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
@pytest.mark.parametrize("algorithm", ("raking", "raking_memoize", "warp_scans"))
@pytest.mark.parametrize("array", (False, True))
def test_block_forms(api, algorithm, array):
    @cute.kernel
    def kernel(memory: cute.Pointer):
        value = cutlass.Int32(cute.arch.thread_idx()[0])
        if cutlass.const_expr(array):
            payload = api.ThreadData(2, dtype=cutlass.Int32, alignment=64)
            payload[0] = value
            payload[1] = value
            value = payload
        result = api.exclusive_scan(
            api.this_block(),
            value,
            scan_op="max",
            initial_value=-7,
            algorithm=algorithm,
        )
        if cutlass.const_expr(array):
            result = result[0]
        cute.make_tensor(memory, cute.make_layout(1))[0] = result

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=(8, 4, 2))

    assert cute.compile[(GPUArch("sm_80"),)](launch, _pointer()) is not None


@pytest.mark.parametrize("dtype", tuple(ALL_PROVIDER_TYPES))
def test_typed_zero_partial(dtype):
    @cute.kernel
    def kernel(memory: cute.Pointer):
        aggregate = cutlass_coop.ThreadData(1, alignment=64)
        result = cutlass_coop.exclusive_sum(
            cutlass_coop.this_warp().group_by(8),
            dtype(1),
            valid_items=5,
            aggregate_output=aggregate,
        )
        payload = cutlass_coop.ThreadData(1, dtype=dtype, values=[result])
        total = cutlass_coop.sum(cutlass_coop.this_block(), payload)
        cute.make_tensor(memory, cute.make_layout(1))[0] = total + aggregate[0]

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=(8, 4, 2))

    assert cute.compile[(GPUArch("sm_80"),)](launch, _pointer(dtype)) is not None


@pytest.mark.parametrize(
    "dtype", (cutlass.Int32, cutlass.Int64, cutlass.Uint32, cutlass.Uint64)
)
def test_dynamic_prefix(dtype):
    @cute.kernel
    def kernel(memory: cute.Pointer, count: dtype):
        result = cutlass_coop.exclusive_sum(
            cutlass_coop.this_warp().group_by(8), cutlass.Int32(1), valid_items=count
        )
        cute.make_tensor(memory, cute.make_layout(1))[0] = result

    @cute.jit
    def launch(memory: cute.Pointer, count: dtype):
        kernel(memory, count).launch(grid=1, block=(8, 4, 2))

    assert cute.compile[(GPUArch("sm_80"),)](launch, _pointer(), dtype(5)) is not None


@pytest.mark.parametrize("seed", (2, np.float32(2)))
def test_float_seed(seed):
    @cute.kernel
    def kernel(memory: cute.Pointer):
        result = cutlass_coop.exclusive_scan(
            cutlass_coop.this_block(), cutlass.Float32(1), initial_value=seed
        )
        cute.make_tensor(memory, cute.make_layout(1))[0] = result

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=32)

    assert (
        cute.compile[(GPUArch("sm_80"),)](launch, _pointer(cutlass.Float32)) is not None
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
        group = cutlass_coop.this_warp().group_by(8)
        value = cutlass.Int32(1)
        if cutlass.const_expr(case == "zero"):
            cutlass_coop.exclusive_sum(group, value, valid_items=0)
        elif cutlass.const_expr(case == "too_many"):
            cutlass_coop.exclusive_sum(group, value, valid_items=9)
        elif cutlass.const_expr(case == "block_prefix"):
            cutlass_coop.inclusive_sum(cutlass_coop.this_block(), value, valid_items=1)
        elif cutlass.const_expr(case == "warp_array"):
            cutlass_coop.inclusive_sum(
                group, cutlass_coop.ThreadData(1, dtype=cutlass.Int32, values=[value])
            )
        elif cutlass.const_expr(case == "warp_storage"):
            cutlass_coop.exclusive_sum(
                group, value, temp_storage=cutlass_coop.TempStorage()
            )
        elif cutlass.const_expr(case == "warp_algorithm"):
            cutlass_coop.exclusive_sum(group, value, algorithm="raking")
        elif cutlass.const_expr(case == "missing_seed"):
            cutlass_coop.exclusive_scan(group, value, scan_op="max")
        elif cutlass.const_expr(case == "inclusive_seed"):
            cutlass_coop.scan(group, value, mode="inclusive", initial_value=1)
        elif cutlass.const_expr(case == "float_seed"):
            cutlass_coop.exclusive_scan(group, value, initial_value=1.0)
        elif cutlass.const_expr(case == "typed_seed"):
            cutlass_coop.exclusive_scan(group, value, initial_value=cutlass.Int64(1))
        elif cutlass.const_expr(case == "nonfinite_seed"):
            cutlass_coop.exclusive_scan(
                group, cutlass.Float32(1), initial_value=np.float32(np.inf)
            )
        elif cutlass.const_expr(case == "large_seed"):
            cutlass_coop.exclusive_scan(
                group, cutlass.Float32(1), initial_value=1 << 200
            )
        elif cutlass.const_expr(case == "aggregate_extent"):
            cutlass_coop.exclusive_sum(
                group, value, aggregate_output=cutlass_coop.ThreadData(2)
            )
        elif cutlass.const_expr(case == "aggregate_dtype"):
            cutlass_coop.exclusive_sum(
                group,
                value,
                aggregate_output=cutlass_coop.ThreadData(1, dtype=cutlass.Float32),
            )
        elif cutlass.const_expr(case == "aggregate_scalar"):
            cutlass_coop.exclusive_sum(group, value, aggregate_output=value)
        elif cutlass.const_expr(case == "bitwise_float"):
            cutlass_coop.inclusive_scan(group, cutlass.Float32(1), scan_op="bit_and")
        elif cutlass.const_expr(case == "callback"):
            cutlass_coop.inclusive_scan(group, value, scan_op=lambda a, b: a + b)
        else:
            cutlass_coop.exclusive_sum(
                cutlass_coop.this_block(),
                value,
                temp_storage=cutlass_coop.TempStorage(1),
            )

    @cute.jit
    def launch():
        kernel().launch(grid=1, block=64)

    with pytest.raises(Exception, match=expected):
        cute.compile[(GPUArch("sm_80"),)](launch)


@pytest.mark.parametrize("ssa", (False, True))
@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
def test_register_payload(ssa, api):
    @cute.kernel
    def kernel(memory: cute.Pointer):
        fragment = cute.make_rmem_tensor((2,), cutlass.Int32)
        fragment.fill(cutlass.Int32(1))
        if cutlass.const_expr(ssa):
            value = fragment.load()
        else:
            value = fragment
        result = api.inclusive_sum(api.this_block(), value)
        cute.make_tensor(memory, cute.make_layout(1))[0] = result[0]

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=32)

    if api is coop:
        with pytest.raises(Exception, match="scalar or fixed-size ThreadData"):
            cute.compile[(GPUArch("sm_80"),)](launch, _pointer())
    else:
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
            cutlass_coop.inclusive_sum(
                cutlass_coop.this_warp().group_by(8), cutlass.Int32(1)
            )
        elif cutlass.const_expr(case == "warp_scans"):
            cutlass_coop.inclusive_sum(
                cutlass_coop.this_block(), cutlass.Int32(1), algorithm="warp_scans"
            )
        else:
            cutlass_coop.inclusive_sum(cutlass_coop.this_block(), cutlass.Int32(1))

    @cute.jit
    def launch(block_size: cutlass.Int32):
        if cutlass.const_expr(case == "exact"):
            kernel().launch(grid=1, block=block_size)
        else:
            kernel().launch(grid=1, block=48)

    with pytest.raises(Exception, match=expected):
        cute.compile[(GPUArch("sm_80"),)](launch, 48)
