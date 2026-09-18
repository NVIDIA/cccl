# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Built-in Reduce traces validate routing, payloads, and integer controls."""

import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import cute
from cutlass.base_dsl.compiler import GPUArch
from cutlass.cute.runtime import make_ptr

from cuda import coop
from cuda.coop import cutlass as cutlass_coop

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.compile]


def _pointer(dtype=cutlass.Int32):
    return make_ptr(dtype, 0, cute.AddressSpace.gmem, assumed_align=16)


@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
@pytest.mark.parametrize(
    "algorithm", (None, "raking_commutative_only", "raking", "warp_reductions")
)
@pytest.mark.parametrize("array", (False, True))
def test_scalar_and_payload(api, algorithm, array):
    @cute.kernel
    def kernel(memory: cute.Pointer):
        value = cutlass.Int32(cute.arch.thread_idx()[0])
        if cutlass.const_expr(array):
            payload = api.ThreadData(2, dtype=cutlass.Int32)
            payload[0] = value
            payload[1] = value
            value = payload
        result = api.reduce(
            api.this_block(),
            value,
            binary_op="max",
            broadcast=False,
            algorithm=algorithm,
        )
        cute.make_tensor(memory, cute.make_layout(1))[0] = result

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=(8, 4, 2))

    assert cute.compile[(GPUArch("sm_80"),)](launch, _pointer()) is not None


@pytest.mark.parametrize(
    "dtype",
    (
        cutlass.Int8,
        cutlass.Uint8,
        cutlass.Int16,
        cutlass.Uint16,
        cutlass.Int32,
        cutlass.Uint32,
        cutlass.Int64,
        cutlass.Uint64,
        cutlass.Float32,
        cutlass.Float64,
    ),
)
def test_typed_result_consumption(dtype):
    @cute.kernel
    def kernel(memory: cute.Pointer):
        first = cutlass_coop.sum(cutlass_coop.this_block(), dtype(1))
        payload = cutlass_coop.ThreadData(1, dtype=dtype, values=[first])
        result = cutlass_coop.reduce(
            cutlass_coop.this_block(), payload, binary_op="max"
        )
        cute.make_tensor(memory, cute.make_layout(1))[0] = result

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=32)

    assert cute.compile[(GPUArch("sm_80"),)](launch, _pointer(dtype)) is not None


@pytest.mark.parametrize(
    "dtype", (cutlass.Int32, cutlass.Int64, cutlass.Uint32, cutlass.Uint64)
)
@pytest.mark.parametrize("warp", (False, True))
def test_dynamic_prefix_compile(dtype, warp):
    @cute.kernel
    def kernel(memory: cute.Pointer, count: dtype):
        if cutlass.const_expr(warp):
            group = cutlass_coop.this_warp().group_by(8)
        else:
            group = cutlass_coop.this_block()
        result = cutlass_coop.sum(
            group, cutlass.Int32(1), valid_items=count, broadcast=False
        )
        cute.make_tensor(memory, cute.make_layout(1))[0] = result

    @cute.jit
    def launch(memory: cute.Pointer, count: dtype):
        kernel(memory, count).launch(grid=1, block=(8, 4, 2))

    assert cute.compile[(GPUArch("sm_80"),)](launch, _pointer(), dtype(5)) is not None


@pytest.mark.parametrize(
    "case, expected",
    (
        ("zero", "valid_items.*positive integer"),
        ("too_many", "valid_items.*exceeds group size"),
        ("array_prefix", "valid_items is not supported for array inputs"),
        ("broadcast_prefix", "broadcast=True|broadcast=False"),
        ("warp_algorithm", "BlockReduce|block group"),
        ("bitwise_float", "integer dtype"),
        ("callback", "custom callbacks"),
        ("nondeterministic", "algorithm must be one of"),
        ("literal_overflow", "not representable"),
    ),
)
def test_invalid_controls(case, expected):
    @cute.kernel
    def kernel():
        group = cutlass_coop.this_block()
        value = cutlass.Int32(1)
        if cutlass.const_expr(case == "zero"):
            cutlass_coop.sum(group, value, broadcast=False, valid_items=0)
        elif cutlass.const_expr(case == "too_many"):
            cutlass_coop.sum(group, value, broadcast=False, valid_items=65)
        elif cutlass.const_expr(case == "array_prefix"):
            cutlass_coop.sum(
                group,
                cutlass_coop.ThreadData(1, dtype=cutlass.Int32, values=[value]),
                broadcast=False,
                valid_items=1,
            )
        elif cutlass.const_expr(case == "broadcast_prefix"):
            cutlass_coop.sum(group, value, valid_items=1)
        elif cutlass.const_expr(case == "warp_algorithm"):
            cutlass_coop.sum(
                cutlass_coop.this_warp(), value, broadcast=False, algorithm="raking"
            )
        elif cutlass.const_expr(case == "bitwise_float"):
            cutlass_coop.reduce(group, cutlass.Float32(1), binary_op="bit_and")
        elif cutlass.const_expr(case == "callback"):
            cutlass_coop.reduce(group, value, binary_op=lambda a, b: a + b)
        elif cutlass.const_expr(case == "literal_overflow"):
            cutlass_coop.sum(group, 1 << 40)
        else:
            cutlass_coop.sum(
                group,
                value,
                broadcast=False,
                algorithm="warp_reductions_nondeterministic",
            )

    @cute.jit
    def launch():
        kernel().launch(grid=1, block=64)

    with pytest.raises(Exception, match=expected):
        cute.compile[(GPUArch("sm_80"),)](launch)


def test_missing_exact_block():
    @cute.kernel
    def kernel():
        cutlass_coop.sum(cutlass_coop.this_block(), cutlass.Int32(1))

    @cute.jit
    def launch(block_size: cutlass.Int32):
        kernel().launch(grid=1, block=(block_size, 1, 1))

    with pytest.raises(Exception, match="exact block dimensions"):
        cute.compile[(GPUArch("sm_80"),)](launch, 64)


@pytest.mark.parametrize("ssa", (False, True))
@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
def test_register_payload_boundary(ssa, api):
    @cute.kernel
    def kernel(memory: cute.Pointer):
        fragment = cute.make_rmem_tensor((2,), cutlass.Int32)
        fragment.fill(cutlass.Int32(1))
        if cutlass.const_expr(ssa):
            value = fragment.load()
        else:
            value = fragment
        result = api.sum(api.this_block(), value)
        cute.make_tensor(memory, cute.make_layout(1))[0] = result

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=32)

    if api is coop:
        with pytest.raises(Exception, match="scalar or fixed-size ThreadData"):
            cute.compile[(GPUArch("sm_80"),)](launch, _pointer())
    else:
        assert cute.compile[(GPUArch("sm_80"),)](launch, _pointer()) is not None
