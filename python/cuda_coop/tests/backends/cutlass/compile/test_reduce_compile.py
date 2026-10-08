# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile Reduce payloads and controls without executing a kernel.

Typed null pointers and an explicit SM80 target exercise specialization,
provider compilation, and result typing. Runtime tests check result ownership
and numerical behavior.
"""

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


@pytest.mark.parametrize(
    "api", (coop, cutlass_coop), ids=("common", "qualified")
)
@pytest.mark.parametrize(
    "algorithm", (None, "raking_commutative_only", "raking", "warp_reductions")
)
@pytest.mark.parametrize("array", (False, True))
def test_scalar_and_payload(api, algorithm, array):
    @cute.kernel
    def kernel(memory: cute.Pointer, items_per_thread: cutlass.Constexpr):
        value = cutlass.Int32(cute.arch.thread_idx()[0])
        if cutlass.const_expr(array):
            payload = api.ThreadData(items_per_thread, dtype=cutlass.Int32)
            payload[0] = value
            payload[1] = value
            value = payload
        result = api.reduce(
            api.this_block(),
            value,
            binary_op="max",
            algorithm=algorithm,
        )
        if api.this_block().rank() == 0:
            cute.make_tensor(memory, cute.make_layout(1))[0] = result

    @cute.jit
    def launch(memory: cute.Pointer, items_per_thread: cutlass.Constexpr):
        kernel(memory, items_per_thread).launch(grid=1, block=(8, 4, 2))

    assert cute.compile[(GPUArch("sm_80"),)](launch, _pointer(), 2) is not None


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
    """Use a reduction result as typed input to another reduction.

    The first scalar enters a ThreadData payload and then a second collective.
    This checks that the result retains a usable scalar type throughout the
    compiler path, including narrow integers and floating-point values.
    """

    @cute.kernel
    def kernel(memory: cute.Pointer, items_per_thread: cutlass.Constexpr):
        first = cutlass_coop.sum(cutlass_coop.this_block(), dtype(1))
        value = dtype(0)
        if cutlass_coop.this_block().rank() == 0:
            value = first
        payload = cutlass_coop.ThreadData(
            items_per_thread, dtype=dtype, values=[value]
        )
        result = cutlass_coop.reduce(
            cutlass_coop.this_block(), payload, binary_op="max"
        )
        if cutlass_coop.this_block().rank() == 0:
            cute.make_tensor(memory, cute.make_layout(1))[0] = result

    @cute.jit
    def launch(memory: cute.Pointer, items_per_thread: cutlass.Constexpr):
        kernel(memory, items_per_thread).launch(grid=1, block=32)

    assert (
        cute.compile[(GPUArch("sm_80"),)](launch, _pointer(dtype), 1)
        is not None
    )


@pytest.mark.parametrize(
    "dtype", (cutlass.Int32, cutlass.Int64, cutlass.Uint32, cutlass.Uint64)
)
@pytest.mark.parametrize("warp", (False, True))
def test_dynamic_prefix_compile(dtype, warp):
    """Compile prefix counts with signed and unsigned 32- and 64-bit types.

    The count remains a runtime operand for both block and logical-warp
    reductions. This checks the accepted operand types; separate runtime cases
    check values outside the valid range.
    """

    @cute.kernel
    def kernel(memory: cute.Pointer, count: dtype):
        if cutlass.const_expr(warp):
            group = cutlass_coop.this_warp().group_by(8)
        else:
            group = cutlass_coop.this_block()
        result = cutlass_coop.sum(group, cutlass.Int32(1), valid_items=count)
        if group.rank() == 0:
            cute.make_tensor(memory, cute.make_layout(64))[
                cutlass_coop.this_block().rank()
            ] = result

    @cute.jit
    def launch(memory: cute.Pointer, count: dtype):
        kernel(memory, count).launch(grid=1, block=(8, 4, 2))

    assert (
        cute.compile[(GPUArch("sm_80"),)](launch, _pointer(), dtype(5))
        is not None
    )


@pytest.mark.parametrize(
    "case, expected",
    (
        ("zero", "valid_items.*positive integer"),
        ("too_many", "valid_items.*exceeds group size"),
        ("array_prefix", "valid_items is not supported for array inputs"),
        ("broadcast", "unexpected keyword.*broadcast"),
        ("thread", "requires a block, physical warp, or logical warp group"),
        (
            "mapped_warp",
            "requires a block, physical warp, or logical warp group",
        ),
        ("cluster", "requires a block, physical warp, or logical warp group"),
        ("warp_array_prefix", "valid_items is not supported for array inputs"),
        (
            "warp_array_runtime_prefix",
            "valid_items is not supported for array inputs",
        ),
        ("warp_storage", "TempStorage is supported only for block groups"),
        ("non_power_partition", "only one non-power-of-two group"),
        ("undersized_storage", "(?i)(capacity|size|smaller)"),
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
            cutlass_coop.sum(group, value, valid_items=0)
        elif cutlass.const_expr(case == "too_many"):
            cutlass_coop.sum(group, value, valid_items=65)
        elif cutlass.const_expr(case == "array_prefix"):
            cutlass_coop.sum(
                group,
                cutlass_coop.ThreadData(
                    items_per_thread=1, dtype=cutlass.Int32, values=[value]
                ),
                valid_items=1,
            )
        elif cutlass.const_expr(case == "broadcast"):
            cutlass_coop.sum(group, value, broadcast=True)
        elif cutlass.const_expr(case == "thread"):
            cutlass_coop.sum(cutlass_coop.this_thread(), value)
        elif cutlass.const_expr(case == "mapped_warp"):
            cutlass_coop.sum(group.group_by(2), value)
        elif cutlass.const_expr(case == "cluster"):
            cutlass_coop.sum(cutlass_coop.this_cluster(), value)
        elif cutlass.const_expr(
            case in ("warp_array_prefix", "warp_array_runtime_prefix")
        ):
            count = 1
            if cutlass.const_expr(case == "warp_array_runtime_prefix"):
                count = cutlass.Int32(1)
            cutlass_coop.sum(
                cutlass_coop.this_warp(),
                cutlass_coop.ThreadData(
                    items_per_thread=1, dtype=cutlass.Int32, values=[value]
                ),
                valid_items=count,
            )
        elif cutlass.const_expr(case == "non_power_partition"):
            lanes = cutlass_coop.this_warp().group_by(12, exhaustive=False)
            if lanes.is_member():
                cutlass_coop.sum(lanes, value)
        elif cutlass.const_expr(case == "warp_storage"):
            cutlass_coop.sum(
                cutlass_coop.this_warp(),
                value,
                temp_storage=cutlass_coop.TempStorage(),
            )
        elif cutlass.const_expr(case == "undersized_storage"):
            cutlass_coop.sum(
                group,
                value,
                temp_storage=cutlass_coop.TempStorage(1),
            )
        elif cutlass.const_expr(case == "warp_algorithm"):
            cutlass_coop.sum(
                cutlass_coop.this_warp(),
                value,
                algorithm="raking",
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
@pytest.mark.parametrize(
    "api", (coop, cutlass_coop), ids=("common", "qualified")
)
def test_register_payload_boundary(ssa, api):
    """Keep native CuTe register payloads behind the qualified API.

    The same register tensor and its loaded SSA value are accepted through the
    CUTLASS API. The common API requires a scalar or ThreadData payload, so it
    must reject both native forms before provider compilation can use them.
    """

    @cute.kernel
    def kernel(memory: cute.Pointer):
        fragment = cute.make_rmem_tensor((2,), cutlass.Int32)
        fragment.fill(cutlass.Int32(1))
        if cutlass.const_expr(ssa):
            value = fragment.load()
        else:
            value = fragment
        result = api.sum(api.this_block(), value)
        if api.this_block().rank() == 0:
            cute.make_tensor(memory, cute.make_layout(1))[0] = result

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=32)

    if api is coop:
        with pytest.raises(Exception, match="scalar or fixed-size ThreadData"):
            cute.compile[(GPUArch("sm_80"),)](launch, _pointer())
    else:
        assert cute.compile[(GPUArch("sm_80"),)](launch, _pointer()) is not None


@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize("width", [8, 32])
@pytest.mark.parametrize(
    "form", ["thread_data", "register_tensor", "tensor_ssa"]
)
def test_warp_payload_forms(items_per_thread, width, form):
    """Compile CUB warp-array overloads and qualified register conversions."""

    @cute.kernel
    def kernel(memory: cute.Pointer, items_per_thread: cutlass.Constexpr):
        group = cutlass_coop.this_warp()
        if cutlass.const_expr(width != 32):
            group = group.group_by(width)
        payload = cutlass_coop.ThreadData(items_per_thread)
        cutlass_coop.load(group, memory, payload)
        if cutlass.const_expr(form == "register_tensor"):
            values = payload.to_register_tensor()
        elif cutlass.const_expr(form == "tensor_ssa"):
            values = payload.to_tensor_ssa()
        else:
            values = payload
        total = cutlass_coop.sum(group, values)
        largest = cutlass_coop.reduce(group, values, binary_op="max")
        if group.rank() == 0:
            cute.make_tensor(memory, cute.make_layout(64))[
                cutlass_coop.this_block().rank()
            ] = total + largest

    @cute.jit
    def launch(memory: cute.Pointer, items_per_thread: cutlass.Constexpr):
        kernel(memory, items_per_thread).launch(grid=1, block=64)

    assert (
        cute.compile[(GPUArch("sm_80"),)](launch, _pointer(), items_per_thread)
        is not None
    )
