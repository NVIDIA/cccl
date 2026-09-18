# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Exchange compile gates cover group ABI, payloads, and ranked controls."""

from enum import Enum

import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import cute
from cutlass.base_dsl.compiler import GPUArch
from cutlass.cute.runtime import make_ptr

from cuda import coop
from cuda.coop import cutlass as cutlass_coop
from cuda.coop.cutlass._compiler._types import ALL_PROVIDER_TYPES, INTEGER_VALUE_TYPES

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.compile]


def _pointer(dtype=cutlass.Int32):
    return make_ptr(dtype, 0, cute.AddressSpace.gmem, assumed_align=16)


@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
@pytest.mark.parametrize("width", (1, 2, 4, 8, 16, 32))
@pytest.mark.parametrize("mode", ("striped_to_blocked", "blocked_to_striped"))
def test_logical_layouts(api, width, mode):
    @cute.kernel
    def kernel(memory: cute.Pointer):
        payload = api.ThreadData(2, dtype=cutlass.Int32, alignment=64)
        payload[0] = cutlass.Int32(1)
        payload[1] = cutlass.Int32(2)
        result = api.exchange(api.this_warp().group_by(width), payload, mode=mode)
        cute.make_tensor(memory, cute.make_layout(1))[0] = result[0]

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=(8, 4, 2))

    assert cute.compile[(GPUArch("sm_80"),)](launch, _pointer()) is not None


@pytest.mark.parametrize("dtype", tuple(ALL_PROVIDER_TYPES))
def test_typed_result(dtype):
    @cute.kernel
    def kernel(memory: cute.Pointer):
        payload = cutlass_coop.ThreadData(2, dtype=dtype, values=[dtype(1), dtype(2)])
        result = cutlass_coop.exchange(cutlass_coop.this_block(), payload)
        total = cutlass_coop.sum(cutlass_coop.this_block(), result)
        cute.make_tensor(memory, cute.make_layout(1))[0] = total

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=(8, 4, 2))

    assert cute.compile[(GPUArch("sm_80"),)](launch, _pointer(dtype)) is not None


@pytest.mark.parametrize(
    "rank_type", (cutlass.Int8, cutlass.Int16, cutlass.Int32, cutlass.Int64)
)
def test_rank_width(rank_type):
    @cute.kernel
    def kernel(memory: cute.Pointer):
        value = cutlass_coop.ThreadData(2, dtype=cutlass.Int32, values=[1, 2])
        ranks = cutlass_coop.ThreadData(
            2, dtype=rank_type, values=[rank_type(0), rank_type(1)]
        )
        result = cutlass_coop.exchange(
            cutlass_coop.this_block(), value, mode="scatter_to_blocked", ranks=ranks
        )
        cute.make_tensor(memory, cute.make_layout(1))[0] = result[0]

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=32)

    assert cute.compile[(GPUArch("sm_80"),)](launch, _pointer()) is not None


@pytest.mark.parametrize("flag_type", tuple(INTEGER_VALUE_TYPES))
def test_flag_width(flag_type):
    @cute.kernel
    def kernel(memory: cute.Pointer):
        value = cutlass_coop.ThreadData(2, dtype=cutlass.Int32, values=[1, 2])
        ranks = cutlass_coop.ThreadData(2, dtype=cutlass.Int32, values=[0, 1])
        flags = cutlass_coop.ThreadData(
            2, dtype=flag_type, values=[flag_type(0), flag_type(2)]
        )
        result = cutlass_coop.exchange(
            cutlass_coop.this_block(),
            value,
            mode="scatter_to_striped_flagged",
            ranks=ranks,
            valid_flags=flags,
        )
        cute.make_tensor(memory, cute.make_layout(1))[0] = result[0]

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=32)

    assert cute.compile[(GPUArch("sm_80"),)](launch, _pointer()) is not None


class _Mode(str, Enum):
    BLOCKED = "striped_to_blocked"


@pytest.mark.parametrize(
    "case, expected",
    (
        ("scalar", "fixed-size ThreadData"),
        ("warp_scatter", "mode for .* groups must be one of"),
        ("missing_ranks", "requires ranks"),
        ("extra_ranks", "does not accept ranks"),
        ("missing_flags", "requires valid_flags"),
        ("extra_flags", "does not accept valid_flags"),
        ("rank_extent", "matching items_per_thread"),
        ("rank_unsigned", "Int8|signed integer"),
        ("flag_float", "Int8|integer"),
        ("flag_bool", "valid_flags.*integer, non-boolean dtype"),
        ("timeslicing_integer", "compile-time bool"),
        ("timeslicing_warp", "only to blocks"),
        ("timeslicing_guarded", "warp_time_slicing is not supported"),
        ("selector_enum", "compile-time string"),
        ("common_scatter", "mode must be one of"),
        ("nested", "nested|unsupported|cannot"),
        ("partial", "complete"),
        ("warp_striped_partial", "multiple of 32"),
        ("overflow", "not representable"),
    ),
)
def test_invalid_controls(case, expected):
    @cute.kernel
    def kernel():
        value = cutlass_coop.ThreadData(2, dtype=cutlass.Int32, values=[1, 2])
        ranks = cutlass_coop.ThreadData(2, dtype=cutlass.Int32, values=[0, 1])
        group = cutlass_coop.this_block()
        if cutlass.const_expr(case == "scalar"):
            cutlass_coop.exchange(group, cutlass.Int32(1))
        elif cutlass.const_expr(case == "warp_scatter"):
            cutlass_coop.exchange(
                cutlass_coop.this_warp(), value, mode="scatter_to_striped", ranks=ranks
            )
        elif cutlass.const_expr(case == "missing_ranks"):
            cutlass_coop.exchange(group, value, mode="scatter_to_blocked")
        elif cutlass.const_expr(case == "extra_ranks"):
            cutlass_coop.exchange(group, value, ranks=ranks)
        elif cutlass.const_expr(case == "missing_flags"):
            cutlass_coop.exchange(
                group, value, mode="scatter_to_striped_flagged", ranks=ranks
            )
        elif cutlass.const_expr(case == "extra_flags"):
            cutlass_coop.exchange(group, value, valid_flags=ranks)
        elif cutlass.const_expr(case == "rank_extent"):
            cutlass_coop.exchange(
                group,
                value,
                mode="scatter_to_blocked",
                ranks=cutlass_coop.ThreadData(1, dtype=cutlass.Int32, values=[0]),
            )
        elif cutlass.const_expr(case == "rank_unsigned"):
            cutlass_coop.exchange(
                group,
                value,
                mode="scatter_to_blocked",
                ranks=cutlass_coop.ThreadData(2, dtype=cutlass.Uint32, values=[0, 1]),
            )
        elif cutlass.const_expr(case == "flag_float"):
            cutlass_coop.exchange(
                group,
                value,
                mode="scatter_to_striped_flagged",
                ranks=ranks,
                valid_flags=cutlass_coop.ThreadData(
                    2, dtype=cutlass.Float32, values=[0.0, 1.0]
                ),
            )
        elif cutlass.const_expr(case == "flag_bool"):
            cutlass_coop.exchange(
                group,
                value,
                mode="scatter_to_striped_flagged",
                ranks=ranks,
                valid_flags=cutlass_coop.ThreadData(
                    2, dtype=bool, values=[True, False]
                ),
            )
        elif cutlass.const_expr(case == "timeslicing_integer"):
            cutlass_coop.exchange(group, value, warp_time_slicing=1)
        elif cutlass.const_expr(case == "timeslicing_warp"):
            cutlass_coop.exchange(
                cutlass_coop.this_warp(), value, warp_time_slicing=True
            )
        elif cutlass.const_expr(case == "timeslicing_guarded"):
            cutlass_coop.exchange(
                group,
                value,
                mode="scatter_to_striped_guarded",
                ranks=ranks,
                warp_time_slicing=True,
            )
        elif cutlass.const_expr(case == "selector_enum"):
            cutlass_coop.exchange(group, value, mode=_Mode.BLOCKED)
        elif cutlass.const_expr(case == "common_scatter"):
            coop.exchange(group, value, mode="scatter_to_blocked")
        elif cutlass.const_expr(case == "nested"):
            cutlass_coop.exchange(
                cutlass_coop.this_warp().group_by(16).group_by(8), value
            )
        elif cutlass.const_expr(case == "partial"):
            cutlass_coop.exchange(cutlass_coop.this_warp().group_by(8), value)
        elif cutlass.const_expr(case == "warp_striped_partial"):
            cutlass_coop.exchange(group, value, mode="blocked_to_warp_striped")
        else:
            cutlass_coop.exchange(
                group, cutlass_coop.ThreadData(2, values=[1 << 40, 2])
            )

    @cute.jit
    def launch():
        if cutlass.const_expr(case in ("partial", "warp_striped_partial")):
            kernel().launch(grid=1, block=48)
        else:
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
        result = api.exchange(api.this_block(), value)
        cute.make_tensor(memory, cute.make_layout(1))[0] = result[0]

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=32)

    if api is coop:
        with pytest.raises(Exception, match="fixed-size ThreadData"):
            cute.compile[(GPUArch("sm_80"),)](launch, _pointer())
    else:
        assert cute.compile[(GPUArch("sm_80"),)](launch, _pointer()) is not None


def test_missing_exact_block():
    @cute.kernel
    def kernel():
        cutlass_coop.exchange(
            cutlass_coop.this_block(),
            cutlass_coop.ThreadData(2, dtype=cutlass.Int32, values=[1, 2]),
        )

    @cute.jit
    def launch(block_size: cutlass.Int32):
        kernel().launch(grid=1, block=block_size)

    with pytest.raises(Exception, match="exact block dimensions"):
        cute.compile[(GPUArch("sm_80"),)](launch, 64)
