# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Common-profile diagnostics and qualified register input compilation."""

import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import cute
from cutlass.base_dsl.common import DSLRuntimeError
from cutlass.base_dsl.compiler import GPUArch
from cutlass.cute.runtime import make_ptr

from cuda import coop
from cuda.coop import cutlass as cutlass_coop

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.compile]


def _compile(api=coop, *, block=64, bad=None, width=8):
    @cute.kernel
    def kernel(memory: cute.Pointer):
        payload = api.ThreadData(3, dtype=cutlass.Int32)
        payload[0], payload[1], payload[2] = (
            cutlass.Int32(1),
            cutlass.Int32(2),
            cutlass.Int32(3),
        )
        if cutlass.const_expr(bad == "scalar"):
            payload = cutlass.Int32(1)
        elif cutlass.const_expr(bad == "register"):
            payload = cute.make_rmem_tensor(3, cutlass.Int32)
            payload.fill(cutlass.Int32(1))
        group = api.this_warp().group_by(width)
        if cutlass.const_expr(bad == "group"):
            group = api.this_block()
        operator = "sum"
        if cutlass.const_expr(bad == "operator"):
            operator = "divide"
        layout = "diagonal" if bad == "layout" else "striped"
        result = api.reduce_batched(
            group, payload, binary_op=operator, output_layout=layout
        )
        outputs = cute.make_tensor(memory, cute.make_layout(block))
        if group.rank() < 3:
            outputs[api.this_block().rank()] = result[0]

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=block)

    pointer = make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16)
    return cute.compile[(GPUArch("sm_80"),)](launch, pointer)


@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
@pytest.mark.parametrize("width", (1, 8, 32))
def test_entry_points(api, width):
    assert _compile(api, width=width) is not None


@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
@pytest.mark.parametrize(
    "bad,message",
    (
        ("scalar", "ThreadData"),
        ("group", "warp"),
        ("operator", "operator|binary_op"),
        ("layout", "output_layout"),
    ),
)
def test_invalid_profiles(api, bad, message):
    with pytest.raises(
        (TypeError, ValueError, NotImplementedError, DSLRuntimeError), match=message
    ):
        _compile(api, bad=bad)


def test_complete_warp_required():
    with pytest.raises(
        (ValueError, NotImplementedError, DSLRuntimeError), match="complete"
    ):
        _compile(block=48)


def test_qualified_register_input():
    assert _compile(cutlass_coop, bad="register") is not None
    with pytest.raises((TypeError, DSLRuntimeError), match="ThreadData"):
        _compile(coop, bad="register")
