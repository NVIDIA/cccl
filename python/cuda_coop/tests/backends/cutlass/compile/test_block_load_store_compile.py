# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Invalid block movement inputs fail before linking or device execution."""

import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import cute
from cutlass.base_dsl.compiler import GPUArch
from cutlass.cute.runtime import make_ptr

from cuda.coop import cutlass as cutlass_coop

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.compile]


@pytest.mark.parametrize(
    "case,message",
    (
        ("load-dtype", "dtype"),
        ("store-dtype", "dtype"),
        ("scalar-dtype", "dtype"),
        ("boolean-count", "integer"),
        ("float-offset", "integer"),
        ("negative-offset", "non-negative"),
        ("too-many-valid", "valid_items"),
        ("too-few-valid", "valid_items"),
    ),
)
def test_invalid_movement_inputs(case, message):
    @cute.kernel
    def kernel(memory: cute.Pointer):
        group = cutlass_coop.this_block()
        if cutlass.const_expr(case == "load-dtype"):
            cutlass_coop.load(
                group, memory, cutlass_coop.ThreadData(1, dtype=cutlass.Float32)
            )
        elif cutlass.const_expr(case == "store-dtype"):
            payload = cutlass_coop.ThreadData(1, dtype=cutlass.Float32)
            payload[0] = cutlass.Float32(1)
            cutlass_coop.store(group, memory, payload)
        elif cutlass.const_expr(case == "scalar-dtype"):
            cutlass_coop.store(group, memory, cutlass.Float32(1))
        elif cutlass.const_expr(case == "boolean-count"):
            cutlass_coop.load(
                group, memory, cutlass_coop.ThreadData(1), valid_items=True
            )
        elif cutlass.const_expr(case == "float-offset"):
            cutlass_coop.load(group, memory, cutlass_coop.ThreadData(1), offset=1.5)
        elif cutlass.const_expr(case == "negative-offset"):
            cutlass_coop.load(group, memory, cutlass_coop.ThreadData(1), offset=-1)
        elif cutlass.const_expr(case == "too-many-valid"):
            cutlass_coop.load(group, memory, cutlass_coop.ThreadData(1), valid_items=33)
        else:
            cutlass_coop.load(group, memory, cutlass_coop.ThreadData(1), valid_items=-1)

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=32)

    pointer = make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16)
    with pytest.raises(Exception, match=message):
        cute.compile[(GPUArch("sm_80"),)](launch, pointer)


def test_dynamic_block_dimensions_are_not_assumed_exact():
    @cute.kernel
    def kernel(memory: cute.Pointer):
        cutlass_coop.load(cutlass_coop.this_block(), memory, cutlass_coop.ThreadData(1))

    @cute.jit
    def launch(memory: cute.Pointer, block_size: cutlass.Int32):
        kernel(memory).launch(grid=1, block=(block_size, 1, 1))

    pointer = make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16)
    with pytest.raises(Exception, match="exact block dimensions"):
        cute.compile[(GPUArch("sm_80"),)](launch, pointer, 32)
