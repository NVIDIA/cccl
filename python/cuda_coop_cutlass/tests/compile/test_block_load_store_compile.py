# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check movement diagnostics with CuTe compilation for an explicit target.

Null device pointers supply argument types without allocating buffers or
launching kernels. Invalid scalar controls and payload types must fail during
tracing. A runtime block-size argument must not become an exact launch fact
merely because its example value is 32.
"""

import cutlass
import pytest
from cutlass import cute
from cutlass.base_dsl.compiler import GPUArch
from cutlass.cute.runtime import make_ptr

import cuda.coop.cutlass as coop

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.compile]


@pytest.mark.parametrize(
    "algorithm",
    (
        " DIRECT ",
        " STRIPED ",
        " VECTORIZE ",
        " TRANSPOSE ",
        " WARP-TRANSPOSE ",
        " WARP-TRANSPOSE-TIMESLICED ",
    ),
)
@pytest.mark.parametrize("items_per_thread", (1, 4))
def test_algorithm_selector_spellings(algorithm, items_per_thread):
    """Normalize movement selector spelling through the qualified API.

    Uppercase names, surrounding spaces, and hyphenated names must be accepted
    as supported algorithm names. Typed null pointers and an SM80 target check
    compilation only; the test never launches a kernel.
    """

    @cute.kernel
    def kernel(memory: cute.Pointer, items_per_thread: cutlass.Constexpr):
        group = coop.this_block()
        payload = coop.load(
            group,
            memory,
            items_per_thread=items_per_thread,
            algorithm=algorithm,
        )
        coop.store(group, memory, payload, algorithm=algorithm)

    @cute.jit
    def launch(memory: cute.Pointer, items_per_thread: cutlass.Constexpr):
        kernel(memory, items_per_thread).launch(grid=1, block=32)

    pointer = make_ptr(
        cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16
    )
    assert (
        cute.compile[(GPUArch("sm_80"),)](launch, pointer, items_per_thread)
        is not None
    )


@pytest.mark.parametrize(
    "case,message",
    (
        ("store-dtype", "dtype"),
        ("scalar-dtype", "dtype"),
        ("boolean-count", "integer"),
        ("dynamic-items", "integer"),
        ("dynamic-alignment", "alignment"),
        ("float-offset", "integer"),
        ("negative-offset", "non-negative"),
        ("too-many-valid", "valid_items"),
        ("too-few-valid", "valid_items"),
    ),
)
def test_invalid_movement_inputs(case, message):
    @cute.kernel
    def kernel(memory: cute.Pointer):
        group = coop.this_block()
        if cutlass.const_expr(case == "store-dtype"):
            payload = coop.ThreadData(items_per_thread=1, dtype=cutlass.Float32)
            payload[0] = cutlass.Float32(1)
            coop.store(group, memory, payload)
        elif cutlass.const_expr(case == "scalar-dtype"):
            coop.store(group, memory, cutlass.Float32(1))
        elif cutlass.const_expr(case == "boolean-count"):
            coop.load(group, memory, valid_items=True, items_per_thread=1)
        elif cutlass.const_expr(case == "dynamic-items"):
            coop.load(
                group,
                memory,
                items_per_thread=cute.arch.thread_idx()[0] + 1,
            )
        elif cutlass.const_expr(case == "dynamic-alignment"):
            coop.load(
                group,
                memory,
                items_per_thread=1,
                alignment=cute.arch.thread_idx()[0] + 1,
            )
        elif cutlass.const_expr(case == "float-offset"):
            coop.load(group, memory, offset=1.5, items_per_thread=1)
        elif cutlass.const_expr(case == "negative-offset"):
            coop.load(group, memory, offset=-1, items_per_thread=1)
        elif cutlass.const_expr(case == "too-many-valid"):
            coop.load(group, memory, valid_items=33, items_per_thread=1)
        else:
            coop.load(group, memory, valid_items=-1, items_per_thread=1)

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=32)

    pointer = make_ptr(
        cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16
    )
    with pytest.raises(Exception, match=message):
        cute.compile[(GPUArch("sm_80"),)](launch, pointer)


def test_dynamic_block_dimensions_are_not_assumed_exact():
    """Keep runtime launch values distinct from compile-time dimensions.

    The launcher receives block_size as Int32, although compilation is called
    with 32. Cooperative specialization requires exact block dimensions. It
    must reject this dynamic input instead of using the sample value.
    """

    @cute.kernel
    def kernel(memory: cute.Pointer, items_per_thread: cutlass.Constexpr):
        coop.load(coop.this_block(), memory, items_per_thread=items_per_thread)

    @cute.jit
    def launch(
        memory: cute.Pointer,
        block_size: cutlass.Int32,
        items_per_thread: cutlass.Constexpr,
    ):
        kernel(memory, items_per_thread).launch(
            grid=1, block=(block_size, 1, 1)
        )

    pointer = make_ptr(
        cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16
    )
    with pytest.raises(Exception, match="exact block dimensions"):
        cute.compile[(GPUArch("sm_80"),)](launch, pointer, 32, 1)
