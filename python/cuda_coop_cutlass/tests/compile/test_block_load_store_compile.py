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
import numpy as np
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
    """Normalize movement selector spelling through the CUTLASS API.

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
        ("store-overflow", "not representable in Int32"),
        ("store-payload-overflow", "not representable in Int32"),
        ("boolean-count", "integer"),
        ("dynamic-items", "integer"),
        ("dynamic-alignment", "alignment"),
        ("float-offset", "integer"),
        ("negative-offset", "non-negative"),
        ("too-many-valid", "valid_items"),
        ("too-few-valid", "valid_items"),
        ("declared-float", "does not match the memory dtype Int32"),
        ("declared-width", "does not match the memory dtype Int32"),
        ("declared-signed-view", "does not match the memory dtype Uint32"),
        ("declared-value", "compile-time scalar type"),
        ("declared-unsupported", "dtype must be one of"),
        ("declared-float-memory", "does not match the memory dtype Float32"),
        ("declared-default", "not representable in Uint32"),
        ("declared-runtime-default", "must match the memory dtype"),
        ("declared-store-payload", "destination dtype does not match"),
        ("undeclared-unsigned-store", "pass dtype=Uint32"),
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
        elif cutlass.const_expr(case == "store-overflow"):
            coop.store(group, memory, 1 << 40)
        elif cutlass.const_expr(case == "store-payload-overflow"):
            coop.store(group, memory, coop.ThreadData.from_values(1 << 40))
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
        elif cutlass.const_expr(case == "too-few-valid"):
            coop.load(group, memory, valid_items=-1, items_per_thread=1)
        elif cutlass.const_expr(case == "declared-float"):
            coop.load(group, memory, items_per_thread=1, dtype=cutlass.Float32)
        elif cutlass.const_expr(case == "declared-width"):
            coop.load(group, memory, items_per_thread=1, dtype=cutlass.Uint16)
        elif cutlass.const_expr(case == "declared-signed-view"):
            view = cute.recast_tensor(
                cute.make_tensor(memory, cute.make_layout(32)), cutlass.Uint32
            )
            coop.load(group, view, items_per_thread=1, dtype=cutlass.Int32)
        elif cutlass.const_expr(case == "declared-value"):
            coop.load(group, memory, items_per_thread=1, dtype=32)
        elif cutlass.const_expr(case == "declared-unsupported"):
            coop.load(group, memory, items_per_thread=1, dtype=cutlass.Float16)
        elif cutlass.const_expr(case == "declared-float-memory"):
            view = cute.recast_tensor(
                cute.make_tensor(memory, cute.make_layout(32)), cutlass.Float32
            )
            coop.load(group, view, items_per_thread=1, dtype=cutlass.Uint32)
        elif cutlass.const_expr(case == "declared-default"):
            coop.load(
                group,
                memory,
                items_per_thread=1,
                dtype=cutlass.Uint32,
                valid_items=31,
                oob_default=-7,
            )
        elif cutlass.const_expr(case == "declared-runtime-default"):
            coop.load(
                group,
                memory,
                items_per_thread=1,
                dtype=cutlass.Uint32,
                valid_items=31,
                oob_default=cutlass.Int32(1),
            )
        elif cutlass.const_expr(case == "declared-store-payload"):
            coop.store(group, memory, cutlass.Int32(1), dtype=cutlass.Uint32)
        else:
            coop.store(group, memory, cutlass.Uint32(1))

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=32)

    pointer = make_ptr(
        cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16
    )
    with pytest.raises(Exception, match=message):
        cute.compile[(GPUArch("sm_80"),)](launch, pointer)


@pytest.mark.parametrize("operand", ("pointer", "offset-pointer", "tensor"))
@pytest.mark.parametrize(
    "dtype,value_type",
    (
        (cutlass.Uint32, cutlass.Uint32),
        (np.uint32, cutlass.Uint32),
        (np.dtype(np.uint32), cutlass.Uint32),
        (cutlass.Int32, cutlass.Int32),
    ),
)
def test_declared_memory_dtype(operand, dtype, value_type):
    """Declare the element type of memory that CuTe reports as signed.

    A pointer argument, pointer arithmetic and ``cute.make_tensor`` all
    report ``Int32`` for memory the host declared ``Uint32``. The declared
    type becomes the payload dtype, bounds the default, and lets Store
    accept a payload of that type.
    """

    default = 4_000_000_000 if value_type is cutlass.Uint32 else -7

    @cute.kernel
    def kernel(memory: cute.Pointer):
        group = coop.this_block()
        if cutlass.const_expr(operand == "pointer"):
            target = memory
        elif cutlass.const_expr(operand == "offset-pointer"):
            target = memory + 32
        else:
            target = cute.make_tensor(memory, cute.make_layout(64))
        payload = coop.load(
            group,
            target,
            items_per_thread=1,
            dtype=dtype,
            valid_items=31,
            oob_default=default,
        )
        assert payload.dtype is value_type
        filled = coop.load(
            group,
            target,
            items_per_thread=1,
            dtype=dtype,
            valid_items=31,
            oob_default=value_type(5),
        )
        assert filled.dtype is value_type
        coop.store(group, target, payload, dtype=dtype)

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=32)

    pointer = make_ptr(
        cutlass.Uint32, 0, cute.AddressSpace.gmem, assumed_align=16
    )
    assert cute.compile[(GPUArch("sm_80"),)](launch, pointer) is not None


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
