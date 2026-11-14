# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check storage and block-shape errors during kernel compilation.

Typed null pointers supply the input type without allocating device data.
Each case compiles for an explicit architecture and stops before launch, so
invalid capacity or an incomplete physical warp must fail at compilation.
"""

import re

import cutlass
import pytest
from cutlass import cute
from cutlass.base_dsl.compiler import GPUArch
from cutlass.cute.runtime import make_ptr
from cutlass.memory import SmemAllocator

import cuda.coop.cutlass as coop

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.compile]


@pytest.mark.parametrize("sharing", ("shared", "exclusive"))
@pytest.mark.parametrize(
    "algorithm", ("transpose", "warp_transpose", "warp_transpose_timesliced")
)
def test_undersized_storage_is_rejected_before_kernel_execution(
    sharing, algorithm
):
    @cute.kernel
    def kernel(memory: cute.Pointer):
        storage = coop.TempStorage(1, sharing=sharing)
        coop.load(
            coop.this_block(),
            memory,
            items_per_thread=4,
            algorithm=algorithm,
            temp_storage=storage,
        )

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=64)

    pointer = make_ptr(
        cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16
    )
    with pytest.raises(Exception, match="(?i)(capacity|size|smaller)"):
        cute.compile[(GPUArch("sm_80"),)](launch, pointer)


@pytest.mark.parametrize(
    "algorithm", ("warp_transpose", "warp_transpose_timesliced")
)
def test_warp_transpose_rejects_partial_physical_warps(algorithm):
    @cute.kernel
    def kernel(memory: cute.Pointer):
        coop.load(
            coop.this_block(), memory, algorithm=algorithm, items_per_thread=4
        )

    @cute.jit
    def launch(memory: cute.Pointer):
        kernel(memory).launch(grid=1, block=33)

    pointer = make_ptr(
        cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16
    )
    with pytest.raises(Exception, match="multiple of 32"):
        cute.compile[(GPUArch("sm_80"),)](launch, pointer)


@pytest.mark.parametrize("requested", (None, 128, 512, "runtime"))
def test_deferred_storage_extends_explicit_launch_size(requested):
    """Include aligned provider scratch while preserving caller reservations."""

    @cute.kernel
    def kernel(memory: cute.Pointer):
        thread, _, _ = cute.arch.thread_idx()
        allocator = SmemAllocator()
        buffer = allocator.allocate_tensor(
            cutlass.Float32, cute.make_layout(32), byte_alignment=16
        )
        buffer[thread] = cutlass.Float32(thread)
        cute.arch.barrier()
        total = coop.sum(coop.this_warp(), cutlass.Float32(thread))
        if thread == 0:
            output = cute.make_tensor(memory, cute.make_layout(1))
            output[0] = total + buffer[31]

    @cute.jit
    def launch(memory: cute.Pointer, runtime_size: cutlass.Int64):
        if cutlass.const_expr(requested == "runtime"):
            kernel(memory).launch(grid=1, block=32, smem=runtime_size)
        else:
            kernel(memory).launch(grid=1, block=32, smem=requested)

    pointer = make_ptr(
        cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=16
    )
    compiled = cute.compile[(GPUArch("sm_80"),)](
        launch, pointer, cutlass.Int64(128)
    )
    host_ir = str(compiled.ir_module)
    # Inspect the actual launch ABI field, not a scratch planner estimate.
    store = re.search(
        r"%\w+ = llvm.getelementptr %\w+\[0, 2\].*\n\s*llvm.store (%\w+),",
        host_ir,
    )
    assert store is not None
    value = store[1]
    if requested == "runtime":
        maximum = re.search(
            re.escape(value) + r" = llvm.select (%\w+), %arg1, (%\w+)",
            host_ir,
        )
        assert maximum is not None
        predicate, inferred = maximum.groups()
        assert re.search(
            re.escape(predicate)
            + r' = llvm.icmp "sgt" %arg1, '
            + re.escape(inferred),
            host_ir,
        )
        assert re.search(
            re.escape(inferred) + r" = llvm.mlir.constant\(144 : i64\)",
            host_ir,
        )
    # The compiler's explicit-size diagnostic compares the same kernel query
    # and requested budget used by this launch. Extending that budget makes
    # its print-only insufficient-size branch unreachable; hardware-limit
    # diagnostics still remain for a runtime request.
    warning = re.search(
        r'constant @(\w+)\("[^\n]*'
        r"may exceed available memory set in kernel launch",
        host_ir,
    )
    if warning is not None:
        assert f"llvm.mlir.addressof @{warning[1]} " not in host_ir
    if requested == "runtime":
        hardware_warning = re.search(
            r'constant @(\w+)\("[^\n]*exceeds current GPU arch', host_ir
        )
        assert hardware_warning is not None
        assert f"llvm.mlir.addressof @{hardware_warning[1]} " in host_ir
    else:
        expected = 512 if requested == 512 else 144
        assert re.search(
            re.escape(value) + rf" = llvm.mlir.constant\({expected} : i64\)",
            host_ir,
        )


@pytest.mark.parametrize(
    "operation", ["implicit_sum", "sum", "reserve", "release"]
)
@pytest.mark.parametrize("branch", ["then", "else"])
def test_runtime_conditional_expression_rejects_storage(operation, branch):
    # This used to retain placeholders belonging to CuTe's erased speculative
    # trace and segfault Python in finalization. Compilation must stop safely.
    def value(storage):
        if operation == "implicit_sum":
            return coop.sum(coop.this_block(), cutlass.Int32(1))
        if operation == "sum":
            return coop.sum(
                coop.this_block(), cutlass.Int32(1), temp_storage=storage
            )
        if operation == "reserve":
            return storage.reserve(1, cutlass.Int32)[0]
        return storage.release(cutlass.Int32)[0]

    @cute.kernel
    def kernel(flag: cutlass.Int32):
        storage = coop.TempStorage(256, alignment=4)
        if cutlass.const_expr(branch == "then"):
            total = value(storage) if flag > 0 else cutlass.Int32(0)
        else:
            total = cutlass.Int32(0) if flag > 0 else value(storage)
        cute.printf("%d", total)

    @cute.jit
    def launch(flag: cutlass.Int32):
        kernel(flag).launch(grid=1, block=64)

    with pytest.raises(Exception, match="use an if statement"):
        cute.compile[(GPUArch("sm_80"),)](launch, cutlass.Int32(1))


@pytest.mark.parametrize("control_flow", ["if", "for", "while"])
@pytest.mark.parametrize("same_descriptor", [False, True])
def test_storage_selection_preserves_identity_or_rejects(
    control_flow, same_descriptor
):
    @cute.kernel
    def kernel(flag: cutlass.Int32):
        storage = coop.TempStorage(256, alignment=4)
        original = storage
        if cutlass.const_expr(control_flow == "if"):
            if flag > 0:
                if cutlass.const_expr(same_descriptor):
                    storage = original
                else:
                    storage = coop.TempStorage(256, alignment=4)
        elif cutlass.const_expr(control_flow == "for"):
            for iteration in range(flag):
                if cutlass.const_expr(same_descriptor):
                    storage = original
                else:
                    storage = coop.TempStorage(256, alignment=4)
                storage.reserve(64, cutlass.Int32)
        else:
            iteration = cutlass.Int32(0)
            while iteration < flag:
                if cutlass.const_expr(same_descriptor):
                    storage = original
                else:
                    storage = coop.TempStorage(256, alignment=4)
                storage.reserve(64, cutlass.Int32)
                iteration += 1
        view = storage.reserve(64, cutlass.Int32)
        view[0] = cutlass.Int32(7)

    @cute.jit
    def launch(flag: cutlass.Int32):
        kernel(flag).launch(grid=1, block=64)

    if same_descriptor:
        assert cute.compile[(GPUArch("sm_80"),)](launch, cutlass.Int32(2))
    else:
        with pytest.raises(Exception, match="different descriptor instances"):
            cute.compile[(GPUArch("sm_80"),)](launch, cutlass.Int32(2))


@pytest.mark.parametrize("select_at_runtime", [False, True])
def test_host_storage_cannot_be_selected_by_runtime_control_flow(
    select_at_runtime,
):
    first = coop.TempStorage(64, alignment=4)
    second = coop.TempStorage(128, alignment=4, sharing="exclusive")

    @cute.kernel
    def kernel(flag: cutlass.Int32):
        storage = first
        if cutlass.const_expr(select_at_runtime):
            if flag > 0:
                storage = second
        view = storage.reserve(16, cutlass.Int32)
        view[0] = cutlass.Int32(7)

    @cute.jit
    def launch(flag: cutlass.Int32):
        kernel(flag).launch(grid=1, block=32)

    if select_at_runtime:
        with pytest.raises(Exception, match="Host-created TempStorage"):
            cute.compile[(GPUArch("sm_80"),)](launch, cutlass.Int32(1))
    else:
        assert cute.compile[(GPUArch("sm_80"),)](launch, cutlass.Int32(1))
