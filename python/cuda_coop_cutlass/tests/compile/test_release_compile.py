# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile native released views and reject consumed descriptor uses."""

from io import BytesIO

import cutlass
import numpy as np
import pytest
from cutlass import cute
from cutlass.base_dsl.compiler import GPUArch
from cutlass.base_dsl.dsl import BaseDSL
from cutlass.cute.runtime import make_ptr

import cuda.coop.cutlass as coop
import cuda.coop.cutlass as cutlass_coop

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.compile]


def _compile(kernel, *args):
    @cute.jit
    def launch(memory: cute.Pointer, flag: cutlass.Int32):
        kernel(memory, flag).launch(grid=1, block=64)

    pointer = make_ptr(
        cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16
    )
    return cute.compile[(GPUArch("sm_80"),)](launch, pointer, cutlass.Int32(2))


def test_release_rejects_unmodeled_extension_control_flow():
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import cf, scf

    def native_region(storage, flag):
        region = scf.ExecuteRegionOp([]).regions[0]
        entry = region.blocks.append()
        default = region.blocks.append()
        selected = region.blocks.append()
        with ir.InsertionPoint(entry):
            cf.SwitchOp(
                flag.ir_value(),
                [],
                [],
                [0],
                default,
                [selected],
                case_values=ir.DenseIntElementsAttr.get(
                    np.array([1], dtype=np.int32)
                ),
            )
        with ir.InsertionPoint(default):
            scf.YieldOp([])
        with ir.InsertionPoint(selected):
            storage.release()
            scf.YieldOp([])

    @cute.kernel
    def kernel(memory: cute.Pointer, flag: cutlass.Int32):
        storage = coop.TempStorage(64)
        native_region(storage, flag)

    with pytest.raises(
        Exception, match="cannot analyze control flow cf.switch"
    ):
        _compile(kernel)


@pytest.mark.parametrize("api", (cutlass_coop,))
@pytest.mark.parametrize("dtype", (None, np.int32, cutlass.Float32))
@pytest.mark.parametrize("sharing", ("shared", "exclusive"))
def test_release_native_view(api, dtype, sharing):
    @cute.kernel
    def kernel(memory: cute.Pointer, flag: cutlass.Int32):
        storage = api.TempStorage(sharing=sharing)
        storage.reserve(64, cutlass.Int32)
        view = storage.release(dtype)
        view[0] = view.element_type(7)
        output = cute.make_tensor(memory, cute.make_layout(1))
        output[0] = cutlass.Int32(view[0]) + cutlass.Int32(cute.size(view))

    assert _compile(kernel) is not None


@pytest.fixture
def serialized_modules(monkeypatch):
    # Compile-only mode skips CuTe's cache serialization. Exercise it here
    # after real lifetime analysis, including native tensor SSA transfers.
    serialized = []
    original_build_module = BaseDSL.build_module

    def serialize_module(self, module, function_name):
        buffer = BytesIO()
        module.operation.write_bytecode(buffer)
        serialized.append(buffer.getvalue())
        return original_build_module(self, module, function_name)

    monkeypatch.setattr(BaseDSL, "build_module", serialize_module)
    return serialized


@pytest.mark.parametrize("control_flow", ("if", "for", "while"))
def test_release_native_views_serialize_with_control_flow(
    serialized_modules, control_flow
):

    @cute.kernel
    def kernel(memory: cute.Pointer, flag: cutlass.Int32):
        storage = coop.TempStorage()
        first = storage.reserve(65, cutlass.Int32)
        second = storage.reserve(64, cutlass.Float64, alignment=128)
        released = storage.release()
        output = cute.make_tensor(memory, cute.make_layout(3))
        if cutlass.const_expr(control_flow == "if"):
            if flag > 0:
                output[0] = cutlass.Int32(cute.size(released))
                output[1] = cutlass.Int32(first.iterator.toint())
                output[2] = cutlass.Int32(second.iterator.toint())
        elif cutlass.const_expr(control_flow == "for"):
            for iteration in range(flag):
                output[0] = cutlass.Int32(cute.size(released))
                output[1] = cutlass.Int32(first.iterator.toint())
                output[2] = cutlass.Int32(second.iterator.toint())
        else:
            iteration = cutlass.Int32(0)
            while iteration < flag:
                output[0] = cutlass.Int32(cute.size(released))
                output[1] = cutlass.Int32(first.iterator.toint())
                output[2] = cutlass.Int32(second.iterator.toint())
                iteration += 1

    assert _compile(kernel) is not None
    assert serialized_modules and all(serialized_modules)


@pytest.mark.parametrize(
    "operation", ("reserve", "release", "sync", "direct", "transpose")
)
def test_release_rejects_later_use(operation):
    @cute.kernel
    def kernel(memory: cute.Pointer, flag: cutlass.Int32):
        storage = coop.TempStorage(512, alignment=16)
        alias = storage
        storage.release()
        if cutlass.const_expr(operation == "reserve"):
            alias.reserve(64, cutlass.Int32)
        elif cutlass.const_expr(operation == "release"):
            alias.release()
        elif cutlass.const_expr(operation == "sync"):
            alias.sync()
        else:
            coop.load(
                coop.this_block(),
                memory,
                items_per_thread=1,
                algorithm=operation,
                temp_storage=alias,
            )

    with pytest.raises(Exception, match="after release") as error:
        _compile(kernel)
    assert "test_release_compile.py:" in str(error.value)


@pytest.mark.parametrize(
    "kind",
    ("branches", "fresh_for", "fresh_while", "one_iteration", "zero_trip"),
)
def test_release_accepts_independent_paths_and_fresh_instances(kind):
    @cute.kernel
    def kernel(memory: cute.Pointer, flag: cutlass.Int32):
        if cutlass.const_expr(kind == "branches"):
            storage = coop.TempStorage(64)
            if flag > 0:
                storage.release()
            else:
                storage.release()
        elif cutlass.const_expr(kind == "fresh_for"):
            for iteration in range(flag):
                storage = coop.TempStorage(64)
                storage.release()
        elif cutlass.const_expr(kind == "fresh_while"):
            iteration = cutlass.Int32(0)
            while iteration < flag:
                storage = coop.TempStorage(64)
                storage.release()
                iteration += 1
        elif cutlass.const_expr(kind == "one_iteration"):
            storage = coop.TempStorage(64)
            for iteration in range(1):
                storage.release()
        else:
            storage = coop.TempStorage(64)
            storage.release()
            for iteration in range(0):
                storage.release()

    assert _compile(kernel) is not None


@pytest.mark.parametrize("kind", ("join", "outer_for", "outer_while"))
def test_release_rejects_reachable_join_and_backedge(kind):
    @cute.kernel
    def kernel(memory: cute.Pointer, flag: cutlass.Int32):
        storage = coop.TempStorage(256)
        if cutlass.const_expr(kind == "join"):
            if flag > 0:
                storage.release()
            storage.reserve(64, cutlass.Int32)
        elif cutlass.const_expr(kind == "outer_for"):
            for iteration in range(flag):
                storage.release()
        else:
            iteration = cutlass.Int32(0)
            while iteration < flag:
                storage.release()
                iteration += 1

    with pytest.raises(Exception, match="after release"):
        _compile(kernel)


@pytest.mark.parametrize("selection", ("same", "different"))
def test_release_tracks_runtime_selected_alias(selection, serialized_modules):
    @cute.kernel
    def kernel(memory: cute.Pointer, flag: cutlass.Int32):
        first = coop.TempStorage(64)
        second = first
        if cutlass.const_expr(selection == "different"):
            second = coop.TempStorage(64)
        first_view = first.reserve(1, cutlass.Int32)
        second_view = second.reserve(1, cutlass.Int32)
        selected_view = first_view
        selected = first
        # Carry both a native tensor and a descriptor token through one SCF
        # result tuple; filtering tensor values must preserve token positions.
        if flag > 0:
            selected_view = first_view
            selected = first
        else:
            selected_view = second_view
            selected = second
        selected.release()
        output = cute.make_tensor(memory, cute.make_layout(1))
        output[0] = cutlass.Int32(selected_view.iterator.toint())

    if selection == "same":
        assert _compile(kernel) is not None
        assert serialized_modules and all(serialized_modules)
    else:
        with pytest.raises(Exception, match="different descriptor instances"):
            _compile(kernel)


def test_new_iteration_does_not_revive_old_alias():
    @cute.kernel
    def kernel(memory: cute.Pointer, flag: cutlass.Int32):
        previous = coop.TempStorage(256, alignment=4)
        for iteration in range(flag):
            current = coop.TempStorage(256, alignment=4)
            old_alias = previous
            previous = current
            current.release()
            old_alias.reserve(64, cutlass.Int32)

    with pytest.raises(
        Exception, match="after release|different descriptor instances"
    ):
        _compile(kernel)


@pytest.mark.parametrize("use_alias", (False, True))
def test_inlined_helper_and_bound_method_preserve_consumption(use_alias):
    @cute.jit
    def release(storage):
        return storage.release()

    @cute.kernel
    def kernel(memory: cute.Pointer, flag: cutlass.Int32):
        storage = coop.TempStorage(256)
        method = storage.release
        if cutlass.const_expr(use_alias):
            release(storage)
        else:
            method()
        storage.release()

    with pytest.raises(Exception, match="after release"):
        _compile(kernel)


@pytest.mark.parametrize(
    "size,alignment,dtype,diagnostic",
    [
        (None, None, None, "nonempty"),
        (7, 4, cutlass.Int32, "not divisible"),
        (64, 1, cutlass.Int32, "alignment"),
        (64, 16, object, "supported numeric scalar"),
        (64, 16, 1, "compile-time scalar"),
    ],
)
def test_release_region_validation(size, alignment, dtype, diagnostic):
    @cute.kernel
    def kernel(memory: cute.Pointer, flag: cutlass.Int32):
        coop.TempStorage(size, alignment=alignment).release(dtype)

    with pytest.raises(Exception, match=diagnostic):
        _compile(kernel)


@pytest.mark.parametrize("operation", ("reserve", "release", "sync"))
def test_release_requires_kernel_local_descriptor(operation):
    storage = cutlass_coop.TempStorage(256, alignment=4)

    @cute.kernel
    def kernel(memory: cute.Pointer, flag: cutlass.Int32):
        if cutlass.const_expr(operation == "reserve"):
            storage.reserve(64, cutlass.Int32)
        elif cutlass.const_expr(operation == "sync"):
            storage.sync()
        storage.release()

    with pytest.raises(Exception, match="constructed inside the kernel"):
        _compile(kernel)
