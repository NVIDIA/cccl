# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Final-link, wide-control, and destination contracts without a GPU."""

import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import cute
from cutlass.base_dsl.common import DSLRuntimeError
from cutlass.base_dsl.compiler import GPUArch
from cutlass.cute.runtime import make_ptr

from cuda import coop
from cuda.coop import cutlass as cutlass_coop
from cuda.coop.cutlass._compiler._types import ALL_PROVIDER_TYPES, INTEGER_VALUE_TYPES

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.compile]


def _compile(
    api=coop,
    *,
    bulk=False,
    dtype=cutlass.Int32,
    length_dtype=cutlass.Uint32,
    control_type=cutlass.Uint64,
    block=32,
    bad=None,
    static_offset=None,
    dynamic_capacity=False,
):
    @cute.kernel
    def kernel(memory: cute.Pointer, offset: control_type, capacity: cutlass.Int64):
        values = api.ThreadData(2, dtype=dtype)
        lengths = api.ThreadData(1 if bad == "extent" else 2, dtype=length_dtype)
        for i in cutlass.range_constexpr(2):
            values[i] = dtype(i)
        for i in cutlass.range_constexpr(lengths.items_per_thread):
            lengths[i] = length_dtype(2)
        if cutlass.const_expr(bad == "register"):
            values = values.to_register_tensor()
        group = api.this_warp() if bad == "group" else api.this_block()
        if cutlass.const_expr(bad == "scalar"):
            values = dtype(3)
        control = offset if static_offset is None else static_offset
        if cutlass.const_expr(bulk):
            shape = capacity if dynamic_capacity else 256
            destination = cute.recast_tensor(
                cute.make_tensor(
                    memory, cute.make_layout(shape, stride=2 if bad == "strided" else 1)
                ),
                dtype,
            )
            if cutlass.const_expr(bad == "rank"):
                destination = cute.make_tensor(memory, cute.make_layout((16, 16)))
            if cutlass.const_expr(bad == "pointer"):
                destination = memory
            if cutlass.const_expr(bad == "dtype"):
                destination = cute.recast_tensor(destination, cutlass.Float64)
            if cutlass.const_expr(bad == "rmem"):
                destination = cute.make_rmem_tensor(256, dtype)
            total = api.run_length_decode_into(
                group,
                values,
                lengths,
                destination,
                decoded_items_per_thread=3,
                destination_offset=control,
            )
            output = cute.make_tensor(memory, cute.make_layout(256))
            output[0] = dtype(total)
        else:
            decoded = api.run_length_decode(
                group,
                values,
                lengths,
                decoded_items_per_thread=3,
                decoded_window_offset=control,
            )
            output = cute.recast_tensor(
                cute.make_tensor(memory, cute.make_layout(256)), dtype
            )
            output[0] = decoded[0]

    @cute.jit
    def launch(memory: cute.Pointer, offset: control_type, capacity: cutlass.Int64):
        kernel(memory, offset, capacity).launch(grid=1, block=block)

    pointer = make_ptr(dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    return cute.compile[(GPUArch("sm_80"),)](
        launch, pointer, control_type(0), cutlass.Int64(256)
    )


@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
@pytest.mark.parametrize("bulk", (False, True))
def test_entrypoints(api, bulk):
    assert _compile(api, bulk=bulk) is not None


@pytest.mark.parametrize("dtype", tuple(ALL_PROVIDER_TYPES))
@pytest.mark.parametrize("bulk", (False, True))
def test_value_types(dtype, bulk):
    assert _compile(dtype=dtype, bulk=bulk) is not None


@pytest.mark.parametrize("length_dtype", tuple(INTEGER_VALUE_TYPES))
def test_length_types(length_dtype):
    assert _compile(bulk=True, length_dtype=length_dtype) is not None


@pytest.mark.parametrize("control_type", tuple(INTEGER_VALUE_TYPES))
def test_offset_widths(control_type):
    assert _compile(control_type=control_type) is not None


def test_dynamic_destination_capacity():
    assert _compile(bulk=True, dynamic_capacity=True) is not None


@pytest.mark.parametrize("bad", ("strided", "rank", "pointer", "dtype", "rmem"))
def test_destination_contract(bad):
    with pytest.raises(
        (TypeError, ValueError, DSLRuntimeError),
        match="contiguous one-dimensional global-memory",
    ):
        _compile(bulk=True, bad=bad)


@pytest.mark.parametrize(
    "bad,message",
    (("extent", "matching"), ("group", "block"), ("scalar", "ThreadData")),
)
def test_input_contract(bad, message):
    with pytest.raises(
        (TypeError, ValueError, NotImplementedError, DSLRuntimeError), match=message
    ):
        _compile(bad=bad)


def test_multidimensional_block_rejected():
    with pytest.raises((ValueError, DSLRuntimeError), match="one-dimensional"):
        _compile(block=(8, 4, 1))


@pytest.mark.parametrize("offset", (-1, 1 << 64))
def test_static_offset_rejected(offset):
    with pytest.raises((ValueError, DSLRuntimeError), match="unsigned 64-bit"):
        _compile(static_offset=offset)


def test_qualified_register_conversion():
    assert _compile(cutlass_coop, bad="register") is not None


def test_common_register_rejected():
    with pytest.raises((TypeError, DSLRuntimeError), match="ThreadData"):
        _compile(bad="register")
