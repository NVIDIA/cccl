# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The load and store controls, and scalar store values, coop rejects.

Each case is one of coop's rules for ``valid_items=``, ``oob_default=``,
``offset=`` or a scalar ``store`` value, checked by its message text.
"""

from __future__ import annotations

import pytest
from numba_cuda_mlir import types

from cuda import coop
from cuda.coop.numba_mlir._pipeline import testing
from cuda.coop.numba_mlir._pipeline._errors import TypedPipelineError

from .._support import BLOCK, CHIP, f32_in_out, requires_typed_pipeline

pytestmark = requires_typed_pipeline


def _compile(kernel, *extra):
    signature = f32_in_out()
    if extra:
        signature = types.void(*signature.args, *extra)
    return testing.compile_kernel(kernel, signature, block=BLOCK, chip=CHIP)


def test_oob_default_requires_valid_items():
    def kernel(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        coop.load(block, source, items, oob_default=0.0)
        coop.store(block, output, items)

    with pytest.raises(
        TypedPipelineError,
        match=r"coop\.load: oob_default requires valid_items",
    ):
        _compile(kernel)


@pytest.mark.parametrize("kind", ["load", "store"])
def test_static_valid_items_beyond_the_tile(kind):
    def load_kernel(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        coop.load(block, source, items, valid_items=513)
        coop.store(block, output, items)

    def store_kernel(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        coop.load(block, source, items)
        coop.store(block, output, items, valid_items=-1)

    with pytest.raises(
        TypedPipelineError,
        match=rf"coop\.{kind}: static valid_items must be between zero and "
        r"the group tile size \(512\)",
    ):
        _compile(load_kernel if kind == "load" else store_kernel)


@pytest.mark.parametrize("control", ["valid_items", "offset"])
def test_static_control_needs_int(control):
    def valid_items_kernel(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        coop.load(block, source, items, valid_items=1.5)
        coop.store(block, output, items)

    def offset_kernel(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        coop.load(block, source, items, offset=True)
        coop.store(block, output, items)

    kernel = valid_items_kernel if control == "valid_items" else offset_kernel
    with pytest.raises(
        TypedPipelineError, match=rf"coop load {control} must be an integer"
    ):
        _compile(kernel)


def test_static_offset_nonnegative():
    def kernel(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        coop.load(block, source, items)
        coop.store(block, output, items, offset=-4)

    with pytest.raises(
        TypedPipelineError,
        match=r"coop load/store static offset must be nonnegative",
    ):
        _compile(kernel)


@pytest.mark.parametrize(
    "control_type, message",
    [
        (types.float32, r"must be an integer, not bool or a noninteger scalar"),
        (types.boolean, r"must be an integer, not bool or a noninteger scalar"),
        (
            types.uint64,
            r"must be a signed integer up to 64 bits or an unsigned",
        ),
    ],
    ids=["float32", "bool", "uint64"],
)
@pytest.mark.parametrize("control", ["valid_items", "offset"])
def test_a_runtime_control_must_have_an_integer_type(
    control, control_type, message
):
    def valid_items_kernel(source, output, value):
        block = coop.this_block()
        items = coop.ThreadData(4)
        coop.load(block, source, items)
        coop.store(block, output, items, valid_items=value)

    def offset_kernel(source, output, value):
        block = coop.this_block()
        items = coop.ThreadData(4)
        coop.load(block, source, items)
        coop.store(block, output, items, offset=value)

    kernel = valid_items_kernel if control == "valid_items" else offset_kernel
    with pytest.raises(
        TypedPipelineError, match=rf"coop store {control} {message}"
    ):
        _compile(kernel, control_type)


def test_a_runtime_oob_default_must_match_the_payload_dtype():
    def kernel(source, output, pad):
        block = coop.this_block()
        items = coop.ThreadData(4)
        coop.load(block, source, items, valid_items=100, oob_default=pad)
        coop.store(block, output, items)

    with pytest.raises(
        TypedPipelineError,
        match=r"cuda\.coop\.numba_mlir\.load runtime oob_default dtype "
        r"float64 does not match payload dtype float32",
    ):
        _compile(kernel, types.float64)


def test_a_runtime_oob_default_must_be_numeric():
    def kernel(source, output, pad):
        block = coop.this_block()
        items = coop.ThreadData(4)
        coop.load(block, source, items, valid_items=100, oob_default=pad)
        coop.store(block, output, items)

    with pytest.raises(
        TypedPipelineError,
        match=r"cuda\.coop\.load supports oob_default dtypes int8, .*float64 "
        r"through the common API",
    ):
        _compile(kernel, types.boolean)


def test_a_static_oob_default_does_not_convert_float_to_integer():
    def kernel(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        coop.load(block, source, items, valid_items=100, oob_default=0.5)
        coop.store(block, output, items)

    signature = types.void(types.int32[::1], types.int32[::1])
    with pytest.raises(
        TypedPipelineError, match=r"does not permit float-to-integer conversion"
    ):
        testing.compile_kernel(kernel, signature, block=BLOCK, chip=CHIP)


def test_oob_default_is_only_for_load():
    def kernel(source, output):
        block = coop.this_block()
        items = coop.ThreadData(4)
        coop.load(block, source, items)
        coop.store(block, output, items, valid_items=100, oob_default=0.0)

    with pytest.raises(
        Exception, match=r"unexpected keyword argument 'oob_default'"
    ):
        _compile(kernel)


def test_a_runtime_scalar_store_value_must_match_the_destination():
    def kernel(source, output, value):
        coop.store(coop.this_block(), output, value)

    with pytest.raises(
        TypedPipelineError,
        match=r"cuda\.coop\.numba_mlir\.store value dtype float64 does not "
        r"match destination dtype float32",
    ):
        _compile(kernel, types.float64)


def test_a_static_scalar_store_value_must_fit_the_destination():
    def kernel(output):
        coop.store(coop.this_block(), output, 300)

    signature = types.void(types.int8[::1])
    with pytest.raises(
        TypedPipelineError, match=r"value 300 is outside the range of int8"
    ):
        testing.compile_kernel(kernel, signature, block=BLOCK, chip=CHIP)


def test_a_static_scalar_store_value_must_not_be_bool():
    def kernel(output):
        coop.store(coop.this_block(), output, True)

    signature = types.void(types.int32[::1])
    with pytest.raises(TypedPipelineError, match=r"must not be bool"):
        testing.compile_kernel(kernel, signature, block=BLOCK, chip=CHIP)
