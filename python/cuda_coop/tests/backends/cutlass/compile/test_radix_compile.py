# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Radix compiler contracts and qualified input/output controls."""

import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import cute
from cutlass.base_dsl.common import DSLRuntimeError
from cutlass.base_dsl.compiler import GPUArch
from cutlass.cute.runtime import make_ptr

from cuda import coop
from cuda.coop import cutlass as cutlass_coop

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.compile]


def _compile(
    *,
    api=cutlass_coop,
    mode="sort",
    dtype=cutlass.Int32,
    block=(8, 4, 2),
    items=3,
    bad=None,
    radix_bits=4,
    prefix=False,
):
    @cute.kernel
    def kernel(memory: cute.Pointer, bound: cutlass.Int64):
        values = cute.make_tensor(memory, cute.make_layout(1024))
        keys = api.ThreadData(items, dtype=dtype)
        payload = api.ThreadData(items)
        for item in cutlass.range_constexpr(items):
            keys[item] = dtype(item + 3)
            payload[item] = cutlass.Float64(item + 0.5)
        group = api.this_warp() if bad == "group" else api.this_block()
        if cutlass.const_expr(bad in {"register", "common-register"}):
            keys = cute.make_rmem_tensor(items, dtype)
            keys.fill(dtype(3))
        if cutlass.const_expr(bad == "scalar-array"):
            keys = cutlass.Int32(3)
        if cutlass.const_expr(mode == "rank"):
            output = None
            if cutlass.const_expr(prefix):
                count = max(1, ((1 << radix_bits) + 63) // 64)
                output = api.ThreadData(count, dtype=cutlass.Int32)
            if cutlass.const_expr(bad == "prefix-dtype"):
                output = api.ThreadData(1, dtype=cutlass.Uint32)
            if cutlass.const_expr(bad == "prefix-extent"):
                output = api.ThreadData(7, dtype=cutlass.Int32)
            if cutlass.const_expr(bad == "prefix-alias"):
                output = keys
            result = (
                api.radix_rank(
                    group,
                    keys,
                    begin_bit=bound if bad == "rank-runtime" else 0,
                    radix_bits=radix_bits,
                    exclusive_digit_prefix=output,
                )
                if cutlass.const_expr(api is cutlass_coop)
                else api.radix_rank(group, keys, radix_bits=radix_bits)
            )
        else:
            begin = bound
            end = 32
            if cutlass.const_expr(bad == "uint64"):
                begin = cutlass.Uint64(bound)
            if cutlass.const_expr(bad == "bool"):
                begin = True
            if cutlass.const_expr(bad == "float"):
                begin = cutlass.Float32(1)
            if cutlass.const_expr(bad == "empty"):
                begin = 32
            if cutlass.const_expr(mode == "pairs"):
                result, _ = api.radix_sort_pairs(
                    group, keys, payload, begin_bit=begin, end_bit=end
                )
            else:
                result = api.radix_sort_keys(group, keys, begin_bit=begin, end_bit=end)
        for item in cutlass.range_constexpr(items):
            values[item] = cutlass.Int32(result[item])

    @cute.jit
    def launch(memory: cute.Pointer, bound: cutlass.Int64):
        kernel(memory, bound).launch(grid=1, block=block)

    ptr = make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16)
    return cute.compile[(GPUArch("sm_80"),)](launch, ptr, cutlass.Int64(0))


@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
@pytest.mark.parametrize("mode", ("sort", "pairs", "rank"))
@pytest.mark.parametrize(
    "dtype", (cutlass.Int32, cutlass.Uint32, cutlass.Int64, cutlass.Uint64)
)
def test_common_profiles(api, mode, dtype):
    assert _compile(api=api, mode=mode, dtype=dtype) is not None


@pytest.mark.parametrize("bits", (1, 4, 7, 8))
def test_rank_prefix(bits):
    assert _compile(mode="rank", radix_bits=bits, prefix=True) is not None


@pytest.mark.parametrize("dtype", (cutlass.Float32, cutlass.Float64))
def test_qualified_float_sort(dtype):
    assert _compile(dtype=dtype) is not None


@pytest.mark.parametrize(
    "mode,bad,dtype,message",
    (
        ("sort", "group", cutlass.Int32, "physical block"),
        ("sort", "uint64", cutlass.Int32, "integer"),
        ("sort", "bool", cutlass.Int32, "integer"),
        ("sort", "float", cutlass.Int32, "integer"),
        ("sort", "empty", cutlass.Int32, "bit"),
        ("sort", None, cutlass.Int16, "supports"),
        ("rank", None, cutlass.Float32, "supports"),
        ("rank", "rank-runtime", cutlass.Int32, "compile-time"),
        ("rank", "prefix-dtype", cutlass.Int32, "Int32"),
        ("rank", "prefix-extent", cutlass.Int32, "items"),
        ("rank", "prefix-alias", cutlass.Int32, "distinct"),
        ("pairs", "scalar-array", cutlass.Int32, "matching scalar"),
    ),
)
def test_invalid_profiles(mode, bad, dtype, message):
    with pytest.raises(
        (TypeError, ValueError, NotImplementedError, DSLRuntimeError), match=message
    ):
        _compile(mode=mode, bad=bad, dtype=dtype)


def test_qualified_register_payload():
    assert _compile(bad="register") is not None


def test_common_register_rejected():
    with pytest.raises((TypeError, DSLRuntimeError), match="ThreadData"):
        _compile(api=coop, bad="common-register")


def test_non_power_of_two_block():
    assert _compile(block=(12, 4, 1)) is not None
