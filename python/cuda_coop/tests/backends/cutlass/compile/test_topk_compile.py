# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""TopK shape, type, wide count, and qualified conversion compilation."""

import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import cute
from cutlass.base_dsl.common import DSLRuntimeError
from cutlass.base_dsl.compiler import GPUArch
from cutlass.cute.runtime import make_ptr

from cuda import coop
from cuda.coop import cutlass as cutlass_coop
from cuda.coop.cutlass._compiler._types import ALL_PROVIDER_TYPES

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.compile]


def _compile(
    api=cutlass_coop,
    *,
    mode="min",
    pairs=False,
    dtype=cutlass.Int32,
    partial=False,
    block=64,
    bad=None,
    static_k=None,
):
    operation = getattr(api, f"topk_{mode}_{'pairs' if pairs else 'keys'}")

    @cute.kernel
    def kernel(memory: cute.Pointer, count: cutlass.Int64):
        keys = api.ThreadData(2, dtype=dtype)
        values = api.ThreadData(1 if bad == "extent" else 2, dtype=cutlass.Float64)
        keys[0], keys[1] = dtype(7), dtype(3)
        for item in cutlass.range_constexpr(values.items_per_thread):
            values[item] = cutlass.Float64(item)
        group = api.this_warp() if bad == "group" else api.this_block()
        k = count if static_k is None else static_k
        valid = count if partial else None
        if cutlass.const_expr(bad == "uint64"):
            k = cutlass.Uint64(count)
        if cutlass.const_expr(bad == "float"):
            k = cutlass.Float32(1)
        if cutlass.const_expr(bad == "bool"):
            k = True
        if cutlass.const_expr(bad == "count-uint64"):
            valid = cutlass.Uint64(count)
        if cutlass.const_expr(bad == "scalar"):
            keys = cutlass.Int32(3)
        if cutlass.const_expr(bad == "register"):
            keys = cute.make_rmem_tensor(2, dtype)
            keys.fill(dtype(3))
        if cutlass.const_expr(pairs):
            result, _ = operation(group, keys, values, k=k, valid_items=valid)
        else:
            result = operation(group, keys, k=k, valid_items=valid)
        output = cute.make_tensor(memory, cute.make_layout(128))
        output[0] = cutlass.Int32(result[0])

    @cute.jit
    def launch(memory: cute.Pointer, count: cutlass.Int64):
        kernel(memory, count).launch(grid=1, block=block)

    pointer = make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16)
    return cute.compile[(GPUArch("sm_80"),)](launch, pointer, cutlass.Int64(17))


@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
@pytest.mark.parametrize("mode", ("min", "max"))
@pytest.mark.parametrize("pairs", (False, True))
@pytest.mark.parametrize("partial", (False, True))
def test_entrypoints(api, mode, pairs, partial):
    assert _compile(api, mode=mode, pairs=pairs, partial=partial) is not None


@pytest.mark.parametrize("dtype", tuple(ALL_PROVIDER_TYPES))
def test_key_types(dtype):
    assert _compile(dtype=dtype, pairs=True, partial=True) is not None


@pytest.mark.parametrize("k", (0, 1, 128))
def test_static_counts(k):
    assert _compile(static_k=k) is not None


@pytest.mark.parametrize(
    "bad,pairs,message",
    (
        ("group", False, "this_block"),
        ("uint64", False, "integer"),
        ("float", False, "integer"),
        ("bool", False, "integer"),
        ("count-uint64", False, "integer"),
        ("scalar", False, "ThreadData"),
        ("extent", True, "matching"),
    ),
)
def test_invalid_profiles(bad, pairs, message):
    with pytest.raises(
        (TypeError, ValueError, NotImplementedError, DSLRuntimeError), match=message
    ):
        _compile(bad=bad, pairs=pairs)


@pytest.mark.parametrize("k", (-1, 129, 1 << 32))
def test_static_out_of_range(k):
    with pytest.raises((ValueError, DSLRuntimeError), match=r"in \[0, 128\]"):
        _compile(static_k=k)


def test_multidimensional_block_rejected():
    with pytest.raises((ValueError, DSLRuntimeError), match="one-dimensional"):
        _compile(block=(8, 4, 2))


def test_qualified_register_conversion():
    assert _compile(bad="register") is not None


def test_common_register_rejected():
    with pytest.raises((TypeError, DSLRuntimeError), match="ThreadData"):
        _compile(coop, bad="register")
