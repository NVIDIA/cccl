# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""How ``ThreadData`` inference resolves 28 payload-dtype cases.

Each case is compiled, without a launch, for a block of 64 threads and four
items per thread. A case either compiles, and its payload has the expected
type, or fails with the expected diagnostic. Producers determine payload
types before reads and writes. Without a producer, writes must agree and
Store validates their dtype; Store supplies a dtype only without writes.

Calls to cooperative primitives must inline into the kernel. A separately
compiled device function has no launch configuration for cooperative
planning and is rejected on both the planner and typed paths.
"""

from __future__ import annotations

import re
from typing import NamedTuple

import numpy as np
import pytest
from numba_cuda_mlir import cuda, types

from cuda import coop
from cuda.coop.numba_mlir._pipeline import testing

from .._support import CHIP, requires_typed_pipeline

pytestmark = requires_typed_pipeline

ITEMS = 4
BLOCK = (64, 1, 1)


@cuda.jit(device=True)
def scale(td):
    for i in range(ITEMS):
        td[i] = td[i] * 2


@cuda.jit(device=True)
def fill(td, src):
    base = cuda.threadIdx.x * ITEMS
    for i in range(ITEMS):
        td[i] = src[base + i]


@cuda.jit(device=True)
def inner_load(src, td):
    coop.load(coop.this_block(), src, td)


@cuda.jit(device=True, inline="never")
def scale_ni(td):
    for i in range(ITEMS):
        td[i] = td[i] * 2


@cuda.jit(device=True, inline="never")
def fill_ni(td, src):
    base = cuda.threadIdx.x * ITEMS
    for i in range(ITEMS):
        td[i] = src[base + i]


@cuda.jit(device=True, inline="never")
def inner_load_ni(src, td):
    coop.load(coop.this_block(), src, td)


@cuda.jit(device=True)
def apply_op(op, td):
    for i in range(ITEMS):
        td[i] = op(td[i], td[i])


def _plain_max(a, b):
    return max(b, a)


def k1(source, output):
    block = coop.this_block()
    items = coop.ThreadData(ITEMS)
    coop.load(block, source, items)
    for i in range(ITEMS):
        items[i] = items[i] * 2
    coop.store(block, output, items)


def kw(src, dst):
    block = coop.this_block()
    td = coop.ThreadData(items_per_thread=ITEMS, alignment=16)
    coop.load(block, src, td, algorithm="direct")
    coop.store(block, dst, td, algorithm="direct", valid_items=None)


def kw_positional(src, dst):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    coop.load(block, src, td, "direct")
    coop.store(block, dst, td)


def phi_plain(src, dst, flag):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    la = cuda.local.array(ITEMS, types.float32)
    coop.load(block, src, td)
    for i in range(ITEMS):
        la[i] = td[i]
    if flag[0]:
        x = td
    else:
        x = la
    coop.store(block, dst, x)


def phi_plain_ld(src, dst, flag):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    la = cuda.local.array(ITEMS, types.float32)
    if flag[0]:
        x = td
    else:
        x = la
    coop.load(block, src, x)
    coop.store(block, dst, td)


def c3(src, dst, flag):
    block = coop.this_block()
    a = coop.ThreadData(ITEMS)
    b = coop.ThreadData(ITEMS)
    if flag[0]:
        x = a
    else:
        x = b
    coop.load(block, src, x)
    coop.store(block, dst, x)


def h_after(src, dst):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    coop.load(block, src, td)
    scale(td)
    coop.store(block, dst, td)


def h_before(src, dst):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    for it in range(3):
        if it == 1:
            scale(td)
        elif it == 0:
            coop.load(block, src, td)
        else:
            coop.store(block, dst, td)


def h_fill(src, dst):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    fill(td, src)
    coop.store(block, dst, td)


def h_fill_nostore(src, dst):
    td = coop.ThreadData(ITEMS)
    fill(td, src)
    base = cuda.threadIdx.x * ITEMS
    for i in range(ITEMS):
        dst[base + i] = td[i]


def h_inner_load(src, dst):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    inner_load(src, td)
    coop.store(block, dst, td)


def ni_after(src, dst):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    coop.load(block, src, td)
    scale_ni(td)
    coop.store(block, dst, td)


def ni_before(src, dst):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    for it in range(3):
        if it == 1:
            scale_ni(td)
        elif it == 0:
            coop.load(block, src, td)
        else:
            coop.store(block, dst, td)


def ni_fill(src, dst):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    fill_ni(td, src)
    coop.store(block, dst, td)


def ni_fill_nostore(src, dst):
    td = coop.ThreadData(ITEMS)
    fill_ni(td, src)
    base = cuda.threadIdx.x * ITEMS
    for i in range(ITEMS):
        dst[base + i] = td[i]


def ni_inner_load(src, dst):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    inner_load_ni(src, td)
    coop.store(block, dst, td)


def ni_inner_load_only(src, dst):
    td = coop.ThreadData(ITEMS)
    inner_load_ni(src, td)
    base = cuda.threadIdx.x * ITEMS
    for i in range(ITEMS):
        dst[base + i] = td[i]


def e1(src_i, dst):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    coop.load(block, src_i, td)
    td[0] = 1.5
    coop.store(block, dst, td)


def e2(src_i, dst):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    td[0] = 1.5
    coop.load(block, src_i, td)
    coop.store(block, dst, td)


def e3a(src_i, dst_f):
    td = coop.ThreadData(ITEMS)
    base = cuda.threadIdx.x * ITEMS
    for i in range(1, ITEMS):
        td[i] = src_i[base + i]
    td[0] = 2.5
    for i in range(ITEMS):
        dst_f[base + i] = td[i]


def e3b(src_i, dst_f):
    td = coop.ThreadData(ITEMS)
    base = cuda.threadIdx.x * ITEMS
    td[0] = 2.5
    for i in range(1, ITEMS):
        td[i] = src_i[base + i]
    for i in range(ITEMS):
        dst_f[base + i] = td[i]


def d1(src_i, src_f, dst):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    coop.load(block, src_i, td)
    coop.load(block, src_f, td)
    coop.store(block, dst, td)


def d3(src_i, dst_f):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    coop.load(block, src_i, td)
    coop.store(block, dst_f, td)


def d_explicit(src_f, dst):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS, dtype=types.int32)
    coop.load(block, src_f, td)
    coop.store(block, dst, td)


def w_store(src_i, dst_i):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    for i in range(ITEMS):
        td[i] = i
    coop.store(block, dst_i, td)


def k4_extent(src, dst, items_per_thread):
    block = coop.this_block()
    td = coop.ThreadData(items_per_thread)
    coop.load(block, src, td)
    coop.store(block, dst, td)


def align(src, dst):
    block = coop.this_block()
    a = coop.ThreadData(ITEMS, alignment=32)
    b = coop.ThreadData(ITEMS, alignment=8)
    alias = a
    coop.load(block, src, alias)
    coop.load(block, src, b)
    scale_ni(a)
    scale_ni(b)
    for i in range(ITEMS):
        a[i] = a[i] + b[i] - b[i]
    coop.store(block, dst, a)


def cb_plain(src, dst):
    block = coop.this_block()
    td = coop.ThreadData(ITEMS)
    coop.load(block, src, td)
    apply_op(_plain_max, td)
    coop.store(block, dst, td)


F32 = types.float32[::1]
F64 = types.float64[::1]
I32 = types.int32[::1]


def _payload(dtype, strength):
    return f"coop.ThreadData({dtype}, items={ITEMS}, {strength})"


class Case(NamedTuple):
    kernel: object
    args: tuple
    payload: tuple[str, str] | None = None
    error: str | None = None


F32_PRODUCER = _payload("float32", "producer")
WRITES_DISAGREE = (
    r"Failed to infer a consistent dtype from coop\.ThreadData writes"
)
CASES = {
    "k1": Case(k1, (F32, F32), ("items", F32_PRODUCER)),
    "kw": Case(kw, (F32, F32), ("td", F32_PRODUCER)),
    "kw_positional": Case(kw_positional, (F32, F32), ("td", F32_PRODUCER)),
    # Which side of the join Numba types first depends on what the process
    # has already compiled; a plain array first unifies to a plain array,
    # which the common API's store refuses.
    "phi_plain": Case(
        phi_plain,
        (F32, F32, I32),
        error=(
            r"cannot share a variable with a plain array"
            r"|ThreadData value payload in the common API"
        ),
    ),
    "phi_plain_ld": Case(
        phi_plain_ld,
        (F32, F32, I32),
        error=(
            r"cannot share a variable with a plain array"
            r"|ThreadData payload in the common API"
        ),
    ),
    "c3": Case(c3, (F32, F32, I32), ("b", F32_PRODUCER)),
    "h_after": Case(h_after, (F32, F32), ("td", F32_PRODUCER)),
    "h_before": Case(h_before, (F32, F32), ("td", F32_PRODUCER)),
    "h_fill": Case(
        h_fill, (F32, F32), ("td", _payload("float32", "store-dst"))
    ),
    "h_fill_nostore": Case(
        h_fill_nostore, (F32, F32), ("td", _payload("float32", "write"))
    ),
    "h_inner_load": Case(h_inner_load, (F32, F32), ("td", F32_PRODUCER)),
    "ni_after": Case(ni_after, (F32, F32), ("td", F32_PRODUCER)),
    "ni_before": Case(ni_before, (F32, F32), ("td", F32_PRODUCER)),
    "ni_fill": Case(
        ni_fill, (F32, F32), ("td", _payload("float32", "store-dst"))
    ),
    "ni_fill_nostore": Case(
        ni_fill_nostore,
        (F32, F32),
        error=r"undefined",
    ),
    "ni_inner_load": Case(
        ni_inner_load,
        (F32, F32),
        error=r"separately .*compiled device function.*no launch to plan",
    ),
    "ni_inner_load_only": Case(
        ni_inner_load_only,
        (F32, F32),
        error=r"undefined",
    ),
    "e1": Case(e1, (I32, I32), ("td", _payload("int32", "producer"))),
    "e2": Case(e2, (I32, I32), ("td", _payload("int32", "producer"))),
    "e3a": Case(e3a, (I32, F64), error=WRITES_DISAGREE),
    "e3b": Case(e3b, (I32, F64), error=WRITES_DISAGREE),
    "d1": Case(d1, (I32, F32, I32), error=r"conflicts with source dtype"),
    "d3": Case(
        d3,
        (I32, F32),
        error=r"payload dtype int32 conflicts with destination dtype float32",
    ),
    "d_explicit": Case(
        d_explicit, (F32, F32), error=r"conflicts with source dtype float32"
    ),
    "w_store": Case(
        w_store,
        (I32, I32),
        error=r"payload dtype int64 conflicts with destination dtype int32",
    ),
    "k4_extent": Case(
        k4_extent,
        (F32, F32, types.IntegerLiteral(ITEMS)),
        ("td", F32_PRODUCER),
    ),
    "align": Case(align, (F32, F32), ("a", F32_PRODUCER)),
    "cb_plain": Case(
        cb_plain, (F32, F32), error=r"Untyped global name '_plain_max'"
    ),
}


def test_the_table_holds_t4s_28_cases():
    assert len(CASES) == 28


@pytest.mark.parametrize("name", CASES)
def test_case(name):
    case = CASES[name]
    signature = types.void(*case.args)
    if case.error is not None:
        with pytest.raises(Exception, match=re.compile(case.error, re.DOTALL)):
            testing.compile_kernel(
                case.kernel, signature, block=BLOCK, chip=CHIP
            )
        return
    compiled = testing.compile_kernel(
        case.kernel, signature, block=BLOCK, chip=CHIP
    )
    variable, payload = case.payload
    pattern = rf"^\s*{variable} = call .* :: {re.escape(payload)}$"
    typed_ir = compiled.typed_ir
    assert re.search(pattern, typed_ir, re.MULTILINE), typed_ir


def test_a_load_through_a_tuple_types_the_payload():
    """A tuple projection refines the payload it projects."""

    def tuple_shape(src, dst):
        block = coop.this_block()
        td = coop.ThreadData(ITEMS)
        coop.load(block, src, (td,)[0])
        base = cuda.threadIdx.x * ITEMS
        for i in range(ITEMS):
            dst[base + i] = td[i]

    compiled = testing.compile_kernel(
        tuple_shape, types.void(F32, F32), block=BLOCK, chip=CHIP
    )
    pattern = rf"^\s*td = call .* :: {re.escape(F32_PRODUCER)}$"
    assert re.search(pattern, compiled.typed_ir, re.MULTILINE)


def test_captured_numpy_controls_keep_ordinary_dtype_and_call_alignment():
    count = np.int32(2)
    wide_alignment, narrow_alignment = np.int64(32), np.int64(8)

    @cuda.jit(device=True, inline="never")
    def identity(value):
        return value

    def kernel(source, output):
        first = coop.ThreadData(count, alignment=wide_alignment)
        second = coop.ThreadData(count, alignment=narrow_alignment)
        coop.load(coop.this_block(), source, first)
        coop.load(coop.this_block(), source, second)
        observed = identity(count)
        output[cuda.threadIdx.x] = first[0] + second[0] + observed

    compiled = testing.compile_kernel(
        kernel, types.void(I32, I32), block=BLOCK, chip=CHIP
    )
    assert re.search(
        r"^\s*observed = call .* :: int32$", compiled.typed_ir, re.MULTILINE
    )
    for alignment in (32, 8):
        assert re.search(
            rf"memref.alloca.*alignment = {alignment} : i64", compiled.mlir
        )
