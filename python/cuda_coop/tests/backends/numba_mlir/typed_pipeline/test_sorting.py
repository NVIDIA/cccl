# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile result ownership, shape propagation, and Radix control boundaries."""

from types import SimpleNamespace

import numba_cuda_mlir.tools as compiler_tools
import pytest
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.numba_cuda.core.errors import TypingError

import cuda.coop.numba_mlir as numba_coop
from cuda import coop

from ._support import requires_typed_pipeline

pytestmark = requires_typed_pipeline


@pytest.fixture(autouse=True)
def _hidden_target(monkeypatch):
    monkeypatch.setattr(
        compiler_tools,
        "get_gpu_compute_capability",
        lambda as_type=str: (9, 0) if as_type is tuple else "sm_90",
    )
    monkeypatch.setattr(
        cuda,
        "get_current_device",
        lambda: SimpleNamespace(compute_capability=(9, 0)),
    )


def _compile(kernel, signature):
    result = cuda.jit(chip="sm_90")(kernel)._compile_launch_config_signature(
        signature,
        (
            ("grid", (1, 1, 1)),
            ("block", (64, 1, 1)),
            ("sharedmem", 0),
            ("cluster", None),
        ),
    )
    assert result.metadata["cubin"]
    return result


def test_qualified_scalar_radix_results_and_inferred_prefix():
    def kernel(keys, values, output, payload, ranks, prefixes, begin):
        group = coop.this_block()
        t = cuda.threadIdx.x
        pair = numba_coop.radix_sort_pairs(group, keys[t], values[t])
        output[t] = pair[0]
        payload[t] = pair[1]
        prefix = coop.ThreadData(2)
        ranked = numba_coop.radix_rank_keys(
            group,
            keys[t],
            begin_bit=begin,
            radix_bits=7,
            exclusive_digit_prefix=prefix,
        )
        ranks[t] = ranked
        prefixes[t] = prefix[0]

    _compile(
        kernel,
        types.void(
            types.uint64[::1],
            types.int8[::1],
            types.uint64[::1],
            types.int8[::1],
            types.int32[::1],
            types.int32[::1],
            types.int64,
        ),
    )


def test_static_radix_intervals_share_provider_and_keep_call_operands():
    def kernel(keys, output):
        group = coop.this_block()
        value = coop.ThreadData(1)
        coop.load(group, keys, value)
        first = coop.radix_sort_keys(group, value, begin_bit=0, end_bit=4)
        second = coop.radix_sort_keys(group, value, begin_bit=4, end_bit=8)
        output[cuda.threadIdx.x] = first[0] + second[0]

    result = _compile(kernel, types.void(types.uint32[::1], types.uint32[::1]))
    calls = [
        call
        for call in result.metadata["coop_plan"].calls.values()
        if "radix_sort" in call.symbol
    ]
    assert len(calls) == 2 and calls[0].symbol == calls[1].symbol
    assert [call.operands[-2:] for call in calls] == [
        (("constant", 0), ("constant", 4)),
        (("constant", 4), ("constant", 8)),
    ]


def test_merge_comparator_links_with_infinite_partial_sentinel():
    sentinel = float("-inf")

    def greater(a, b):
        return a > b

    def kernel(source, output, count):
        group = coop.this_block()
        keys = coop.ThreadData(2)
        coop.load(group, source, keys)
        result = numba_coop.merge_sort_keys(
            group,
            keys,
            compare_op=greater,
            valid_items=count,
            oob_default=sentinel,
        )
        coop.store(group, output, result)

    _compile(
        kernel, types.void(types.float64[::1], types.float64[::1], types.int64)
    )


@pytest.mark.parametrize("dtype", [types.int32, types.int64])
def test_merge_literal_argument_retains_its_sentinel_dtype(dtype):
    def kernel(source, output, sentinel):
        group = coop.this_block()
        keys = coop.ThreadData(2)
        coop.load(group, source, keys)
        result = coop.merge_sort_keys(
            group, keys, valid_items=64, oob_default=sentinel
        )
        coop.store(group, output, result)

    signature = types.void(dtype[::1], dtype[::1], types.IntegerLiteral(1000))
    if dtype == types.int32:
        with pytest.raises(TypingError, match="oob_default dtype int64"):
            _compile(kernel, signature)
    else:
        _compile(kernel, signature)
