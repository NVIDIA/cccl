# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest


def test_warp_reduce_request_follows_core_semantics():
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Int32

    from cuda.coop.cutlass._dsl.warp import _provider as provider

    reduction = provider._warp_reduce_request(
        request_kind="warp_reduce",
        value_type=Int32,
        op="bit_xor",
        threads_in_warp=8,
        has_valid_items=True,
        items_per_thread=2,
    )
    assert reduction.kind == "warp_reduce"
    assert reduction.op == "bit_xor"
    assert reduction.value_type is Int32
    assert reduction.logical_warp_threads == 8
    assert reduction.has_valid_items
    assert reduction.items_per_thread == 2

    minimum = provider._warp_reduce_request(
        request_kind="warp_reduce",
        value_type=Int32,
        op="min",
        threads_in_warp=16,
        has_valid_items=False,
    )
    assert minimum.kind == "warp_reduce"
    assert minimum.op == "min"

    with pytest.raises(ValueError, match="does not match"):
        provider._warp_reduce_request(
            request_kind="warp_sum",
            value_type=Int32,
            op="max",
            threads_in_warp=32,
            has_valid_items=False,
        )
