# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest


def test_warp_scan_requests_follow_core_semantics():
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Int32

    from cuda.coop.cutlass._dsl.warp import _provider as provider

    scan = provider._warp_scan_request(
        request_kind="warp_exclusive_scan",
        value_type=Int32,
        op="max",
        threads_in_warp=16,
        has_valid_items=True,
        has_warp_aggregate=True,
    )
    assert scan.kind == "warp_exclusive_scan"
    assert scan.op == "max"
    assert scan.value_type is Int32
    assert scan.logical_warp_threads == 16
    assert scan.has_valid_items
    assert scan.has_warp_aggregate

    partial_sum = provider._warp_scan_request(
        request_kind="warp_inclusive_sum",
        value_type=Int32,
        op="sum",
        threads_in_warp=32,
        has_valid_items=True,
        has_warp_aggregate=False,
    )
    assert partial_sum.kind == "warp_inclusive_sum"
    assert partial_sum.op is None
    assert partial_sum.has_valid_items
