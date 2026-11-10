# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import re

from ..support.frontend import (
    CPP_API_FRONTEND_SOURCE,
    assert_provider_trace_scored_selected_implementations,
    assert_provider_trace_uses_requested_architecture,
    compile_and_run_frontend,
)


def test_cpp_api_real_frontend_runtime(tmp_path):
    result = compile_and_run_frontend(
        tmp_path,
        source=CPP_API_FRONTEND_SOURCE,
        executable_name="tile_cpp_api_sm120",
        description="API",
        run_args=("--tile-blocks", "7", "--iterations", "3"),
    )
    trace = result.compile_trace
    assert_provider_trace_uses_requested_architecture(result)
    assert_provider_trace_scored_selected_implementations(result)

    expected_dtype_fields = (
        ("scan", "dtype", "i32"),
        ("sum_batched", "dtype", "u32"),
        ("topk_pairs", "value_dtype", "u64"),
        ("histogram", "dtype", "u8"),
    )
    for implementation, field, dtype in expected_dtype_fields:
        assert re.search(
            rf"semantic_key=coop::cuda_coop_tile_{implementation}\|[^\n]*"
            rf"[|;]{field}={dtype}(?:;|\s)",
            trace,
        )
    for implementation in (
        "histogram",
        "histogram_256",
        "discontinuity_flags",
        "discontinuity_heads_and_tails",
        "exchange",
        "merge_sort_keys",
        "merge_sort_pairs",
    ):
        assert f"semantic_key=coop::cuda_coop_tile_{implementation}|" in trace

    initial_scan_semantics = {
        match.group(1)
        for match in re.finditer(
            r"\[coop_linkage_provider\]\[trace\] resource_estimate "
            r"semantic_key=(coop::cuda_coop_tile_scan\|[^\n]+?) payload=",
            trace,
        )
        if "initial_present=1" in match.group(1)
        and "scan_op=sum" in match.group(1)
    }
    assert len(initial_scan_semantics) == 1
    assert "initial_value=" not in next(iter(initial_scan_semantics))
    assert "initial_value=3" not in trace
    assert "initial_value=9" not in trace

    merge_sort_semantics = {
        match.group(1)
        for match in re.finditer(
            r"semantic_key=(coop::cuda_coop_tile_merge_sort_(?:keys|pairs)"
            r"\|[^\n]+?) payload=",
            trace,
        )
    }
    assert any(
        "merge_sort_keys|" in key
        and "descending=0" in key
        and "partial_tile=0" in key
        for key in merge_sort_semantics
    )
    assert any(
        "merge_sort_keys|" in key
        and "descending=0" in key
        and "partial_tile=1" in key
        for key in merge_sort_semantics
    )
    assert any(
        "merge_sort_pairs|" in key
        and "descending=1" in key
        and "partial_tile=0" in key
        for key in merge_sort_semantics
    )
    assert (
        "PASS: C++ Tile API Scan/Reduce/TopK/MergeSort/Histogram/"
        "Discontinuity/Exchange" in result.stdout
    )
    assert "blocks=7 iterations=3" in result.stdout
