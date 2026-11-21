# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import re

from ..support.frontend import (
    RADIX_RANK_FRONTEND_SOURCE,
    assert_provider_trace_uses_requested_architecture,
    compile_and_run_frontend,
)


def test_cpp_radix_rank_real_frontend_runtime(tmp_path):
    result = compile_and_run_frontend(
        tmp_path,
        source=RADIX_RANK_FRONTEND_SOURCE,
        executable_name="tile_cpp_radix_rank_sm120",
        description="Radix Rank",
    )
    trace = result.compile_trace
    assert_provider_trace_uses_requested_architecture(result)

    assert "semantic_key=coop::cuda_coop_tile_radix_rank|" in trace
    radix_rank_semantics = {
        match.group(1)
        for match in re.finditer(
            r"\[coop_linkage_provider\]\[trace\] resource_estimate "
            r"semantic_key=(coop::cuda_coop_tile_radix_rank\|[^\n]+?) payload=",
            trace,
        )
    }
    assert len(radix_rank_semantics) == 4
    assert all(
        "begin_bit" not in key and "end_bit" not in key
        for key in radix_rank_semantics
    )
    assert any(
        "descending=0" in key
        and "inner_scan_algorithm=warp_scans" in key
        and "smem_config=four_byte" in key
        for key in radix_rank_semantics
    )
    assert any(
        "descending=1" in key
        and "inner_scan_algorithm=raking" in key
        and "smem_config=eight_byte" in key
        for key in radix_rank_semantics
    )
    assert any(
        "descending=0" in key and "dtype=i8" in key and "radix_bits=4" in key
        for key in radix_rank_semantics
    )
    assert any(
        "descending=1" in key and "dtype=i8" in key and "radix_bits=4" in key
        for key in radix_rank_semantics
    )
    assert "PASS: C++ Tile API Radix Rank" in result.stdout
