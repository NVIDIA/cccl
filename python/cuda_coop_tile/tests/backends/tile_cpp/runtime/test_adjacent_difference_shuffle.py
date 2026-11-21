# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import re

from ..support.frontend import (
    ADJACENT_SHUFFLE_FRONTEND_SOURCE,
    assert_provider_trace_uses_requested_architecture,
    compile_and_run_frontend,
)


def test_cpp_adjacent_difference_shuffle_real_frontend_runtime(tmp_path):
    result = compile_and_run_frontend(
        tmp_path,
        source=ADJACENT_SHUFFLE_FRONTEND_SOURCE,
        executable_name="tile_cpp_adjacent_shuffle_sm120",
        description="Adjacent Difference/Shuffle",
    )
    trace = result.compile_trace
    assert_provider_trace_uses_requested_architecture(result)

    assert "semantic_key=coop::cuda_coop_tile_adjacent_difference|" in trace
    assert "semantic_key=coop::cuda_coop_tile_shuffle|" in trace
    assert "semantic_key=coop::cuda_coop_tile_sum_batched|" in trace
    shuffle_semantics = {
        match.group(1)
        for match in re.finditer(
            r"\[coop_linkage_provider\]\[trace\] resource_estimate "
            r"semantic_key=(coop::cuda_coop_tile_shuffle\|[^\n]+?) payload=",
            trace,
        )
    }
    assert len(shuffle_semantics) == 6
    assert all("distance=" not in key for key in shuffle_semantics)
    assert (
        "PASS: C++ Tile API AdjacentDifference/Shuffle/Reduce" in result.stdout
    )
