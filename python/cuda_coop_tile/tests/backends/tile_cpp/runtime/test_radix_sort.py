# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import re

from ..support.frontend import (
    RADIX_SORT_FRONTEND_SOURCE,
    assert_provider_trace_uses_requested_architecture,
    compile_and_run_frontend,
)


def test_cpp_radix_sort_real_frontend_runtime(tmp_path):
    result = compile_and_run_frontend(
        tmp_path,
        source=RADIX_SORT_FRONTEND_SOURCE,
        executable_name="tile_cpp_radix_sort_sm120",
        description="Radix Sort",
    )
    trace = result.compile_trace
    assert_provider_trace_uses_requested_architecture(result)

    assert "semantic_key=coop::cuda_coop_tile_radix_sort_keys|" in trace
    assert "semantic_key=coop::cuda_coop_tile_radix_sort_pairs|" in trace
    assert "key_dtype=f32" in trace
    key_semantics = {
        match.group(1)
        for match in re.finditer(
            r"\[coop_linkage_provider\]\[trace\] resource_estimate "
            r"semantic_key=(coop::cuda_coop_tile_radix_sort_keys\|[^\n]+?)"
            r" payload=",
            trace,
        )
    }
    assert len(key_semantics) == 1
    assert "radix_bits=4" in next(iter(key_semantics))
    assert "inner_scan_algorithm=raking" in trace
    assert "smem_config=eight_byte" in trace
    assert "PASS: C++ Tile API Radix Sort keys/pairs" in result.stdout
