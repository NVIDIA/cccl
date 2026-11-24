# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from ..support.frontend import (
    CPP_API_FRONTEND_SOURCE,
    assert_provider_trace_uses_requested_architecture,
    compile_and_run_frontend,
)


def test_cpp_api_large_logical_grid(tmp_path):
    result = compile_and_run_frontend(
        tmp_path,
        source=CPP_API_FRONTEND_SOURCE,
        executable_name="tile_cpp_api_grid_stress_sm120",
        description="API logical-grid stress",
        run_args=("--tile-blocks", "4096", "--iterations", "10"),
        run_timeout=300,
    )

    assert_provider_trace_uses_requested_architecture(result)
    assert (
        "PASS: C++ Tile API Scan/Reduce/TopK/MergeSort/Histogram/"
        "Discontinuity/Exchange" in result.stdout
    )
    assert "blocks=4096 iterations=10" in result.stdout
