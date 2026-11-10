# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..support.frontend import (
    FRONTEND_SOURCE_ROOT,
    assert_provider_trace_uses_requested_architecture,
    compile_and_run_frontend,
)


def test_cpp_scan_sum_and_generic_reduce_preserve_unsigned_values(tmp_path):
    result = compile_and_run_frontend(
        tmp_path,
        source=FRONTEND_SOURCE_ROOT / "tile_cpp_scan_reduce.cu",
        executable_name="scan_reduce",
        description="scan/reduce",
    )
    assert_provider_trace_uses_requested_architecture(result)
    assert "PASS: unsigned scan, sum, and generic reduction" in result.stdout
    for family in ("scan", "sum_batched", "reduce_batched"):
        assert (
            f"semantic_key=coop::cuda_coop_tile_{family}|"
            in result.compile_trace
        )


def test_cpp_scan_prefixes_match_independent_host_fold(tmp_path):
    result = compile_and_run_frontend(
        tmp_path,
        source=FRONTEND_SOURCE_ROOT / "tile_cpp_scan_prefix.cu",
        executable_name="scan_prefix",
        description="scan prefixes",
    )
    assert_provider_trace_uses_requested_architecture(result)
    assert (
        "PASS: scalar and Tile scan prefixes at extents 128 and 512"
        in result.stdout
    )
