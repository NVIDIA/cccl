# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import re

from ..support.frontend import (
    RUN_LENGTH_DECODE_FRONTEND_SOURCE,
    assert_provider_trace_uses_requested_architecture,
    compile_and_run_frontend,
)


def test_cpp_run_length_decode_real_frontend_runtime(tmp_path):
    result = compile_and_run_frontend(
        tmp_path,
        source=RUN_LENGTH_DECODE_FRONTEND_SOURCE,
        executable_name="tile_cpp_run_length_decode_sm120",
        description="Run-Length Decode",
    )
    trace = result.compile_trace
    assert_provider_trace_uses_requested_architecture(result)

    assert "semantic_key=coop::cuda_coop_tile_run_length_decode|" in trace
    assert (
        "semantic_key=coop::cuda_coop_tile_run_length_decode_with_offsets|"
        in trace
    )
    run_length_decode_semantics = {
        match.group(1)
        for match in re.finditer(
            r"\[coop_linkage_provider\]\[trace\] resource_estimate "
            r"semantic_key=(coop::cuda_coop_tile_run_length_decode"
            r"(?:_with_offsets)?\|[^\n]+?) payload=",
            trace,
        )
    }
    assert run_length_decode_semantics == {
        (
            "coop::cuda_coop_tile_run_length_decode|"
            "block_threads=128;decoded_items_per_thread=2;length_dtype=u32;"
            "runs_per_thread=1;value_dtype=i32"
        ),
        (
            "coop::cuda_coop_tile_run_length_decode_with_offsets|"
            "block_threads=128;decoded_items_per_thread=2;length_dtype=u32;"
            "runs_per_thread=1;value_dtype=i32"
        ),
        (
            "coop::cuda_coop_tile_run_length_decode_with_offsets|"
            "block_threads=128;decoded_items_per_thread=2;length_dtype=u64;"
            "runs_per_thread=1;value_dtype=i32"
        ),
    }
    assert all(
        "decoded_window_offset" not in key
        for key in run_length_decode_semantics
    )
    assert "PASS: C++ Tile API Run-Length Decode" in result.stdout
