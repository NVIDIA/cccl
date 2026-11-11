# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..support.frontend import (
    FRONTEND_SOURCE_ROOT,
    assert_provider_trace_uses_requested_architecture,
    compile_and_run_frontend,
)


def test_cpp_histogram_256_matches_independent_oracle(tmp_path):
    result = compile_and_run_frontend(
        tmp_path,
        source=FRONTEND_SOURCE_ROOT / "tile_cpp_histogram_256_smoke.cu",
        executable_name="histogram_256",
        description="histogram_256",
    )
    assert_provider_trace_uses_requested_architecture(result)
    assert "PASS: histogram_256" in result.stdout
    assert (
        "semantic_key=coop::cuda_coop_tile_histogram_256|"
        in result.compile_trace
    )
