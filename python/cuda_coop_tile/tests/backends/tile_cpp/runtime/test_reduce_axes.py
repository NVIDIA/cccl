# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..support.frontend import (
    FRONTEND_SOURCE_ROOT,
    assert_provider_trace_uses_requested_architecture,
    compile_and_run_frontend,
)


def test_cpp_axis_reductions_compose_across_worker_counts(tmp_path):
    result = compile_and_run_frontend(
        tmp_path,
        source=FRONTEND_SOURCE_ROOT / "tile_cpp_reduce_axes.cu",
        executable_name="reduce_axes",
        description="axis reductions",
    )
    assert_provider_trace_uses_requested_architecture(result)
    assert (
        "PASS: C++ axis reductions and legacy broadcasts "
        "at automatic/4/8/16 worker warps" in result.stdout
    )
    for family in ("sum_batched", "reduce_batched"):
        assert (
            f"semantic_key=coop::cuda_coop_tile_{family}|"
            in result.compile_trace
        )
