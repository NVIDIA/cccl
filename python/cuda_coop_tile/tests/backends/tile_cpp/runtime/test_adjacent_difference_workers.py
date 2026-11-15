# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..support.frontend import (
    FRONTEND_SOURCE_ROOT,
    assert_provider_trace_uses_requested_architecture,
    compile_and_run_frontend,
)


def test_cpp_small_adjacent_difference_worker_selection(tmp_path):
    result = compile_and_run_frontend(
        tmp_path,
        source=FRONTEND_SOURCE_ROOT / "tile_cpp_adjacent_difference_workers.cu",
        executable_name="adjacent_difference_workers",
        description="small Adjacent Difference worker selection",
    )
    assert_provider_trace_uses_requested_architecture(result)
    assert (
        "PASS: small Adjacent Difference with default and four worker warps"
        in result.stdout
    )
