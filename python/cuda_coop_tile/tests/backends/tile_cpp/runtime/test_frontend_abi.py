# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import re

from ..support.frontend import (
    FRONTEND_SOURCE_ROOT,
    assert_provider_trace_uses_requested_architecture,
    compile_and_run_frontend,
)

FRONTEND_ABI_SOURCE = FRONTEND_SOURCE_ROOT / "tile_cpp_frontend_abi_sm120.cu"


def test_cpp_frontend_abi_real_frontend_runtime(tmp_path):
    result = compile_and_run_frontend(
        tmp_path,
        source=FRONTEND_ABI_SOURCE,
        executable_name="tile_cpp_frontend_abi_sm120",
        description="frontend ABI",
    )
    assert_provider_trace_uses_requested_architecture(result)

    for dtype in ("f16", "bf16"):
        assert re.search(
            rf"semantic_key=coop::cuda_coop_tile_sum_batched\|[^\n]*"
            rf"[|;]dtype={dtype}(?:;|\s)",
            result.compile_trace,
        )

    assert "PASS: C++ Tile frontend ABI f16/bf16" in result.stdout
