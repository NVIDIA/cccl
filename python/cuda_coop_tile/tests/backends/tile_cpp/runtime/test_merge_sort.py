# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import re

from ..support.frontend import (
    FRONTEND_SOURCE_ROOT,
    assert_provider_trace_uses_requested_architecture,
    compile_and_run_frontend,
)


def test_cpp_merge_sort_pairs_partial_tiles_and_warp_forms(tmp_path):
    result = compile_and_run_frontend(
        tmp_path,
        source=FRONTEND_SOURCE_ROOT / "tile_cpp_merge_sort.cu",
        executable_name="merge_sort",
        description="Merge Sort keys/pairs, partial tiles, and warp forms",
    )
    assert_provider_trace_uses_requested_architecture(result)
    assert (
        "PASS: C++ Merge Sort keys/pairs, partial tiles, and warp forms"
        in result.stdout
    )
    for family in ("merge_sort_keys", "merge_sort_pairs"):
        for partial in (0, 1):
            assert re.search(
                rf"semantic_key=coop::cuda_coop_tile_{family}\|[^\n]*partial_tile={partial}",
                result.compile_trace,
            )
    assert "threads=32 ipt=1" in result.compile_trace
