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


def test_cpp_topk_keys_pairs_full_partial_runtime(tmp_path):
    result = compile_and_run_frontend(
        tmp_path,
        source=FRONTEND_SOURCE_ROOT / "tile_cpp_topk_sm120.cu",
        executable_name="tile_cpp_topk_sm120",
        description="TopK keys/pairs full/partial",
    )
    assert_provider_trace_uses_requested_architecture(result)
    semantics = {
        match.group(1)
        for match in re.finditer(
            (
                r"\[coop_linkage_provider\]\[trace\] resource_estimate "
                "semantic_key=(coop::cuda_coop_tile_topk_(?:keys|pairs"
                ")\\|[^\\n]+?) payload="
            ),
            result.compile_trace,
        )
    }
    for family in ("keys", "pairs"):
        for threads in (128, 256):
            for descending in (0, 1):
                for full_tile in (0, 1):
                    assert any(
                        f"cuda_coop_tile_topk_{family}|" in key
                        and f"block_threads={threads};" in key
                        and f"descending={descending};" in key
                        and f"full_tile={full_tile};" in key
                        for key in semantics
                    ), (family, threads, descending, full_tile, semantics)
    assert (
        "PASS: C++ Tile API TopK keys/pairs full/partial 128/256"
        in result.stdout
    )
