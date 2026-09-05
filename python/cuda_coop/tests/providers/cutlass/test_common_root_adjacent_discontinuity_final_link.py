# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Final-link evidence for common-root CUTLASS comparison collectives."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ._ltoir_support import (
    _assert_ltoir_inlined,
    _configure_dump_environment,
    _read_one,
    _require_runtime,
    _run_example_subprocess,
)

pytestmark = [pytest.mark.gpu, pytest.mark.link]

_EXPECTED_SYMBOLS = (
    "cuda_coop_cutlass_adjacent_difference_b8x4x2_"
    "subtract_left_i32_x2_partial_predecessor_external_scratch",
    "cuda_coop_cutlass_adjacent_difference_b8x4x2_"
    "subtract_right_i32_x2_successor_external_scratch",
    "cuda_coop_cutlass_discontinuity_b8x4x2_heads_i32_x2_predecessor_external_scratch",
    "cuda_coop_cutlass_discontinuity_b8x4x2_tails_i32_x2_successor_external_scratch",
)


@pytest.mark.evidence_for(
    "group.adjacent_difference", backend="cutlass", evidence="link"
)
@pytest.mark.evidence_for("group.discontinuity", backend="cutlass", evidence="link")
def test_common_and_qualified_comparison_wrappers_are_eliminated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_dump_environment(monkeypatch, tmp_path)
    _require_runtime()

    _run_example_subprocess("_group_adjacent_discontinuity_codegen_probe")

    provider_source = _read_one(
        "bundle/cuda_coop_cutlass_bundle_*.cpp",
        tmp_path=tmp_path,
    )
    provider_symbols = tuple(
        sorted(
            set(
                re.findall(
                    r"\b(cuda_coop_cutlass_(?:adjacent_difference|discontinuity)_"
                    r"[A-Za-z0-9_]+)\(",
                    provider_source,
                )
            )
        )
    )
    assert provider_symbols == tuple(sorted(_EXPECTED_SYMBOLS))

    clean_mlir = _read_one("dsl/*_clean.mlir", tmp_path=tmp_path)
    for symbol in _EXPECTED_SYMBOLS:
        assert provider_source.count(f"{symbol}(") == 1
        # Common and qualified calls must coalesce to one specialization.
        assert clean_mlir.count(f"func.call @{symbol}") == 2

    sass = _assert_ltoir_inlined(
        tmp_path=tmp_path,
        expected_symbols=_EXPECTED_SYMBOLS,
    )
    assert re.search(r"\b(?:CALL|LDL|STL)(?:\.[A-Z0-9_]+)*\b", sass) is None
