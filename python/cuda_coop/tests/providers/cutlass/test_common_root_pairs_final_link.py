# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Final-link evidence for common CUTLASS pair collectives."""

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


@pytest.mark.evidence_for("group.merge_sort_pairs", backend="cutlass", evidence="link")
@pytest.mark.evidence_for("group.radix_sort_pairs", backend="cutlass", evidence="link")
@pytest.mark.evidence_for("group.topk_max_pairs", backend="cutlass", evidence="link")
@pytest.mark.evidence_for("group.topk_min_pairs", backend="cutlass", evidence="link")
def test_common_and_qualified_pair_wrappers_are_eliminated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_dump_environment(monkeypatch, tmp_path)
    _require_runtime()
    _run_example_subprocess("_common_root_pairs_codegen_probe")

    provider_source = _read_one(
        "bundle/cuda_coop_cutlass_bundle_*.cpp", tmp_path=tmp_path
    )
    provider_symbols = tuple(
        symbol
        for symbol in sorted(
            set(
                re.findall(
                    r"\b(cuda_coop_cutlass_[A-Za-z0-9_]+)\(",
                    provider_source,
                )
            )
        )
        if "_pairs_" in symbol
        or "radix_sort_pairs" in symbol
        or ("_topk_" in symbol and "_pair_" in symbol)
    )
    assert len(provider_symbols) == 10
    clean_mlir = _read_one("dsl/*_clean.mlir", tmp_path=tmp_path)
    for symbol in provider_symbols:
        assert provider_source.count(f"{symbol}(") == 1
        expected_calls = 4 if "_topk_" in symbol else 2
        assert clean_mlir.count(f"func.call @{symbol}") == expected_calls

    sass = _assert_ltoir_inlined(
        tmp_path=tmp_path,
        expected_symbols=provider_symbols,
    )
    assert re.search(r"\bCALL(?:\.[A-Z0-9_]+)*\b", sass) is None
