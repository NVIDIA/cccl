# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Final-link evidence for common keys-only CUTLASS TopK."""

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
    "cuda_coop_cutlass_topk_max_keys_i32_bt64_x2",
    "cuda_coop_cutlass_topk_min_keys_i32_bt64_x2",
)


@pytest.mark.evidence_for("group.topk_max_keys", backend="cutlass", evidence="link")
@pytest.mark.evidence_for("group.topk_min_keys", backend="cutlass", evidence="link")
def test_common_and_qualified_topk_wrappers_are_eliminated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_dump_environment(monkeypatch, tmp_path)
    _require_runtime()

    _run_example_subprocess("_common_root_topk_codegen_probe")

    provider_source = _read_one(
        "bundle/cuda_coop_cutlass_bundle_*.cpp",
        tmp_path=tmp_path,
    )
    provider_symbols = tuple(
        sorted(
            set(
                re.findall(
                    r"\b(cuda_coop_cutlass_topk_[A-Za-z0-9_]+)\(",
                    provider_source,
                )
            )
        )
    )
    assert provider_symbols == _EXPECTED_SYMBOLS
    assert "static_assert(planned_temp_bytes >= required_temp_bytes" in provider_source

    clean_mlir = _read_one("dsl/*_clean.mlir", tmp_path=tmp_path)
    for symbol in _EXPECTED_SYMBOLS:
        assert provider_source.count(f"{symbol}(") == 1
        assert clean_mlir.count(f"func.call @{symbol}") == 4

    sass = _assert_ltoir_inlined(
        tmp_path=tmp_path,
        expected_symbols=_EXPECTED_SYMBOLS,
    )
    assert re.search(r"\bCALL(?:\.[A-Z0-9_]+)*\b", sass) is None
