# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Final-link evidence for common-root CUTLASS Load and Store."""

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


@pytest.mark.evidence_for("group.load", backend="cutlass", evidence="link")
@pytest.mark.evidence_for("group.store", backend="cutlass", evidence="link")
def test_common_root_and_scoped_load_store_share_final_cubin_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_dump_environment(monkeypatch, tmp_path)
    _require_runtime()

    _run_example_subprocess(
        "cutlass_scoped_load_store_codegen_probe",
        module_prefix="tests.support.fixtures",
    )

    provider_source = _read_one(
        "bundle/cuda_coop_cutlass_bundle_*.cpp",
        tmp_path=tmp_path,
    )
    symbols = tuple(
        sorted(
            set(
                re.findall(
                    r"void (cuda_coop_cutlass_cub_(?:load|store)_"
                    r"(?:block|warp)_[A-Za-z0-9_]+)\(",
                    provider_source,
                )
            )
        )
    )
    assert len(symbols) == 4
    assert sum("_load_block_" in symbol for symbol in symbols) == 1
    assert sum("_store_block_" in symbol for symbol in symbols) == 1
    assert sum("_load_warp_" in symbol for symbol in symbols) == 1
    assert sum("_store_warp_" in symbol for symbol in symbols) == 1

    clean_mlir = _read_one("dsl/*_clean.mlir", tmp_path=tmp_path)
    for symbol in symbols:
        assert provider_source.count(f"void {symbol}(") == 1
        assert clean_mlir.count(f"func.call @{symbol}") == 2

    sass = _assert_ltoir_inlined(
        tmp_path=tmp_path,
        expected_symbols=symbols,
    )
    assert re.search(r"\b(?:CALL|LDL|STL)(?:\.[A-Z0-9_]+)*\b", sass) is None
