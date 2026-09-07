# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Final-link evidence for common CUTLASS Run-Length Decode."""

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

_EXPECTED_SYMBOL = "cuda_coop_cutlass_cub_run_length_decode_b8x4x2_vu64_lu64_r2_x3"
_EXPECTED_OFFSETS_SYMBOL = f"{_EXPECTED_SYMBOL}_offsets"


@pytest.mark.evidence_for(
    "group.run_length_decode",
    backend="cutlass",
    evidence="link",
)
def test_common_and_qualified_decode_wrapper_is_eliminated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_dump_environment(monkeypatch, tmp_path)
    _require_runtime()

    _run_example_subprocess("_common_root_run_length_decode_codegen_probe")

    provider_source = _read_one(
        "bundle/cuda_coop_cutlass_bundle_*.cpp",
        tmp_path=tmp_path,
    )
    provider_symbols = tuple(
        sorted(
            set(
                re.findall(
                    r"\b(cuda_coop_cutlass_cub_run_length_decode_[A-Za-z0-9_]+)\(",
                    provider_source,
                )
            )
        )
    )
    assert provider_symbols == (_EXPECTED_SYMBOL, _EXPECTED_OFFSETS_SYMBOL)
    assert "#include <cuda/std/type_traits>" in provider_source
    assert provider_source.count(f"{_EXPECTED_SYMBOL}(") == 1
    assert provider_source.count(f"{_EXPECTED_OFFSETS_SYMBOL}(") == 1
    assert "unsigned long long decoded_window_offset" in provider_source
    assert "decoded_window_offset < 0" not in provider_source
    assert "decoded_offset < decoded_total" in provider_source
    assert "decoded_total - decoded_offset : 0ull" in provider_source
    assert "local_target_0 < decoded_remaining" in provider_source
    assert "first_target" not in provider_source
    assert "static_cast<unsigned long long>(~0ull)" in provider_source

    clean_mlir = _read_one("dsl/*_clean.mlir", tmp_path=tmp_path)
    assert (
        len(re.findall(rf"func\.call @{re.escape(_EXPECTED_SYMBOL)}\(", clean_mlir))
        == 5
    )
    assert (
        len(
            re.findall(
                rf"func\.call @{re.escape(_EXPECTED_OFFSETS_SYMBOL)}\(", clean_mlir
            )
        )
        == 1
    )

    sass = _assert_ltoir_inlined(
        tmp_path=tmp_path,
        expected_symbols=(_EXPECTED_SYMBOL, _EXPECTED_OFFSETS_SYMBOL),
    )
    assert re.search(r"\b(?:CALL|LDL|STL)(?:\.[A-Z0-9_]+)*\b", sass) is None
