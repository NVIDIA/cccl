# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import hashlib

import pytest

pytest.importorskip("cutlass")

from cuda.coop.cutlass._compiler import _bundle as _provider_bundle
from cuda.coop.cutlass._compiler import _bundle_contract as _provider_contract
from cuda.coop.cutlass._compiler import _cache as _provider_cache
from cuda.coop.cutlass._compiler import _nvrtc as _provider_nvrtc


def test_provider_source_dump_survives_compile_cache_hit(tmp_path, monkeypatch):
    source = 'extern "C" __device__ int cuda_coop_cutlass_probe() { return 1; }\n'
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    identity = _provider_contract.make_bundle_identity(
        source_hash=source_hash,
        bundle_format="ltoir",
        bundle_arch="compute_80",
        bundle_sm_arch="sm_80",
        compiler_options=_provider_contract.bundle_compiler_options(
            "ltoir",
            "compute_80",
        ),
        layout_expressions=(),
    )
    cache_identity = _provider_contract.make_bundle_cache_identity(
        identity,
        include_key=_provider_contract.include_dirs_cache_key([]),
        producer_compiler_version=(
            _provider_nvrtc.get_version()
            or _provider_bundle._UNKNOWN_COMPILER_PROCESS_TOKEN
        ),
    )
    cached_artifact = tmp_path / "cached.ltoir"
    cached_artifact.write_bytes(b"cached")
    dump_dir = tmp_path / "source"

    monkeypatch.setenv(
        "CUDA_COOP_CUTLASS_PROVIDER_DUMP_DIR",
        str(dump_dir),
    )
    _provider_bundle.reset_compile_state()
    _provider_cache._SOURCE_CACHE[cache_identity.cache_key] = (
        _provider_cache._CachedBundle(
            path=str(cached_artifact),
            layouts_by_expression={},
        )
    )

    result = _provider_bundle.compile_bundle_source(
        source,
        scope="test",
        provider_dir=str(tmp_path),
        registered_headers=dict,
        select_bundle_format=lambda: "ltoir",
        resolve_nvrtc_sm_arch=lambda: "sm_80",
        resolve_nvrtc_arch=lambda: "compute_80",
    )

    assert result == str(cached_artifact)
    dumped = dump_dir / f"cuda_coop_cutlass_bundle_{source_hash}.cpp"
    assert dumped.read_text(encoding="utf-8") == source
