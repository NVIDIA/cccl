# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import hashlib

import pytest

pytest.importorskip("cutlass")

from cuda.coop.cutlass._dsl import _provider_bundle  # noqa: E402


def test_provider_source_dump_survives_compile_cache_hit(tmp_path, monkeypatch):
    source = 'extern "C" __device__ int cuda_coop_cutlass_probe() { return 1; }\n'
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    identity = _provider_bundle.make_bundle_identity(
        source_hash=source_hash,
        bundle_format="ltoir",
        bundle_arch="compute_120",
        bundle_sm_arch="sm_120",
        compiler_options=_provider_bundle.bundle_compiler_options(
            "ltoir",
            "compute_120",
        ),
        layout_expressions=(),
    )
    cache_identity = _provider_bundle.make_bundle_cache_identity(
        identity,
        include_key=_provider_bundle.include_dirs_cache_key([]),
        producer_compiler_version=(
            _provider_bundle.get_nvrtc_version()
            or _provider_bundle._UNKNOWN_COMPILER_PROCESS_TOKEN
        ),
    )
    cached_artifact = tmp_path / "cached.ltoir"
    artifact_blob = b"cached"
    cached_artifact.write_bytes(artifact_blob)
    dump_dir = tmp_path / "source"

    monkeypatch.setenv(
        "CUDA_COOP_CUTLASS_PROVIDER_DUMP_DIR",
        str(dump_dir),
    )
    monkeypatch.setenv(
        _provider_bundle.CACHE_DIR_ENV,
        str(tmp_path / "provider-cache"),
    )
    _provider_bundle.reset_compile_state()
    cached = _provider_bundle._CachedBundle(
        path=str(cached_artifact),
        layouts_by_expression={},
    )
    _provider_bundle._write_bundle_metadata(
        str(cached_artifact),
        artifact_blob,
        cached,
        cache_identity,
        scope="test",
    )
    _provider_bundle._SOURCE_CACHE[cache_identity.cache_key] = cached

    result = _provider_bundle.compile_bundle_source(
        source,
        scope="test",
        provider_dir=str(tmp_path),
        registered_headers=lambda: {},
        select_bundle_format=lambda: "ltoir",
        resolve_nvrtc_sm_arch=lambda: "sm_120",
        resolve_nvrtc_arch=lambda: "compute_120",
    )

    assert result == str(cached_artifact)
    dumped = dump_dir / f"cuda_coop_cutlass_bundle_{source_hash}.cpp"
    assert dumped.read_text(encoding="utf-8") == source
