# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Capture an existing JIT result and distinguish strict and fallback misses."""

from pathlib import Path

import pytest

from cuda.coop.cutlass import aot
from cuda.coop.cutlass._compiler import _bundle, _cache, _nvrtc

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.compile]


def test_capture_cached_provider_and_handle_changed_specialization(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(_cache.CACHE_DIR_ENV, str(tmp_path / "cache"))
    monkeypatch.setattr(_cache, "_SOURCE_CACHE", {})
    pack = tmp_path / "pack"

    def compile_provider(value):
        source = f'extern "C" __device__ int value() {{ return {value}; }}'
        return Path(
            _bundle.compile_bundle_source(
                source, arch="compute_90", required_headers=()
            )
        )

    original = compile_provider(1).read_bytes()
    _cache._SOURCE_CACHE.clear()
    with monkeypatch.context() as cached:
        cached.setattr(
            _nvrtc,
            "compile_ltoir",
            lambda *args: pytest.fail("JIT cache missed"),
        )
        with aot.capture(pack):
            assert compile_provider(1).read_bytes() == original

    with aot.use(pack, mode="auto"):
        changed = compile_provider(2).read_bytes()
    assert changed != original

    def forbid_jit(*args):
        pytest.fail("Required pack consulted the provider JIT")

    monkeypatch.setattr(_nvrtc, "resolve_compile_context", forbid_jit)
    with aot.use(pack, mode="required"):
        assert compile_provider(1).read_bytes() == original
        # Even though the JIT cache now contains value(2), the pack does not.
        with pytest.raises(aot.PackMissError):
            compile_provider(2)
