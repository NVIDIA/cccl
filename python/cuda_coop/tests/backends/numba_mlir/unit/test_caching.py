# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import hashlib

import pytest

from cuda.coop.numba_mlir import _types
from cuda.coop.numba_mlir._compiler import _caching
from cuda.coop.numba_mlir._semantic import _numba_semantic_token

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]


@pytest.mark.parametrize("payload", ("[]", "3", "null"))
def test_cache_treats_non_object_json_as_a_miss(tmp_path, payload):
    path = tmp_path / "cache-entry.json"
    path.write_text(payload, encoding="utf-8")

    assert _caching._read_cache(path) is _caching._CACHE_MISS


def test_disk_cache_disables_itself_when_its_directory_is_unavailable(monkeypatch):
    calls = []

    def unavailable(_cache_identity):
        raise PermissionError("cache directory is read-only")

    monkeypatch.setattr(_caching, "_CACHE_USABLE", True)
    monkeypatch.setattr(_caching, "_cache_identity_path", unavailable)

    @_caching.disk_cache
    def compute(value):
        calls.append(value)
        return value + 1

    assert compute(3) == 4
    assert compute(3) == 4
    assert calls == [3, 3]
    assert not _caching._CACHE_USABLE


def test_disk_cache_bypasses_unsupported_key_values(monkeypatch, tmp_path):
    calls = []
    unsupported = object()

    monkeypatch.setattr(_caching, "_CACHE_USABLE", True)
    monkeypatch.setattr(
        _caching,
        "_cache_identity_path",
        lambda _cache_identity: str(tmp_path),
    )

    @_caching.disk_cache
    def compute(value):
        calls.append(value)
        return 4 if value is unsupported else value + 1

    assert compute(unsupported) == 4
    assert compute(unsupported) == 4
    assert compute(3) == 4
    assert compute(3) == 4
    assert calls == [unsupported, unsupported, 3]
    assert _caching._CACHE_USABLE


def test_disk_cache_bypasses_unsupported_result_values(monkeypatch, tmp_path):
    calls = []
    unsupported = object()

    monkeypatch.setattr(_caching, "_CACHE_USABLE", True)
    monkeypatch.setattr(
        _caching,
        "_cache_identity_path",
        lambda _cache_identity: str(tmp_path),
    )

    @_caching.disk_cache
    def compute(value):
        calls.append(value)
        return unsupported if value == "unsupported" else value + 1

    assert compute("unsupported") is unsupported
    assert compute("unsupported") is unsupported
    assert not tuple(tmp_path.glob(".*"))
    assert compute(3) == 4
    assert compute(3) == 4
    assert calls == ["unsupported", "unsupported", 3]
    assert _caching._CACHE_USABLE


def test_disk_cache_persists_byte_valued_compiler_options(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(_caching, "_CACHE_USABLE", True)
    monkeypatch.setattr(
        _caching,
        "_cache_identity_path",
        lambda _cache_identity: str(tmp_path),
    )

    @_caching.disk_cache
    def compute(options):
        calls.append(options)
        return b"ltoir"

    options = (b"--std=c++17", b"--gpu-architecture=compute_90")
    assert compute(options) == b"ltoir"
    assert compute(options) == b"ltoir"
    assert calls == [options]
    assert len(tuple(tmp_path.iterdir())) == 1


def test_symbol_hash_uses_numba_type_identity():
    assert _numba_semantic_token(_types.types.int32) == _numba_semantic_token(
        (
            "numba-cuda-mlir-type",
            type(_types.types.int32).__module__,
            type(_types.types.int32).__qualname__,
            "int32",
        )
    )
    int32 = hashlib.sha1()
    _types._hash_symbol_value(int32, _types.types.int32)
    another_int32 = hashlib.sha1()
    _types._hash_symbol_value(another_int32, _types.types.int32)
    float32 = hashlib.sha1()
    _types._hash_symbol_value(float32, _types.types.float32)

    assert int32.digest() == another_int32.digest()
    assert int32.digest() != float32.digest()


def test_numba_type_tokens_remain_distinct():
    values = (
        _types.types.int32,
        _types.types.float32,
        _types.types.float64,
        _types.types.uint32,
    )
    tokens = tuple(_numba_semantic_token(value) for value in values)

    assert len(set(tokens)) == len(values)
