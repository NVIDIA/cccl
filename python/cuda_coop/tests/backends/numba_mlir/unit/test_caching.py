# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from cuda.coop.numba_mlir import _caching


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
