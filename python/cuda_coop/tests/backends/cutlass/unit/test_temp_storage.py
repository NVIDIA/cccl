# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from enum import Enum
from importlib import import_module

import pytest

pytest.importorskip("cutlass")
TempStorage = import_module("cuda.coop.cutlass._temp_storage").TempStorage

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.unit]


@pytest.mark.parametrize("sharing", ["shared", "exclusive"])
@pytest.mark.parametrize("auto_sync", [None, True, False])
@pytest.mark.parametrize("capacity", [None, 1024])
def test_storage_defaults(sharing, auto_sync, capacity):
    storage = TempStorage(capacity, sharing=sharing, auto_sync=auto_sync)
    assert storage.size_in_bytes == capacity
    assert storage.capacity_size_in_bytes == capacity
    assert storage.alignment is None
    assert storage.sharing == sharing
    assert storage.auto_sync is (True if auto_sync is None else auto_sync)
    assert storage.is_deferred


def test_storage_alignment_and_sharing():
    class Alignment:
        def __index__(self):
            return 64

    storage = TempStorage(1024, alignment=Alignment(), sharing=" EXCLUSIVE ")
    assert storage.alignment == 64
    assert storage.sharing == "exclusive"
    assert storage.auto_sync is True


@pytest.mark.parametrize("capacity", [True, False, 1.5, "128"])
def test_capacity_type_errors(capacity):
    with pytest.raises(TypeError, match="size_in_bytes"):
        TempStorage(capacity)


@pytest.mark.parametrize("capacity", [0, -1])
def test_capacity_range_errors(capacity):
    with pytest.raises(ValueError, match="positive"):
        TempStorage(capacity)


@pytest.mark.parametrize("alignment", [True, 0, -1, 3, 1.5])
def test_alignment_errors(alignment):
    with pytest.raises((TypeError, ValueError), match="alignment"):
        TempStorage(alignment=alignment)


def test_storage_option_errors():
    class Sharing(str, Enum):
        SHARED = "shared"

    for sharing in (None, 1, Sharing.SHARED):
        with pytest.raises(TypeError, match="sharing"):
            TempStorage(sharing=sharing)
    with pytest.raises(ValueError, match="sharing"):
        TempStorage(sharing="private")
    for auto_sync in (0, 1, "true"):
        with pytest.raises(TypeError, match="auto_sync"):
            TempStorage(auto_sync=auto_sync)
    with pytest.raises(TypeError):
        TempStorage(1024, 16)


def test_manual_sync_calls_block_barrier(monkeypatch):
    arch = import_module("cutlass.cute.arch")
    calls = []
    monkeypatch.setattr(arch, "sync_threads", lambda: calls.append("block"))
    assert TempStorage(auto_sync=False).sync() is None
    assert calls == ["block"]


def test_loop_protocol_keeps_identity():
    typing = import_module("cutlass.base_dsl.typing")
    storage = TempStorage(1024, sharing="exclusive", alignment=64)
    assert typing.implements_dynamic_expression(storage)
    assert storage.__extract_mlir_values__() == []
    assert storage.__new_from_mlir_values__([]) is storage
    with pytest.raises(ValueError, match="no runtime MLIR values"):
        storage.__new_from_mlir_values__([object()])
