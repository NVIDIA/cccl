# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable TempStorage alignment validation and backend forwarding."""

from importlib import import_module

import numpy as np
import pytest

api = import_module("cuda.coop._core.api.temp_storage")


@pytest.mark.parametrize("alignment", [None, 1, 2, 4, 8, 16, np.int64(32)])
def test_temp_storage_forwards_minimum_alignment(monkeypatch, alignment):
    calls = []
    payload = object()

    def constructor(size_in_bytes, *, alignment, auto_sync, sharing):
        calls.append((size_in_bytes, alignment, auto_sync, sharing))
        return payload

    monkeypatch.setattr(api, "_backend_member", lambda name: constructor)
    assert api.TempStorage(64, alignment=alignment) is payload
    assert calls == [(64, alignment, None, "shared")]
    assert calls[0][1] is None or type(calls[0][1]) is int


@pytest.mark.parametrize(
    ("alignment", "error", "message"),
    [
        (True, TypeError, "alignment must be an integer or None"),
        (1.5, TypeError, "alignment must be an integer or None"),
        (0, ValueError, "alignment must be a positive integer"),
        (-1, ValueError, "alignment must be a positive integer"),
        (3, ValueError, "alignment must be a power of 2"),
    ],
)
def test_temp_storage_rejects_invalid_alignment_before_dispatch(
    monkeypatch, alignment, error, message
):
    def unexpected_dispatch(name):
        pytest.fail("invalid alignment reached backend dispatch")

    monkeypatch.setattr(api, "_backend_member", unexpected_dispatch)
    with pytest.raises(error, match=message):
        api.TempStorage(64, alignment=alignment)
