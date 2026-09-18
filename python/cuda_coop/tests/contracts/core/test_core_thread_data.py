# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable ThreadData alignment validation and backend forwarding."""

from importlib import import_module

import numpy as np
import pytest

api = import_module("cuda.coop._core.api.thread_data")


@pytest.mark.parametrize("alignment", [None, 1, 2, 4, 8, 16, np.int64(32)])
def test_thread_data_forwards_minimum_alignment(monkeypatch, alignment):
    calls = []
    payload = object()

    def constructor(items_per_thread, *, dtype, alignment):
        calls.append((items_per_thread, dtype, alignment))
        return payload

    monkeypatch.setattr(api, "_backend_member", lambda name: constructor)
    assert api.ThreadData(4, np.float32, alignment=alignment) is payload
    assert calls == [(4, np.float32, alignment)]
    assert calls[0][2] is None or type(calls[0][2]) is int


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
def test_thread_data_rejects_invalid_alignment_before_dispatch(
    monkeypatch, alignment, error, message
):
    def unexpected_dispatch(name):
        pytest.fail("invalid alignment reached backend dispatch")

    monkeypatch.setattr(api, "_backend_member", unexpected_dispatch)
    with pytest.raises(error, match=message):
        api.ThreadData(4, np.float32, alignment=alignment)
