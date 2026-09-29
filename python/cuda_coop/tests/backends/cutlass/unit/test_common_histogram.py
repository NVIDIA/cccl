# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Common histogram frontend checks for CUTLASS tracing."""

from importlib import import_module

import numpy as np
import pytest

from cuda import coop
from cuda.coop._core import this_block

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.unit]


class _ReadonlySamples:
    items_per_thread = 3
    dtype = np.uint8

    def __len__(self):
        return 3

    def __getitem__(self, index):
        return (1, 2, 1)[index]


def test_common_readonly_inputs_and_typed_controls(monkeypatch):
    api = import_module("cuda.coop._core.api.histogram")
    dispatch = import_module("cuda.coop._core.api._dispatch")
    samples = _ReadonlySamples()
    sentinel = object()
    monkeypatch.setattr(
        api, "_group_primitive_marker", lambda *a, **k: sentinel
    )
    with monkeypatch.context() as compiler_context:
        compiler_context.setattr(
            dispatch, "_backend_module_name", lambda: "test.backend"
        )
        compiler_context.setattr(
            import_module("cuda.coop._core.api.histogram"),
            "_backend_module_name",
            lambda: "test.backend",
        )
        assert (
            coop.histogram(
                this_block(), samples, bins=4, counter_dtype=np.int64
            )
            is sentinel
        )
        with pytest.raises(TypeError, match="ThreadData"):
            coop.histogram(this_block(), 1, bins=4)
    assert list(samples) == [1, 2, 1]
