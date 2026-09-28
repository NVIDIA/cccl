# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Common run length frontend checks for CUTLASS tracing."""

from importlib import import_module

import numpy as np
import pytest

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.unit]


class _ReadonlyThreadData:
    def __init__(self, items_per_thread=2, *, dtype=np.int32):
        self.items_per_thread = items_per_thread
        self.dtype = dtype
        self._items = [0] * items_per_thread

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]


@pytest.mark.parametrize("bulk", [False, True])
def test_common_run_inputs_accept_readonly_payloads_and_reject_float_lengths(
    monkeypatch, bulk
):
    import numpy as np

    from cuda import coop
    from cuda.coop._core import this_block

    api = import_module("cuda.coop._core.api.run_length")
    dispatch = import_module("cuda.coop._core.api._dispatch")
    calls = []
    monkeypatch.setattr(
        api, "_group_primitive_marker", lambda *args, **kwargs: calls.append(args)
    )
    operation = coop.run_length_decode_into if bulk else coop.run_length_decode
    args = (_ReadonlyThreadData(dtype=np.int16), _ReadonlyThreadData(dtype=np.uint64))
    with monkeypatch.context() as compiler_context:
        compiler_context.setattr(
            dispatch, "_backend_module_name", lambda: "test.backend"
        )
        compiler_context.setattr(
            import_module("cuda.coop._core.api.run_length"),
            "_backend_module_name",
            lambda: "test.backend",
        )
        operation(
            this_block(),
            *args,
            *([object()] if bulk else []),
            decoded_items_per_thread=4,
        )
        with pytest.raises(
            TypeError,
            match=f"cuda\\.coop\\.{operation.__name__} supports run_lengths dtypes",
        ):
            operation(
                this_block(),
                args[0],
                _ReadonlyThreadData(dtype=np.float32),
                *([object()] if bulk else []),
                decoded_items_per_thread=4,
            )
    assert len(calls) == 1
