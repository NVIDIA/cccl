# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from importlib import import_module

import pytest

from cuda.coop._core import this_block

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.unit]


@pytest.mark.parametrize(
    "operation,options",
    [
        ("radix_sort_keys", {"descending": 1}),
        ("radix_sort_keys", {"begin_bit": -1}),
        ("radix_sort_keys", {"end_bit": 33}),
        ("radix_rank_keys", {"radix_bits": 9}),
        ("radix_rank_keys", {"begin_bit": 31}),
        ("radix_rank_keys", {"end_bit": 6, "radix_bits": 4}),
    ],
)
def test_common_frontend_rejects_invalid_controls_before_dispatch(
    monkeypatch, operation, options
):
    from cuda.coop._core.api import _dispatch
    from cuda.coop._core.api import radix_sort as radix

    class Payload:
        items_per_thread = 2
        dtype = int

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return index

    with monkeypatch.context() as compiler_context:
        compiler_context.setattr(
            _dispatch, "_backend_module_name", lambda: "test.backend"
        )
        compiler_context.setattr(
            import_module("cuda.coop._core.api.radix_sort"),
            "_backend_module_name",
            lambda: "test.backend",
        )
        with pytest.raises((TypeError, ValueError)):
            getattr(radix, operation)(this_block(), Payload(), **options)
