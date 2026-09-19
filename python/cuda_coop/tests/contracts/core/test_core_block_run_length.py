# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Window sizing and complete-operation provider contracts."""

import pytest

from cuda.coop._core import INT32, UINT64, ArgumentBinding, CxxFunction, Value
from cuda.coop._core.block.run_length import make_block_run_length_decode_spec


def _spec(**kwargs):
    params = dict(
        item_dtype=INT32,
        run_length_dtype=INT32,
        block_dim=(32, 1, 1),
        runs_per_thread=2,
        decoded_items_per_thread=4,
    )
    params.update(kwargs)
    return make_block_run_length_decode_spec(**params)


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("runs_per_thread", 0),
        ("decoded_items_per_thread", 0),
        ("decoded_items_per_thread", True),
        ("decoded_items_per_thread", 1.5),
        ("decoded_items_per_thread", 2**31),
        ("block_dim", (32, 2, 1)),
    ],
)
def test_invalid_tile_shapes(parameter, value):
    with pytest.raises((ValueError, TypeError)):
        _spec(**{parameter: value})


@pytest.mark.parametrize("offset", [-1, True, 1.25, 2**64])
def test_invalid_static_offsets(offset):
    with pytest.raises((TypeError, ValueError), match="offset"):
        _spec(offset=ArgumentBinding.static(offset))


def test_wide_offsets_are_not_narrowed_in_provider_abi():
    static = _spec(offset=ArgumentBinding.static(2**32 + 7)).specialization
    assert static.parameters[0][-1] == CxxFunction(
        f"{2**32 + 7}ULL", UINT64, name="offset"
    )
    runtime = _spec(offset=ArgumentBinding.runtime()).specialization
    assert runtime.parameters[0][-1] == Value(UINT64, name="offset")
    assert runtime.template_arguments["ControlT"] == UINT64


def test_bulk_returns_total_and_keeps_output_pointers_explicit():
    spec = _spec(bulk=True, relative_offsets=True).specialization
    names = [parameter.name for parameter in spec.parameters[0]]
    assert names == [
        "temp_storage",
        "run_values",
        "run_lengths",
        "destination",
        "capacity",
        "relative_offsets",
        "relative_capacity",
        "offset",
        "total",
    ]
    assert spec.algorithm.output_by_reference


@pytest.mark.parametrize("bulk", [False, True])
def test_common_run_inputs_accept_readonly_payloads_and_reject_float_lengths(
    monkeypatch, bulk
):
    from importlib import import_module

    import numpy as np

    from cuda.coop._core import this_block
    from tests.contracts.core.test_core_group_load_store import _ReadonlyThreadData

    api = import_module("cuda.coop._core.api.run_length")
    dispatch = import_module("cuda.coop._core.api._dispatch")
    calls = []
    monkeypatch.setattr(
        api, "_group_primitive_marker", lambda *args, **kwargs: calls.append(args)
    )
    operation = api.run_length_decode_into if bulk else api.run_length_decode
    args = (_ReadonlyThreadData(dtype=np.int16), _ReadonlyThreadData(dtype=np.uint64))
    with dispatch._compiler_scope("test.backend"):
        operation(
            this_block(),
            *args,
            *([object()] if bulk else []),
            decoded_items_per_thread=4,
        )
        with pytest.raises(TypeError, match="run_lengths"):
            operation(
                this_block(),
                args[0],
                _ReadonlyThreadData(dtype=np.float32),
                *([object()] if bulk else []),
                decoded_items_per_thread=4,
            )
    assert len(calls) == 1
