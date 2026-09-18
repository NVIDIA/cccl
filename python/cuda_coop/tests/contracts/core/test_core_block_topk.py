# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-independent TopK count and dimensional contracts."""

import pytest

from cuda.coop._core import INT32, INT64, ArgumentBinding, CxxFunction, Value
from cuda.coop._core.block.topk import make_block_topk_spec


def _spec(**overrides):
    args = dict(
        key_dtype=INT32,
        block_dim=(64, 1, 1),
        items_per_thread=2,
        selection="min",
        k=ArgumentBinding.static(7),
    )
    args.update(overrides)
    return make_block_topk_spec(**args)


@pytest.mark.parametrize("name", ["k", "num_valid"])
@pytest.mark.parametrize("count", [-1, 129, 2**32, True, 1.5])
def test_bad_static_counts(name, count):
    with pytest.raises((ValueError, TypeError), match=name):
        _spec(**{name: ArgumentBinding.static(count)})


def test_counts_preserve_wide_runtime_abi_and_full_tile_constant():
    spec = _spec(k=ArgumentBinding.runtime()).specialization
    parameters = spec.parameters[0]
    assert parameters[-2:] == (
        Value(INT64, name="k"),
        CxxFunction("128", INT64, name="num_valid"),
    )


@pytest.mark.parametrize("block", [(32, 2, 1), (32, 1, 2)])
def test_multidimensional_topk_is_rejected(block):
    with pytest.raises(ValueError, match="one-dimensional"):
        _spec(block_dim=block)


@pytest.mark.parametrize("k,count", [(0, 0), (17, 0), (17, 7), (128, 128)])
def test_empty_and_short_prefixes_are_valid(k, count):
    _spec(k=ArgumentBinding.static(k), num_valid=ArgumentBinding.static(count))
