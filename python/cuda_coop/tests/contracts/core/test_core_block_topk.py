# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check TopK count bindings and block limits before provider compilation.

Static counts must fit the tile. Runtime counts retain a signed int64 ABI,
and an omitted valid count becomes the full tile constant. Empty or short
valid prefixes remain legal even when k asks for more items.
"""

import pytest

from cuda.coop._core import INT32, INT64, ArgumentBinding, CxxFunction, Value
from cuda.coop._core.block.topk import make_block_topk_specialization


def _specialization(**overrides):
    args = {
        "key_dtype": INT32,
        "block_dim": (64, 1, 1),
        "items_per_thread": 2,
        "selection": "min",
        "k": ArgumentBinding.static(7),
    }
    args.update(overrides)
    return make_block_topk_specialization(**args)


@pytest.mark.parametrize("name", ["k", "num_valid"])
@pytest.mark.parametrize("count", [-1, 129, 2**32, True, 1.5])
def test_bad_static_counts(name, count):
    with pytest.raises((ValueError, TypeError), match=name):
        _specialization(**{name: ArgumentBinding.static(count)})


def test_counts_preserve_wide_runtime_abi_and_full_tile_constant():
    specialization = _specialization(k=ArgumentBinding.runtime()).specialization
    parameters = specialization.parameters[0]
    assert parameters[-2:] == (
        Value(INT64, name="k"),
        CxxFunction("128", INT64, name="num_valid"),
    )


@pytest.mark.parametrize("block", [(32, 2, 1), (32, 1, 2)])
def test_multidimensional_topk_is_rejected(block):
    with pytest.raises(ValueError, match="one-dimensional"):
        _specialization(block_dim=block)


@pytest.mark.parametrize("k,count", [(0, 0), (17, 0), (17, 7), (128, 128)])
def test_empty_and_short_prefixes_are_valid(k, count):
    _specialization(
        k=ArgumentBinding.static(k), num_valid=ArgumentBinding.static(count)
    )
