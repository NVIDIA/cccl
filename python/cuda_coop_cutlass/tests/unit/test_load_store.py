# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check Load result ownership and argument choices before compilation."""

from copy import copy
from importlib import import_module

import numpy as np
import pytest

import cuda.coop.cutlass as coop

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.unit]


@pytest.mark.parametrize("items_per_thread", (1, 4))
@pytest.mark.parametrize("alignment", (None, np.int64(1), 128))
def test_load_result_ownership(monkeypatch, items_per_thread, alignment):
    """Fresh results preserve dtype, alignment, and independent ownership.

    The provider spy isolates facade behavior from memory access and code
    generation, which are covered by the runtime movement tests.
    """

    movement = import_module("cuda.coop.cutlass._group_load_store")
    lowering = import_module("cuda.coop.cutlass._lowering._load_store")
    calls = []

    def provider_load(**kwargs):
        output = kwargs["output"]
        calls.append(output)
        output.dtype = np.int32
        for item, value in enumerate(kwargs["source"]):
            output[item] = np.int32(value)

    monkeypatch.setattr(
        movement,
        "_resolve_group",
        lambda group, algorithm, temp_storage, operation: (
            group,
            None,
            algorithm,
        ),
    )
    monkeypatch.setattr(lowering, "provider_load", provider_load)
    source = list(range(items_per_thread))
    group = coop.this_block()
    original = coop.load(
        group,
        source,
        items_per_thread=items_per_thread,
        alignment=alignment,
    )
    assert isinstance(original, coop.ThreadData)
    assert original.items_per_thread == items_per_thread
    assert original.dtype is np.int32
    assert original.alignment == alignment
    assert copy(original).alignment == alignment
    alias = original
    updated = [value + 10 for value in source]
    original = coop.load(group, updated, items_per_thread=items_per_thread)
    assert original is not alias
    assert list(alias) == source
    assert list(original) == updated

    assert calls == [alias, original]


def test_load_requires_explicit_extent_and_rejects_output():
    group = coop.this_block()
    with pytest.raises(TypeError, match="items_per_thread"):
        coop.load(group, object())
    with pytest.raises(TypeError):
        coop.load(group, object(), coop.ThreadData(1), items_per_thread=1)
    for count, error in (
        (True, TypeError),
        (1.5, TypeError),
        (0, ValueError),
        (-1, ValueError),
    ):
        with pytest.raises(error, match="items_per_thread"):
            coop.load(group, object(), items_per_thread=count)


@pytest.mark.parametrize(
    "alignment,error",
    (
        (True, TypeError),
        (1.5, TypeError),
        (0, ValueError),
        (-8, ValueError),
        (3, ValueError),
    ),
)
def test_load_alignment_invalid(alignment, error):
    with pytest.raises(error, match="alignment"):
        coop.load(
            coop.this_block(), object(), items_per_thread=1, alignment=alignment
        )
