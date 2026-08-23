# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402


import pytest

from cuda.coop._core import ArgumentKind, ParameterRole, RuntimeValue
from cuda.coop._core.block import (
    BLOCK_TOPK_TYPE,
    ArgumentBinding,
    TopKPayload,
    TopKTilePolicy,
    binding,
    make_block_topk_spec,
)


@pytest.mark.parametrize("selection", ["max", "min"])
@pytest.mark.parametrize("with_values", [False, True])
@pytest.mark.parametrize("full_tile", [False, True])
def test_block_topk_selects_named_method(selection, with_values, full_tile):
    spec = make_block_topk_spec(
        key_dtype="int",
        value_dtype="float" if with_values else None,
        block_dim=(64, 1, 1),
        items_per_thread=2,
        selection=selection,
        num_valid=(
            ArgumentBinding.omitted() if full_tile else ArgumentBinding.runtime()
        ),
    )

    payload = "pairs" if with_values else "keys"
    suffix = "full" if full_tile else "partial"
    assert spec.method_name == f"{selection}_{payload}_{suffix}"
    assert spec.payload is (TopKPayload.PAIRS if with_values else TopKPayload.KEYS)
    assert spec.tile_policy is (
        TopKTilePolicy.FULL if full_tile else TopKTilePolicy.PARTIAL
    )
    assert spec.tile_size == 128
    assert spec.specialization.struct_name == "BlockTopKCoop"


def test_block_topk_classifies_static_and_runtime_arguments():
    spec = make_block_topk_spec(
        key_dtype="int",
        block_dim=(32, 1, 1),
        items_per_thread=4,
        selection="max",
        num_valid=ArgumentBinding.static(99),
        begin_bit=ArgumentBinding.runtime(),
        end_bit=ArgumentBinding.static(16),
    )

    classifications = spec.specialization.classify_method()
    assert [(item.name, item.kind, item.role) for item in classifications] == [
        ("temp_storage", ArgumentKind.RUNTIME, ParameterRole.TEMP_STORAGE),
        ("keys", ArgumentKind.RUNTIME, ParameterRole.INOUT),
        ("k", ArgumentKind.RUNTIME, ParameterRole.INPUT),
        ("num_valid", ArgumentKind.STATIC, ParameterRole.CONSTANT),
        ("begin_bit", ArgumentKind.RUNTIME, ParameterRole.INPUT),
        ("end_bit", ArgumentKind.STATIC, ParameterRole.CONSTANT),
    ]


def test_runtime_values_do_not_enter_block_topk_semantic_identity():
    first = make_block_topk_spec(
        key_dtype="int",
        block_dim=(32, 1, 1),
        items_per_thread=1,
        selection="min",
        num_valid=binding(RuntimeValue("first")),
    )
    second = make_block_topk_spec(
        key_dtype="int",
        block_dim=(32, 1, 1),
        items_per_thread=1,
        selection="min",
        num_valid=binding(RuntimeValue("second")),
    )
    static = make_block_topk_spec(
        key_dtype="int",
        block_dim=(32, 1, 1),
        items_per_thread=1,
        selection="min",
        num_valid=ArgumentBinding.static(17),
    )

    assert first.semantic_key == second.semantic_key
    assert first.semantic_key != static.semantic_key


def test_block_topk_alias_has_all_named_forwarders():
    for selection in ("max", "min"):
        for payload in ("keys", "pairs"):
            for suffix in ("full", "partial"):
                assert f"void {selection}_{payload}_{suffix}(" in BLOCK_TOPK_TYPE.code


def test_block_topk_rejects_incomplete_bit_range():
    with pytest.raises(ValueError, match="provided together"):
        make_block_topk_spec(
            key_dtype="int",
            block_dim=(32, 1, 1),
            items_per_thread=1,
            selection="max",
            begin_bit=ArgumentBinding.static(0),
        )
