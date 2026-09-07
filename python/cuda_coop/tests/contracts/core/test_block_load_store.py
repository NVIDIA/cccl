# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import numpy as np
import pytest

from cuda.coop._core import (
    ArgumentBinding,
    GroupLoadStoreAlgorithm,
    GroupLoadStoreKind,
    GroupLoadStoreSemantics,
    GroupLoweringTarget,
    LaunchFactOrigin,
    LaunchFacts,
    StorageOwnership,
    SynchronizationScope,
    make_group_primitive_call,
    plan_group_primitive,
    this_block,
)
from cuda.coop._core._symbols import semantic_token


def _plan(operation: GroupLoadStoreSemantics, block_dim=(8, 4, 2)):
    call = make_group_primitive_call(this_block(), operation)
    return plan_group_primitive(
        call,
        LaunchFacts(
            exact_block_dim=block_dim,
            provenance=LaunchFactOrigin(
                fact="exact_block_dim",
                source="test",
                verified=True,
            ),
        ),
    )


def test_full_tile_load_has_direct_cub_block_plan() -> None:
    plan = _plan(
        GroupLoadStoreSemantics(
            kind="load",
            dtype="int32",
            items_per_thread=2,
        )
    ).require_supported()

    assert plan.target is GroupLoweringTarget.CUB_BLOCK
    assert plan.resolved_group.static_size == 64
    assert plan.participation is not None
    assert plan.participation.exact_block_dim == (8, 4, 2)
    assert plan.participation.uniform_arguments == ()
    assert plan.implementation is not None
    assert plan.implementation.kind.value == "load"
    assert plan.implementation.block_dim == (8, 4, 2)
    assert plan.result is not None
    assert plan.result.result_items_per_thread == 2
    assert plan.synchronization is not None
    assert plan.synchronization.storage_reuse_barrier is SynchronizationScope.BLOCK
    assert plan.temp_storage is not None
    assert plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert plan.provenance is not None
    assert plan.provenance.header == "cub/block/block_load.cuh"
    assert plan.artifact_key is not None


def test_partial_load_tracks_valid_default_and_offset_bindings() -> None:
    operation = GroupLoadStoreSemantics(
        kind=GroupLoadStoreKind.LOAD,
        dtype="float32",
        items_per_thread=3,
        valid_items=ArgumentBinding.runtime(),
        oob_default=ArgumentBinding.static(-1.0),
        offset=ArgumentBinding.static(7),
    )
    plan = _plan(operation).require_supported()

    assert operation.algorithm is GroupLoadStoreAlgorithm.DIRECT
    assert operation.has_valid_items
    assert operation.has_oob_default
    assert operation.has_offset
    assert plan.participation is not None
    assert plan.participation.uniform_arguments == (
        "valid_items",
        "oob_default",
        "offset",
    )
    assert plan.participation.valid_member_selection == (
        "first valid_items tile elements"
    )
    assert plan.implementation is not None
    assert plan.implementation.valid_items
    assert plan.implementation.oob_default
    assert plan.implementation.pointer_offset


def test_partial_store_has_no_result_and_uses_internal_storage() -> None:
    plan = _plan(
        GroupLoadStoreSemantics(
            kind="store",
            dtype="uint32",
            items_per_thread=4,
            valid_items=ArgumentBinding.runtime(),
            offset=ArgumentBinding.runtime(),
        )
    ).require_supported()

    assert plan.result is None
    assert plan.implementation is not None
    assert plan.implementation.kind.value == "store"
    assert plan.temp_storage is not None
    assert plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert plan.temp_storage.address_space is None
    assert plan.provenance is not None
    assert plan.provenance.header == "cub/block/block_store.cuh"


def test_missing_exact_dimensions_is_typed_unsupported_plan() -> None:
    operation = GroupLoadStoreSemantics(
        kind="load",
        dtype="int32",
        items_per_thread=1,
    )
    call = make_group_primitive_call(this_block(), operation)
    plan = plan_group_primitive(call, LaunchFacts())

    assert plan.target is GroupLoweringTarget.UNSUPPORTED
    assert plan.artifact_key == (
        "unsupported",
        "unsupported",
        "missing_exact_block_dim",
    )
    with pytest.raises(NotImplementedError, match="exact block dimensions"):
        plan.require_supported()


def test_unverified_exact_dimensions_are_not_compiler_facts() -> None:
    operation = GroupLoadStoreSemantics(
        kind="load",
        dtype="int32",
        items_per_thread=1,
    )
    call = make_group_primitive_call(this_block(), operation)
    plan = plan_group_primitive(call, LaunchFacts(exact_block_dim=64))

    assert plan.target is GroupLoweringTarget.UNSUPPORTED
    assert plan.unsupported is not None
    assert plan.unsupported.code.value == "unverified_exact_block_dim"
    with pytest.raises(NotImplementedError, match="compiler-verified"):
        plan.require_supported()


def test_unsupported_plan_identity_preserves_the_reason_code() -> None:
    operation = GroupLoadStoreSemantics(
        kind="load",
        dtype="int32",
        items_per_thread=1,
    )
    call = make_group_primitive_call(this_block(), operation)
    missing = plan_group_primitive(call, LaunchFacts())
    unverified = plan_group_primitive(call, LaunchFacts(exact_block_dim=64))

    assert missing.semantic_key == unverified.semantic_key
    assert missing.artifact_key != unverified.artifact_key
    assert missing != unverified


def test_artifact_identity_tracks_partial_tile_controls() -> None:
    full = _plan(
        GroupLoadStoreSemantics(
            kind="load",
            dtype="int32",
            items_per_thread=2,
        )
    )
    partial = _plan(
        GroupLoadStoreSemantics(
            kind="load",
            dtype="int32",
            items_per_thread=2,
            valid_items=ArgumentBinding.runtime(),
        )
    )

    assert full.artifact_key == _plan(full.call.operation).artifact_key
    assert full.artifact_key != partial.artifact_key


@pytest.mark.parametrize(
    ("left", "right"),
    (
        (True, 1),
        (0.0, -0.0),
        ([1, 2], [1, 3]),
        ({"value": [1]}, {"value": [2]}),
    ),
)
def test_binding_identity_preserves_type_representation_and_unhashable_values(
    left,
    right,
) -> None:
    left_binding = ArgumentBinding.static(left)
    same_binding = ArgumentBinding.static(left)
    right_binding = ArgumentBinding.static(right)

    assert left_binding == same_binding
    assert hash(left_binding) == hash(same_binding)
    assert left_binding != right_binding
    assert left_binding.semantic_key != right_binding.semantic_key


def test_semantics_with_unhashable_static_binding_remain_hashable() -> None:
    semantics = GroupLoadStoreSemantics(
        kind="load",
        dtype="int32",
        items_per_thread=1,
        valid_items=ArgumentBinding.runtime(),
        oob_default=ArgumentBinding.static([1]),
    )

    assert isinstance(hash(semantics), int)


def test_semantic_tokens_sort_heterogeneous_mapping_keys_stably() -> None:
    left = {1: "integer", "1": "string"}
    right = {"1": "string", 1: "integer"}

    assert semantic_token(left) == semantic_token(right)


@pytest.mark.parametrize("items_per_thread", (0, -1, True, "two"))
def test_semantics_reject_invalid_item_counts(items_per_thread) -> None:
    with pytest.raises(ValueError, match="items_per_thread must be a positive"):
        GroupLoadStoreSemantics(
            kind="load",
            dtype="int32",
            items_per_thread=items_per_thread,
        )


def test_semantics_reject_invalid_oob_contracts() -> None:
    with pytest.raises(ValueError, match="valid only for group load"):
        GroupLoadStoreSemantics(
            kind="store",
            dtype="int32",
            items_per_thread=1,
            valid_items=ArgumentBinding.runtime(),
            oob_default=ArgumentBinding.static(-1),
        )
    with pytest.raises(ValueError, match="requires valid_items"):
        GroupLoadStoreSemantics(
            kind="load",
            dtype="int32",
            items_per_thread=1,
            oob_default=ArgumentBinding.static(-1),
        )


def test_only_direct_algorithm_is_accepted() -> None:
    with pytest.raises(ValueError):
        GroupLoadStoreSemantics(
            kind="load",
            dtype="int32",
            items_per_thread=1,
            algorithm="striped",
        )


@pytest.mark.parametrize("valid_items", (True, 65, -1, 1.5))
def test_static_valid_items_are_checked_against_tile(valid_items) -> None:
    with pytest.raises((TypeError, ValueError), match="valid_items"):
        operation = GroupLoadStoreSemantics(
            kind="load",
            dtype="int32",
            items_per_thread=1,
            valid_items=ArgumentBinding.static(valid_items),
        )
        _plan(operation, block_dim=64)


@pytest.mark.parametrize("offset", (True, 1.5, "one"))
def test_static_offset_must_be_an_integer(offset) -> None:
    with pytest.raises(TypeError, match="static offset must be an integer"):
        operation = GroupLoadStoreSemantics(
            kind="store",
            dtype="int32",
            items_per_thread=1,
            offset=ArgumentBinding.static(offset),
        )
        _plan(operation)


def test_static_offset_must_be_non_negative() -> None:
    operation = GroupLoadStoreSemantics(
        kind="store",
        dtype="int32",
        items_per_thread=1,
        offset=ArgumentBinding.static(-1),
    )
    with pytest.raises(ValueError, match="offset must be at least 0"):
        _plan(operation)


def test_static_valid_items_must_fit_signed_i32() -> None:
    with pytest.raises(ValueError, match="fit a signed 32-bit integer"):
        operation = GroupLoadStoreSemantics(
            kind="load",
            dtype="int32",
            items_per_thread=1,
            valid_items=ArgumentBinding.static(1 << 31),
        )
        _plan(operation, block_dim=1)


def test_static_offset_accepts_signed_i64_maximum() -> None:
    operation = GroupLoadStoreSemantics(
        kind="store",
        dtype="int32",
        items_per_thread=1,
        offset=ArgumentBinding.static((1 << 63) - 1),
    )

    assert _plan(operation).require_supported().call.operation is operation


def test_static_offset_rejects_signed_i64_overflow() -> None:
    with pytest.raises(ValueError, match="fit a signed 64-bit integer"):
        operation = GroupLoadStoreSemantics(
            kind="store",
            dtype="int32",
            items_per_thread=1,
            offset=ArgumentBinding.static(1 << 63),
        )
        _plan(operation)


def test_static_integer_controls_are_canonicalized_before_identity() -> None:
    numpy_controls = GroupLoadStoreSemantics(
        kind="load",
        dtype="int32",
        items_per_thread=1,
        valid_items=ArgumentBinding.static(np.int32(1)),
        offset=ArgumentBinding.static(np.int64(2)),
    )
    python_controls = GroupLoadStoreSemantics(
        kind="load",
        dtype="int32",
        items_per_thread=1,
        valid_items=ArgumentBinding.static(1),
        offset=ArgumentBinding.static(2),
    )

    assert type(numpy_controls.valid_items.value) is int
    assert type(numpy_controls.offset.value) is int
    assert numpy_controls == python_controls
