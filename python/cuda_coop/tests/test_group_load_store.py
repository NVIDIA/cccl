# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check the compiler contracts produced by shared Load and Store planning.

Plans select a CUB implementation and describe its arguments, participation,
scratch storage, and reuse barriers. These tests inspect plans on the host;
they do not compile or execute the selected device code.
"""

import numpy as np
import pytest

from cuda.coop._core import (
    INT64,
    ArgumentBinding,
    ArgumentKind,
    Array,
    GroupLoadStoreAlgorithm,
    GroupLoweringTarget,
    Pointer,
    PreconditionEnforcement,
    StorageOwnership,
    SynchronizationScope,
    UnsupportedReasonCode,
    this_block,
    this_warp,
)
from tests._group_planning import _load_store, _plan


@pytest.mark.parametrize("kind", ["load", "store"])
def test_block_plan_selects_cub_with_exact_launch_dimensions(kind):
    plan = _plan(this_block(), _load_store(kind), (8, 4, 1))

    assert plan.target is GroupLoweringTarget.CUB_BLOCK
    assert plan.provenance.cpp_class == f"cub::Block{kind.title()}"
    arguments = plan.implementation.template_arguments
    assert tuple(arguments[f"BLOCK_DIM_{axis}"] for axis in "XYZ") == (8, 4, 1)


@pytest.mark.parametrize("kind", ["load", "store"])
@pytest.mark.parametrize("width", [1, 2, 4, 8, 16, 32])
def test_warp_plan_selects_cub_for_the_group_width(kind, width):
    group = this_warp() if width == 32 else this_warp().group_by(width)
    plan = _plan(group, _load_store(kind), 64)

    assert plan.target is GroupLoweringTarget.CUB_WARP
    assert plan.provenance.cpp_class == f"cub::Warp{kind.title()}"
    assert (
        plan.implementation.template_arguments["LOGICAL_WARP_THREADS"] == width
    )


@pytest.mark.parametrize("kind", ["load", "store"])
@pytest.mark.parametrize("group", [this_block(), this_warp()])
def test_load_store_abi_offsets_the_global_pointer(kind, group):
    plan = _plan(group, _load_store(kind, offset=ArgumentBinding.runtime()))
    _, pointer, items, offset = plan.implementation.parameters[0]

    assert isinstance(pointer, Pointer) and isinstance(items, Array)
    assert (pointer.is_output, items.is_output) == (
        kind == "store",
        kind == "load",
    )
    assert offset.pointer_arg_index == 0
    assert offset.dtype == INT64


@pytest.mark.parametrize("algorithm", list(GroupLoadStoreAlgorithm))
def test_block_scratch_and_reuse_barrier_follow_the_algorithm(algorithm):
    plan = _plan(this_block(), _load_store(algorithm=algorithm))
    uses_scratch = algorithm in {
        GroupLoadStoreAlgorithm.TRANSPOSE,
        GroupLoadStoreAlgorithm.WARP_TRANSPOSE,
        GroupLoadStoreAlgorithm.WARP_TRANSPOSE_TIMESLICED,
    }

    assert plan.implementation.template_arguments["ALGORITHM"] == (
        f"::cub::BLOCK_LOAD_{algorithm.value.upper()}"
    )
    assert plan.temp_storage.ownership is (
        StorageOwnership.IMPLEMENTATION
        if uses_scratch
        else StorageOwnership.NONE
    )
    assert plan.synchronization.storage_reuse_barrier is (
        SynchronizationScope.BLOCK
        if uses_scratch
        else SynchronizationScope.NONE
    )


@pytest.mark.parametrize("width", [8, 32])
@pytest.mark.parametrize(
    "algorithm", ["direct", "striped", "vectorize", "transpose"]
)
def test_warp_scratch_is_per_group(width, algorithm):
    group = this_warp() if width == 32 else this_warp().group_by(width)
    plan = _plan(group, _load_store(algorithm=algorithm), 64)
    uses_scratch = algorithm == "transpose"

    assert plan.implementation.template_arguments["ALGORITHM"] == (
        f"::cub::WARP_LOAD_{algorithm.upper()}"
    )
    assert plan.temp_storage.instances == (
        64 // width if uses_scratch else None
    )
    assert plan.synchronization.storage_reuse_barrier is (
        SynchronizationScope.WARP if uses_scratch else SynchronizationScope.NONE
    )


@pytest.mark.parametrize("width", [8, 32])
def test_warp_effective_offset_leaves_room_for_every_group_tile(width):
    group = this_warp() if width == 32 else this_warp().group_by(width)
    maximum_offset = (1 << 63) - 1 - (64 // width - 1) * width * 2
    plan = _plan(
        group, _load_store(offset=ArgumentBinding.static(maximum_offset)), 64
    )

    # The adapter adds the group tile origin to this preserved user offset.
    assert plan.call.operation.offset.value == maximum_offset
    assert plan.implementation.metadata["effective_offset_stride"] == width * 2
    assert (
        plan.implementation.parameters[0][-1].argument_kind
        is ArgumentKind.RUNTIME
    )
    assert (
        plan.participation.argument_preconditions[0].maximum == maximum_offset
    )
    with pytest.raises(ValueError, match="tile origin must fit"):
        _plan(
            group,
            _load_store(offset=ArgumentBinding.static(maximum_offset + 1)),
            64,
        )


@pytest.mark.parametrize(
    ("group", "tile_size"),
    [(this_block(), 128), (this_warp(), 64), (this_warp().group_by(8), 16)],
)
def test_valid_items_bounds_use_the_group_tile(group, tile_size):
    for value in (0, tile_size):
        _plan(group, _load_store(valid_items=ArgumentBinding.static(value)), 64)
    for value in (-1, tile_size + 1):
        with pytest.raises(ValueError, match="group tile size"):
            _plan(
                group,
                _load_store(valid_items=ArgumentBinding.static(value)),
                64,
            )

    runtime = _plan(
        group, _load_store(valid_items=ArgumentBinding.runtime()), 64
    )
    condition = runtime.participation.argument_preconditions[0]
    assert (condition.minimum, condition.maximum) == (0, tile_size)
    assert condition.enforcement is PreconditionEnforcement.CALLER


@pytest.mark.parametrize(
    ("group", "scope"),
    [
        (this_block(), SynchronizationScope.BLOCK),
        (this_warp(), SynchronizationScope.WARP),
    ],
)
def test_caller_scratch_auto_sync_changes_the_reuse_barrier_and_artifact(
    group, scope
):
    def plan(auto_sync):
        return _plan(
            group,
            _load_store(
                algorithm="transpose",
                storage_ownership=StorageOwnership.CALLER,
                storage_sharing="shared",
                storage_auto_sync=auto_sync,
            ),
        )

    synchronized = plan(True)
    unsynchronized = plan(False)

    assert synchronized.temp_storage.exact_layout_required
    assert synchronized.synchronization.storage_reuse_barrier is scope
    assert (
        unsynchronized.synchronization.storage_reuse_barrier
        is SynchronizationScope.NONE
    )
    assert synchronized.artifact_key != unsynchronized.artifact_key


def test_unused_scratch_does_not_change_the_artifact():
    implicit = _plan(this_block(), _load_store())
    explicit = _plan(
        this_block(),
        _load_store(
            storage_ownership=StorageOwnership.CALLER,
            storage_sharing="shared",
            storage_size_in_bytes=256,
            storage_alignment=16,
        ),
    )

    assert explicit.temp_storage.ownership is StorageOwnership.NONE
    assert implicit.artifact_key == explicit.artifact_key


@pytest.mark.parametrize(
    "algorithm", ["warp_transpose", "warp_transpose_timesliced"]
)
@pytest.mark.parametrize("group", [this_warp(), this_warp().group_by(8)])
def test_warp_groups_reject_block_only_algorithms(group, algorithm):
    plan = _plan(group, _load_store(algorithm=algorithm), 64)

    assert plan.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT


@pytest.mark.parametrize(
    ("group", "block_size", "reason"),
    [
        (
            this_warp().group_by(3, exhaustive=False),
            64,
            UnsupportedReasonCode.GROUP_KIND,
        ),
        (this_warp(), 48, UnsupportedReasonCode.PARTIAL_PHYSICAL_WARP),
        (
            this_warp().group_by(8),
            48,
            UnsupportedReasonCode.PARTIAL_PHYSICAL_WARP,
        ),
    ],
)
def test_unsupported_warp_membership(group, block_size, reason):
    plan = _plan(group, _load_store(), block_size)

    assert plan.unsupported.code is reason


def test_different_algorithms_and_warp_widths_do_not_share_artifacts():
    plans = [
        *(
            _plan(this_block(), _load_store(algorithm=algorithm))
            for algorithm in GroupLoadStoreAlgorithm
        ),
        *(
            _plan(this_warp().group_by(width), _load_store())
            for width in (8, 16, 32)
        ),
    ]

    assert len({plan.artifact_key for plan in plans}) == len(plans)


def test_numpy_static_controls_share_the_same_artifact():
    plain = _load_store(
        valid_items=ArgumentBinding.static(5), offset=ArgumentBinding.static(7)
    )
    numpy = _load_store(
        valid_items=ArgumentBinding.static(np.int32(5)),
        offset=ArgumentBinding.static(np.int64(7)),
    )

    assert (
        _plan(this_block(), plain).artifact_key
        == _plan(this_block(), numpy).artifact_key
    )


def test_offsets_reject_negative_constants_and_record_runtime_preconditions():
    with pytest.raises(ValueError, match="static offset must be nonnegative"):
        _load_store(offset=ArgumentBinding.static(-1))

    runtime = _plan(this_block(), _load_store(offset=ArgumentBinding.runtime()))
    condition = runtime.participation.argument_preconditions[0]
    assert condition.minimum == 0
    assert condition.enforcement is PreconditionEnforcement.CALLER
