# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

import copy

import pytest

from ....support.paths import PACKAGE_ROOT

SOURCE_ROOT = PACKAGE_ROOT

from cuda.coop._core import ResultVisibility, ThreadHierarchy, this_block, this_warp
from cuda.coop.cutlass._internal import ThreadData
from cuda.coop.cutlass._value_metadata import (
    DefinedThreadDomain,
    ResolvedGroupIdentity,
    attach_thread_data_metadata,
    merge_value_metadata,
    metadata_for_group,
    thread_data_metadata,
    validate_operand_domains,
)


def _resolved_block(width=64, source="test"):
    return this_block().with_hierarchy(
        ThreadHierarchy._resolved(block_dim=width),
        source=source,
    )


def test_resolved_group_identity_ignores_source_but_preserves_shape():
    first = ResolvedGroupIdentity(_resolved_block(source="reqntid"))
    second = ResolvedGroupIdentity(_resolved_block(source="launch_config"))
    dimensional = ResolvedGroupIdentity(
        this_block().with_hierarchy(ThreadHierarchy._resolved(block_dim=(8, 4, 2)))
    )

    assert first == second
    assert hash(first) == hash(second)
    assert first != dimensional


def test_explicit_narrower_group_is_covered_by_full_domain_metadata():
    block = _resolved_block()
    warp = this_warp().with_hierarchy(block.hierarchy)
    loaded = metadata_for_group(block, visibility=ResultVisibility.PER_MEMBER)

    assert loaded.defined_domain == DefinedThreadDomain.all_callers()
    assert loaded.defined_domain.covers(ResolvedGroupIdentity(warp))


def test_implicit_current_per_member_metadata_needs_no_static_identity():
    current = this_block()
    metadata = metadata_for_group(
        current,
        visibility=ResultVisibility.PER_MEMBER,
    )

    assert metadata.defined_domain == DefinedThreadDomain.all_callers()
    with pytest.raises(ValueError, match="static resolved group"):
        metadata_for_group(current, visibility=ResultVisibility.GROUP_ROOT)


def test_non_exhaustive_mapping_domain_rejects_wider_group():
    block = _resolved_block(320)
    mapped = block.group_by(3, exhaustive=False)
    metadata = metadata_for_group(
        mapped,
        visibility=ResultVisibility.ALL_MEMBERS,
    )

    assert metadata.defined_domain.covers(ResolvedGroupIdentity(mapped))
    assert not metadata.defined_domain.covers(ResolvedGroupIdentity(block))


def test_root_only_domain_rejects_every_cooperative_target():
    block = _resolved_block()
    metadata = metadata_for_group(block, visibility=ResultVisibility.GROUP_ROOT)

    assert metadata.defined_domain.contains_roots_only
    assert not metadata.defined_domain.covers(ResolvedGroupIdentity(block))


def test_thread_data_copy_and_composition_preserve_or_merge_metadata():
    block = _resolved_block()
    warp = this_warp().with_hierarchy(block.hierarchy)
    block_metadata = metadata_for_group(
        block,
        visibility=ResultVisibility.PER_MEMBER,
    )
    warp_metadata = metadata_for_group(
        warp,
        visibility=ResultVisibility.ALL_MEMBERS,
    )
    first = attach_thread_data_metadata(
        ThreadData.from_values(1, 2),
        block_metadata,
    )
    second = attach_thread_data_metadata(
        ThreadData.from_values(3),
        warp_metadata,
    )

    assert thread_data_metadata(copy.copy(first)) == block_metadata
    assert thread_data_metadata(copy.deepcopy(first)) == block_metadata
    merged = merge_value_metadata(
        (thread_data_metadata(first), thread_data_metadata(second))
    )
    assert merged is not None
    assert merged.defined_domain == DefinedThreadDomain.all_callers()
    assert merged.visibility is ResultVisibility.PER_MEMBER


def test_new_uninitialized_thread_data_drops_group_metadata():
    value = attach_thread_data_metadata(
        ThreadData.from_values(1, 2),
        metadata_for_group(
            _resolved_block(),
            visibility=ResultVisibility.PER_MEMBER,
        ),
    )

    assert thread_data_metadata(value._new_uninitialized()) is None


def test_domain_validation_rejects_root_only_and_wider_targets():
    block = _resolved_block(320)
    mapped = block.group_by(3, exhaustive=False)
    mapped_value = attach_thread_data_metadata(
        ThreadData.from_values(1),
        metadata_for_group(mapped, visibility=ResultVisibility.ALL_MEMBERS),
    )
    root_value = attach_thread_data_metadata(
        ThreadData.from_values(2),
        metadata_for_group(block, visibility=ResultVisibility.GROUP_ROOT),
    )

    validate_operand_domains(
        mapped,
        {"value": mapped_value},
        scope="cuda.coop.cutlass",
        primitive_name="reduce",
    )
    with pytest.raises(ValueError, match="not defined for every member"):
        validate_operand_domains(
            block,
            {"value": mapped_value},
            scope="cuda.coop.cutlass",
            primitive_name="reduce",
        )
    with pytest.raises(ValueError, match="defined only at group roots"):
        validate_operand_domains(
            block,
            {"value": root_value},
            scope="cuda.coop.cutlass",
            primitive_name="store",
        )
