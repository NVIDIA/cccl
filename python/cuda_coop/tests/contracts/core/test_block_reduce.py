# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import numpy as np
import pytest

from cuda.coop._core import (
    ArgumentBinding,
    BlockReduceAlgorithm,
    BlockReduceOperation,
    BlockReduceOperator,
    GroupLoweringTarget,
    GroupReduceSemantics,
    LaunchFactOrigin,
    LaunchFacts,
    ResultOwnership,
    ResultVisibility,
    StorageOwnership,
    SynchronizationScope,
    make_block_reduce_spec,
    make_group_primitive_call,
    normalize_block_reduce_algorithm,
    normalize_block_reduce_operator,
    plan_group_primitive,
    this_block,
)
from cuda.coop._core._symbols import semantic_token


def _plan(operation: GroupReduceSemantics, block_dim=(8, 4, 2)):
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


def test_block_sum_has_root_only_cub_plan() -> None:
    plan = _plan(GroupReduceSemantics(dtype="int32", operation="sum"))
    plan.require_supported()

    assert plan.target is GroupLoweringTarget.CUB_BLOCK
    assert plan.resolved_group.static_size == 64
    assert plan.participation is not None
    assert plan.participation.exact_block_dim == (8, 4, 2)
    assert plan.participation.complete_membership
    assert plan.participation.uniform_arguments == ()
    assert plan.implementation is not None
    assert plan.implementation.operation is BlockReduceOperation.SUM
    assert plan.implementation.binary_op is BlockReduceOperator.SUM
    assert plan.implementation.method_name == "Sum"
    assert plan.result is not None
    assert plan.result.visibility is ResultVisibility.ROOT_ONLY
    assert plan.result.ownership is ResultOwnership.GROUP_ROOT
    assert plan.result.root_rank == 0
    assert plan.synchronization is not None
    assert plan.synchronization.storage_reuse_barrier is SynchronizationScope.BLOCK
    assert plan.temp_storage is not None
    assert plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert plan.provenance is not None
    assert plan.provenance.header == "cub/block/block_reduce.cuh"
    assert plan.provenance.cpp_class == "cub::BlockReduce"
    assert plan.provenance.method == "Sum"


@pytest.mark.parametrize(
    "algorithm",
    (
        BlockReduceAlgorithm.RAKING_COMMUTATIVE_ONLY,
        BlockReduceAlgorithm.RAKING,
        BlockReduceAlgorithm.WARP_REDUCTIONS,
    ),
)
def test_block_reduce_preserves_deterministic_algorithm(algorithm) -> None:
    plan = _plan(
        GroupReduceSemantics(
            dtype="float32",
            operation="reduce",
            binary_op="max",
            algorithm=algorithm,
        )
    ).require_supported()

    assert plan.implementation is not None
    assert plan.implementation.algorithm is algorithm
    assert plan.implementation.binary_op is BlockReduceOperator.MAX
    assert plan.provenance is not None
    assert plan.provenance.method == "Reduce"


@pytest.mark.parametrize(
    ("alias", "expected"),
    (
        (None, BlockReduceOperator.SUM),
        ("+", BlockReduceOperator.SUM),
        ("multiply", BlockReduceOperator.MULTIPLIES),
        ("minimum", BlockReduceOperator.MIN),
        ("maximum", BlockReduceOperator.MAX),
        ("&", BlockReduceOperator.BIT_AND),
        ("|", BlockReduceOperator.BIT_OR),
        ("^", BlockReduceOperator.BIT_XOR),
    ),
)
def test_builtin_operator_aliases_are_canonical(alias, expected) -> None:
    assert normalize_block_reduce_operator(alias) is expected


def test_partial_reduce_tracks_uniform_valid_prefix() -> None:
    operation = GroupReduceSemantics(
        dtype="uint32",
        operation="reduce",
        binary_op="bit_xor",
        algorithm="raking",
        valid_items=ArgumentBinding.runtime(),
    )
    plan = _plan(operation).require_supported()

    assert operation.has_valid_items
    assert plan.participation is not None
    assert plan.participation.uniform_arguments == ("valid_items",)
    assert plan.participation.valid_member_selection == (
        "first valid_items block members"
    )
    assert plan.implementation is not None
    assert plan.implementation.valid_items


def test_static_valid_prefix_accepts_numpy_integer() -> None:
    plain = _plan(
        GroupReduceSemantics(
            dtype="int32",
            operation="sum",
            valid_items=ArgumentBinding.static(5),
        )
    )
    numpy = _plan(
        GroupReduceSemantics(
            dtype="int32",
            operation="sum",
            valid_items=ArgumentBinding.static(np.int32(5)),
        )
    )

    assert plain.artifact_key == numpy.artifact_key


def test_missing_exact_dimensions_is_typed_unsupported_plan() -> None:
    operation = GroupReduceSemantics(dtype="int32", operation="sum")
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
    operation = GroupReduceSemantics(dtype="int32", operation="sum")
    call = make_group_primitive_call(this_block(), operation)
    plan = plan_group_primitive(call, LaunchFacts(exact_block_dim=64))

    assert plan.target is GroupLoweringTarget.UNSUPPORTED
    assert plan.unsupported is not None
    assert plan.unsupported.code.value == "unverified_exact_block_dim"
    with pytest.raises(NotImplementedError, match="compiler-verified"):
        plan.require_supported()


def test_artifact_identity_tracks_operator_algorithm_and_valid_prefix() -> None:
    baseline = _plan(
        GroupReduceSemantics(dtype="int32", binary_op="max", algorithm="raking")
    )
    other_operator = _plan(
        GroupReduceSemantics(dtype="int32", binary_op="min", algorithm="raking")
    )
    other_algorithm = _plan(
        GroupReduceSemantics(
            dtype="int32",
            binary_op="max",
            algorithm="warp_reductions",
        )
    )
    partial = _plan(
        GroupReduceSemantics(
            dtype="int32",
            binary_op="max",
            algorithm="raking",
            valid_items=ArgumentBinding.runtime(),
        )
    )

    assert baseline.artifact_key == _plan(baseline.call.operation).artifact_key
    assert baseline.artifact_key != other_operator.artifact_key
    assert baseline.artifact_key != other_algorithm.artifact_key
    assert baseline.artifact_key != partial.artifact_key


@pytest.mark.parametrize("valid_items", (True, 0, -1, 65, 1.5, "one"))
def test_static_valid_items_are_checked_against_block(valid_items) -> None:
    operation = GroupReduceSemantics(
        dtype="int32",
        operation="sum",
        valid_items=ArgumentBinding.static(valid_items),
    )
    with pytest.raises((TypeError, ValueError), match="valid_items"):
        _plan(operation, block_dim=64)


@pytest.mark.parametrize("algorithm", (True, "nondeterministic", "striped"))
def test_only_supported_deterministic_algorithms_are_accepted(algorithm) -> None:
    with pytest.raises(ValueError, match=r"cuda\.coop reduction algorithm"):
        normalize_block_reduce_algorithm(algorithm)


def test_python_callbacks_are_not_portable_operators() -> None:
    with pytest.raises(ValueError, match=r"cuda\.coop\.reduce binary_op"):
        normalize_block_reduce_operator(lambda left, right: left + right)


def test_sum_diagnostic_uses_the_python_api_name() -> None:
    with pytest.raises(ValueError, match=r"cuda\.coop\.sum requires"):
        make_block_reduce_spec(
            dtype="int32",
            block_dim=32,
            operation="sum",
            binary_op="max",
        )


def test_block_spec_normalizes_dimensions_and_selectors() -> None:
    spec = make_block_reduce_spec(
        dtype="int32",
        block_dim=(32, 2),
        operation="reduce",
        binary_op="maximum",
        algorithm="raking-commutative-only",
        valid_items=True,
    )

    assert spec.block_dim == (32, 2, 1)
    assert spec.binary_op is BlockReduceOperator.MAX
    assert spec.algorithm is BlockReduceAlgorithm.RAKING_COMMUTATIVE_ONLY
    assert spec.valid_items


def test_semantic_tokens_sort_heterogeneous_mapping_keys_stably() -> None:
    left = {1: "integer", "1": "string"}
    right = {"1": "string", 1: "integer"}

    assert semantic_token(left) == semantic_token(right)
