# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402


import pytest

from cuda.coop._core import (
    COMPLETE_WARP_GROUP_KINDS,
    THREAD_GROUP_KINDS,
    AlgorithmSpec,
    ArgumentBinding,
    ArgumentKind,
    CudaxReturnKind,
    CxxFunction,
    CxxOperator,
    Dependency,
    GroupAdjacentDifferenceSemantics,
    GroupDiscontinuitySemantics,
    GroupExchangeMode,
    GroupExchangeSemantics,
    GroupLoadStoreAlgorithm,
    GroupLoadStoreKind,
    GroupLoadStoreSemantics,
    GroupLoweringTarget,
    GroupMergeSortSemantics,
    GroupOperandKind,
    GroupRadixRankSemantics,
    GroupRadixSortSemantics,
    GroupReduceSemantics,
    GroupScanMode,
    GroupScanSemantics,
    GroupShuffleSemantics,
    LaunchFactOrigin,
    LaunchFacts,
    LogicalResultContract,
    ParameterRole,
    PreconditionEnforcement,
    ResultOwnership,
    ResultVisibility,
    RuntimeValue,
    StatefulOperator,
    StorageOwnership,
    SynchronizationScope,
    ThreadGroup,
    ThreadHierarchy,
    UnsupportedReasonCode,
    make_group_primitive_call,
    make_scan_semantics,
    merge_launch_facts,
    plan_group_primitive,
    resolve_thread_group,
    this_block,
    this_cluster,
    this_grid,
    this_thread,
    this_warp,
)
from cuda.coop._core.block import (
    BlockAdjacentDifferenceDirection,
    BlockDiscontinuityMode,
    BlockReduceAlgorithm,
    BlockScanAlgorithm,
    make_block_adjacent_difference_semantics,
    make_block_discontinuity_semantics,
    make_block_exchange_semantics,
    make_block_exchange_spec,
    make_block_merge_sort_semantics,
    make_block_radix_rank_semantics,
    make_block_radix_sort_semantics,
    make_block_reduce_semantics,
    make_block_scan_spec,
    make_block_shuffle_semantics,
)
from cuda.coop._core.warp import (
    make_warp_exchange_spec,
    make_warp_reduce_spec,
    make_warp_scan_spec,
)


def _reduce(**overrides):
    broadcast = overrides.pop("broadcast", True)
    cub_algorithm = overrides.pop("cub_algorithm", None)
    operation = overrides.pop("operation", "sum")
    operand_kind = GroupOperandKind(overrides.pop("operand_kind", "scalar"))
    reduce_operator = overrides.pop("reduce_operator", None)
    if operation == "max":
        operation = "reduce"
        reduce_operator = CxxOperator("::cuda::maximum<>{}", "int")
    primitive = make_block_reduce_semantics(
        dtype=overrides.pop("dtype", "int"),
        operation=operation,
        value_kind=operand_kind.value,
        items_per_thread=overrides.pop("items_per_thread", 1),
        valid_items=overrides.pop("valid_items", False),
        reduce_operator=reduce_operator,
    )
    assert not overrides
    return GroupReduceSemantics(
        primitive=primitive,
        broadcast=broadcast,
        cub_algorithm=cub_algorithm,
    )


def _exchange(mode="blocked_to_striped", items_per_thread=3, **overrides):
    semantics = GroupExchangeSemantics(
        make_block_exchange_semantics(
            dtype=overrides.pop("dtype", "int"),
            mode=mode,
            items_per_thread=items_per_thread,
            warp_time_slicing=overrides.pop("warp_time_slicing", False),
            rank_dtype=overrides.pop("rank_dtype", None),
            valid_flag_dtype=overrides.pop("valid_flag_dtype", None),
        )
    )
    assert not overrides
    return semantics


def _adjacent_difference(**overrides):
    semantics = GroupAdjacentDifferenceSemantics(
        make_block_adjacent_difference_semantics(
            dtype=overrides.pop("dtype", "int"),
            items_per_thread=overrides.pop("items_per_thread", 2),
            direction=overrides.pop("direction", "left"),
            difference_operator=overrides.pop(
                "difference_operator",
                CxxOperator(
                    "::cuda::std::minus<T>",
                    Dependency("T"),
                    name="difference_op",
                ),
            ),
            valid_items=overrides.pop("valid_items", None),
            tile_predecessor_item=overrides.pop("tile_predecessor_item", None),
            tile_successor_item=overrides.pop("tile_successor_item", None),
        )
    )
    assert not overrides
    return semantics


def _discontinuity(**overrides):
    semantics = GroupDiscontinuitySemantics(
        make_block_discontinuity_semantics(
            dtype=overrides.pop("dtype", "int"),
            flag_dtype=overrides.pop("flag_dtype", "flag"),
            items_per_thread=overrides.pop("items_per_thread", 2),
            mode=overrides.pop("mode", "heads"),
            flag_operator=overrides.pop(
                "flag_operator",
                CxxOperator(
                    "::cuda::std::not_equal_to<T>",
                    Dependency("T"),
                    name="flag_op",
                ),
            ),
            tile_predecessor_item=overrides.pop("tile_predecessor_item", None),
            tile_successor_item=overrides.pop("tile_successor_item", None),
        )
    )
    assert not overrides
    return semantics


def _shuffle(**overrides):
    items_per_thread = overrides.pop("items_per_thread", None)
    semantics = GroupShuffleSemantics(
        make_block_shuffle_semantics(
            dtype=overrides.pop("dtype", "int"),
            mode=overrides.pop(
                "mode",
                "offset" if items_per_thread is None else "up",
            ),
            items_per_thread=items_per_thread,
            distance=overrides.pop(
                "distance",
                (
                    ArgumentBinding.runtime()
                    if items_per_thread is None
                    else ArgumentBinding.omitted()
                ),
            ),
            block_prefix=overrides.pop("block_prefix", False),
            block_suffix=overrides.pop("block_suffix", False),
        )
    )
    assert not overrides
    return semantics


def _merge_sort(**overrides):
    descending = overrides.pop("descending", False)
    semantics = GroupMergeSortSemantics(
        make_block_merge_sort_semantics(
            key_dtype=overrides.pop("key_dtype", "int"),
            value_dtype=overrides.pop("value_dtype", None),
            items_per_thread=overrides.pop("items_per_thread", 2),
            compare_operator=CxxOperator(
                (
                    "::cuda::std::greater<KeyT>"
                    if descending
                    else "::cuda::std::less<KeyT>"
                ),
                Dependency("KeyT"),
                name="compare_op",
            ),
            valid_items=overrides.pop("valid_items", None),
            oob_default=overrides.pop("oob_default", None),
        )
    )
    assert not overrides
    return semantics


def _radix_sort(**overrides):
    operand_kind = GroupOperandKind(overrides.pop("operand_kind", "array"))
    primitive = make_block_radix_sort_semantics(
        key_dtype=overrides.pop("key_dtype", "int"),
        value_dtype=overrides.pop("value_dtype", None),
        items_per_thread=overrides.pop("items_per_thread", 2),
        descending=overrides.pop("descending", False),
        begin_bit=RuntimeValue("begin_bit"),
        end_bit=RuntimeValue("end_bit"),
        key_bit_width=overrides.pop("key_bit_width", 32),
        bit_policy="explicit",
    )
    assert not overrides
    return GroupRadixSortSemantics(primitive, operand_kind=operand_kind)


def _radix_rank(**overrides):
    operand_kind = GroupOperandKind(overrides.pop("operand_kind", "array"))
    primitive = make_block_radix_rank_semantics(
        key_dtype=overrides.pop("key_dtype", "unsigned int"),
        items_per_thread=overrides.pop("items_per_thread", 2),
        begin_bit=overrides.pop("begin_bit", 0),
        end_bit=overrides.pop("end_bit", 8),
        key_bit_width=overrides.pop("key_bit_width", 32),
        descending=overrides.pop("descending", False),
        block_threads=overrides.pop("block_threads", 64),
        exclusive_digit_prefix_items_per_thread=overrides.pop("prefix_items", None),
    )
    input_dtype = overrides.pop("input_dtype", "int")
    assert not overrides
    return GroupRadixRankSemantics(
        primitive,
        input_dtype=input_dtype,
        operand_kind=operand_kind,
    )


def _scan(**overrides):
    cub_algorithm = overrides.pop("cub_algorithm", None)
    valid_items = overrides.pop("valid_items", ArgumentBinding.omitted())
    operand_kind = GroupOperandKind(overrides.pop("operand_kind", "scalar"))
    primitive = make_scan_semantics(
        dtype=overrides.pop("dtype", "int"),
        mode=overrides.pop("mode", "exclusive"),
        value_kind=operand_kind.value,
        items_per_thread=overrides.pop("items_per_thread", 1),
        scan_operator=overrides.pop("scan_operator", None),
        initial_value=overrides.pop("initial_value", None),
        aggregate=overrides.pop("aggregate", False),
        prefix_callback=overrides.pop("prefix_callback", None),
    )
    assert not overrides
    return GroupScanSemantics(
        primitive,
        cub_algorithm=cub_algorithm,
        valid_items=valid_items,
    )


def _load_store(kind="load", **overrides):
    return GroupLoadStoreSemantics(
        kind=GroupLoadStoreKind(kind),
        dtype=overrides.pop("dtype", "int"),
        items_per_thread=overrides.pop("items_per_thread", 2),
        algorithm=overrides.pop("algorithm", GroupLoadStoreAlgorithm.DIRECT),
        valid_items=overrides.pop("valid_items", ArgumentBinding.omitted()),
        oob_default=overrides.pop("oob_default", ArgumentBinding.omitted()),
        offset=overrides.pop("offset", ArgumentBinding.omitted()),
    )


def _plan(group, operation, launch=(64, 1, 1)):
    facts = launch if isinstance(launch, LaunchFacts) else LaunchFacts(launch)
    return plan_group_primitive(
        make_group_primitive_call(group, operation),
        facts,
    )


def test_launch_facts_keep_exact_bounds_and_provenance_distinct():
    exact = LaunchFacts(
        exact_block_dim=(8, 4),
        max_block_dim=(16, 8),
        provenance=LaunchFactOrigin("exact_block_dim", "call_metadata"),
    )
    same_facts = LaunchFacts(
        exact_block_dim=(8, 4),
        max_block_dim=(16, 8),
        provenance=LaunchFactOrigin("exact_block_dim", "reqntid"),
    )

    assert exact == same_facts
    assert hash(exact) == hash(same_facts)
    assert exact.exact_block_dim == (8, 4, 1)
    assert exact.exact_block_threads == 32
    assert exact.max_block_threads == 128
    assert exact.provenance != same_facts.provenance


def test_launch_fact_verification_is_diagnostic_not_semantic():
    asserted = LaunchFacts(
        cooperative_launch=True,
        provenance=LaunchFactOrigin(
            "cooperative_launch",
            "call_metadata",
        ),
    )
    verified = LaunchFacts(
        cooperative_launch=True,
        provenance=LaunchFactOrigin(
            "cooperative_launch",
            "kernel_launch_config",
            verified=True,
        ),
    )

    assert asserted == verified
    assert not asserted.is_verified("cooperative_launch")
    assert verified.is_verified("cooperative_launch")
    assert not verified.is_verified("cluster_launch")


def test_merge_launch_facts_reconciles_without_promoting_maximums():
    merged = merge_launch_facts(
        LaunchFacts(
            max_block_dim=(256, 8, 2),
            provenance=LaunchFactOrigin("max_block_dim", "maxntid:a"),
        ),
        LaunchFacts(
            max_block_dim=(128, 16, 1),
            cooperative_launch=True,
            provenance=LaunchFactOrigin("max_block_dim", "maxntid:b"),
        ),
    )

    assert merged.exact_block_dim is None
    assert merged.max_block_dim == (128, 8, 1)
    assert merged.cooperative_launch is True
    assert len(merged.provenance) == 2

    with pytest.raises(ValueError, match="conflicting exact_block_dim"):
        merge_launch_facts(
            LaunchFacts(exact_block_dim=32),
            LaunchFacts(exact_block_dim=64),
        )
    with pytest.raises(ValueError, match="conflicting cooperative_launch"):
        merge_launch_facts(
            LaunchFacts(cooperative_launch=True),
            LaunchFacts(cooperative_launch=False),
        )
    with pytest.raises(ValueError, match="exceeds max_block_dim"):
        LaunchFacts(exact_block_dim=(64, 2), max_block_dim=(32, 2))


def test_exact_launch_is_required_and_current_groups_are_resolved():
    operation = _reduce()
    missing = _plan(
        this_block(),
        operation,
        LaunchFacts(max_block_dim=256),
    )

    assert missing.target is GroupLoweringTarget.UNSUPPORTED
    assert missing.unsupported.code is UnsupportedReasonCode.MISSING_EXACT_BLOCK_DIM
    assert missing.artifact_key is None
    with pytest.raises(NotImplementedError, match="only an upper bound"):
        missing.require_supported()

    resolved = _plan(this_block(), operation, (8, 4, 1))
    assert resolved.resolved_group.block_dim == (8, 4, 1)
    assert resolved.resolved_group.source == "launch_facts"


def test_shared_group_resolution_builds_the_requested_enclosing_hierarchy():
    facts = LaunchFacts(
        exact_block_dim=(8, 4, 2),
        exact_grid_dim=8,
        cluster_launch=False,
        provenance=(
            LaunchFactOrigin("cluster_launch", "launch_config", verified=True),
        ),
    )

    resolved = resolve_thread_group(
        this_thread(),
        facts,
        through_level="grid",
    ).require_supported()

    assert resolved.kind == "thread"
    assert resolved.hierarchy.block_dim == (8, 4, 2)
    assert resolved.hierarchy.cluster_dim == (1, 1, 1)
    assert resolved.hierarchy.grid_dim == (8, 1, 1)
    assert resolved.source == "launch_facts"


@pytest.mark.parametrize(
    ("group", "facts", "message"),
    (
        (
            this_cluster(),
            LaunchFacts(exact_block_dim=32, exact_cluster_dim=2),
            "backend-verified cluster launch state",
        ),
        (
            this_cluster(),
            LaunchFacts(
                exact_block_dim=32,
                exact_cluster_dim=2,
                cluster_launch=False,
                provenance=(
                    LaunchFactOrigin("cluster_launch", "launch_config", verified=True),
                ),
            ),
            "multi-block cluster operations require verified cluster launch",
        ),
        (
            this_grid(),
            LaunchFacts(
                exact_block_dim=32,
                cluster_launch=False,
                provenance=(
                    LaunchFactOrigin("cluster_launch", "launch_config", verified=True),
                ),
            ),
            "grid group operations require exact static grid dimensions",
        ),
        (
            this_grid(),
            LaunchFacts(
                exact_block_dim=32,
                exact_cluster_dim=2,
                exact_grid_dim=3,
                cluster_launch=True,
                provenance=(
                    LaunchFactOrigin("cluster_launch", "launch_config", verified=True),
                ),
            ),
            "grid dimensions must be divisible by the cluster dimensions",
        ),
    ),
)
def test_shared_group_resolution_rejects_incomplete_launch_capabilities(
    group,
    facts,
    message,
):
    resolution = resolve_thread_group(group, facts)

    assert resolution.unsupported is not None
    assert resolution.unsupported.code is UnsupportedReasonCode.LAUNCH_CAPABILITY
    assert message in resolution.unsupported.message


@pytest.mark.parametrize(
    "group",
    [
        this_thread(),
        this_block(),
        this_cluster(),
        this_grid(),
    ],
)
@pytest.mark.parametrize(("block_threads", "is_supported"), [(48, False), (64, True)])
def test_shared_group_resolution_requires_complete_warps_for_warp_queries(
    group,
    block_threads,
    is_supported,
):
    facts = LaunchFacts(
        exact_block_dim=block_threads,
        exact_grid_dim=8,
        cluster_launch=False,
        provenance=(
            LaunchFactOrigin("cluster_launch", "launch_config", verified=True),
        ),
    )

    resolution = resolve_thread_group(group, facts, through_level="warp")

    if is_supported:
        assert resolution.require_supported().block_dim == (block_threads, 1, 1)
    else:
        assert (
            resolution.unsupported.code is UnsupportedReasonCode.PARTIAL_PHYSICAL_WARP
        )
        with pytest.raises(NotImplementedError, match="complete 32-thread warps"):
            resolution.require_supported()


_COMPLETE_WARP_GROUP_SAMPLES = (
    this_warp(),
    this_warp().group_by(8),
    this_block().group_by(1, exhaustive=False),
)
_NON_COMPLETE_WARP_GROUP_SAMPLES = (
    this_thread(),
    this_block(),
    this_cluster(),
    this_grid(),
)


def test_shared_group_samples_cover_complete_warp_partition():
    assert {
        group.kind for group in _COMPLETE_WARP_GROUP_SAMPLES
    } == COMPLETE_WARP_GROUP_KINDS
    assert {
        group.kind for group in _NON_COMPLETE_WARP_GROUP_SAMPLES
    } == THREAD_GROUP_KINDS - COMPLETE_WARP_GROUP_KINDS


@pytest.mark.parametrize(
    "group",
    _COMPLETE_WARP_GROUP_SAMPLES,
    ids=lambda group: group.kind,
)
def test_shared_group_resolution_enforces_every_complete_warp_group_kind(group):
    resolution = resolve_thread_group(
        group,
        LaunchFacts(exact_block_dim=48),
    )

    assert resolution.unsupported is not None
    assert resolution.unsupported.code is UnsupportedReasonCode.PARTIAL_PHYSICAL_WARP


@pytest.mark.parametrize(
    "group",
    _NON_COMPLETE_WARP_GROUP_SAMPLES,
    ids=lambda group: group.kind,
)
def test_shared_group_resolution_allows_non_complete_warp_group_kinds(group):
    facts = LaunchFacts(
        exact_block_dim=48,
        exact_grid_dim=8,
        cluster_launch=False,
        provenance=(
            LaunchFactOrigin("cluster_launch", "launch_config", verified=True),
        ),
    )

    resolved = resolve_thread_group(group, facts).require_supported()

    assert resolved.kind == group.kind
    if group.kind == "thread":
        assert resolved.is_current
    else:
        assert resolved.block_dim == (48, 1, 1)


def test_shared_group_resolution_reconciles_mapped_groups():
    group = this_block().group_by(2, exhaustive=False)

    resolved = resolve_thread_group(
        group,
        LaunchFacts(exact_block_dim=160),
    ).require_supported()

    assert resolved.kind == "warps_within_block"
    assert resolved.static_size == 64
    assert resolved.groups_per_parent == 2
    assert resolved.remainder_count == 1
    assert resolved.complete_membership is False


@pytest.mark.parametrize(
    ("group", "barrier"),
    [
        (this_block(), SynchronizationScope.BLOCK),
        (this_warp(), SynchronizationScope.WARP),
    ],
)
def test_default_reduce_selects_broadcasted_cudax(group, barrier):
    operation = _reduce(
        operand_kind=GroupOperandKind.ARRAY,
        items_per_thread=4,
    )
    plan = _plan(group, operation, 128)

    assert plan.target is GroupLoweringTarget.CUDAX_GROUP
    assert plan.implementation.primitive == "reduce"
    assert plan.implementation.overload == "broadcasted"
    assert plan.implementation.return_kind is CudaxReturnKind.VALUE
    assert [parameter.name for parameter in plan.implementation.parameters] == [
        "item0",
        "item1",
        "item2",
        "item3",
    ]
    assert plan.provenance.library == "CUDAX"
    assert plan.provenance.header == "cuda/experimental/coop.cuh"
    assert plan.result.visibility is ResultVisibility.ALL_MEMBERS
    assert plan.result.primary.ownership is ResultOwnership.EACH_MEMBER
    assert plan.result.primary.root_rank is None
    assert plan.result.operand_kind is GroupOperandKind.SCALAR
    assert plan.result.result_items_per_thread == 1
    assert plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert plan.temp_storage.instances is None
    assert plan.synchronization.storage_reuse_barrier is barrier


def test_logical_result_contract_enforces_group_root_rank_zero():
    valid = LogicalResultContract(
        name="value",
        dtype="int",
        visibility=ResultVisibility.GROUP_ROOT,
        ownership=ResultOwnership.GROUP_ROOT,
        operand_kind=GroupOperandKind.SCALAR,
        items_per_member=1,
        root_rank=0,
    )

    assert valid.root_rank == 0
    for invalid_rank in (None, -1, 1, 999, True):
        with pytest.raises(ValueError, match="root rank 0"):
            LogicalResultContract(
                name="value",
                dtype="int",
                visibility=ResultVisibility.GROUP_ROOT,
                ownership=ResultOwnership.GROUP_ROOT,
                operand_kind=GroupOperandKind.SCALAR,
                items_per_member=1,
                root_rank=invalid_rank,
            )
    with pytest.raises(ValueError, match="cannot define a root rank"):
        LogicalResultContract(
            name="value",
            dtype="int",
            visibility=ResultVisibility.ALL_MEMBERS,
            ownership=ResultOwnership.EACH_MEMBER,
            operand_kind=GroupOperandKind.SCALAR,
            items_per_member=1,
            root_rank=0,
        )


def test_warp_semantics_ignore_parent_cta_but_artifacts_do_not():
    operation = _reduce()
    first = _plan(this_warp(), operation, 64)
    second = _plan(
        ThreadGroup(
            kind="warp",
            hierarchy=ThreadHierarchy._resolved(
                block_dim=128,
                grid_dim=7,
                cluster_dim=2,
            ),
            source="resolved_with_irrelevant_facts",
        ),
        operation,
        128,
    )

    assert first.semantic_key == second.semantic_key
    assert first.artifact_key != second.artifact_key
    assert first.participation.exact_group_size == 32
    assert second.participation.exact_group_size == 32
    assert first.participation.complete_parent_partition


def test_diagnostic_sources_do_not_fragment_plan_or_artifact_identity():
    operation = _reduce()
    first = plan_group_primitive(
        make_group_primitive_call(
            ThreadGroup(
                kind="block",
                source="root_frontend",
            ),
            operation,
            source="root_frontend",
        ),
        LaunchFacts(
            exact_block_dim=64,
            provenance=LaunchFactOrigin("exact_block_dim", "call_metadata"),
        ),
    )
    second = plan_group_primitive(
        make_group_primitive_call(
            ThreadGroup(
                kind="block",
                source="scoped_frontend",
            ),
            operation,
            source="scoped_frontend",
        ),
        LaunchFacts(
            exact_block_dim=64,
            provenance=LaunchFactOrigin("exact_block_dim", "reqntid"),
        ),
    )

    assert first.semantic_key == second.semantic_key
    assert first.artifact_key == second.artifact_key
    assert first == second


def test_group_requests_reuse_scoped_core_semantics_exactly():
    reduce_primitive = make_block_reduce_semantics(
        dtype="int",
        operation="sum",
        value_kind="scalar",
        items_per_thread=1,
    )
    reduce_operation = GroupReduceSemantics(reduce_primitive)
    exchange_primitive = make_block_exchange_semantics(
        dtype="int",
        mode="striped_to_blocked",
        items_per_thread=2,
    )
    exchange_operation = GroupExchangeSemantics(exchange_primitive)
    scan_primitive = make_scan_semantics(
        dtype="int",
        mode="exclusive",
        value_kind="scalar",
        items_per_thread=1,
    )
    scan_operation = GroupScanSemantics(scan_primitive)

    block_scan = make_block_scan_spec(
        dtype="int",
        block_dim=(64, 1, 1),
        items_per_thread=1,
        mode="exclusive",
        algorithm="::cub::BLOCK_SCAN_RAKING",
        value_kind="scalar",
    )
    warp_scan = make_warp_scan_spec(
        dtype="int",
        threads_in_warp=32,
        mode="exclusive",
    )
    warp_reduce = make_warp_reduce_spec(
        dtype="int",
        threads_in_warp=32,
        operation="sum",
    )
    warp_exchange = make_warp_exchange_spec(
        dtype="int",
        items_per_thread=2,
        threads_in_warp=32,
        mode="striped_to_blocked",
    )
    block_exchange = make_block_exchange_spec(
        dtype="int",
        block_dim=(64, 1, 1),
        items_per_thread=2,
        mode="striped_to_blocked",
        value_form="out_of_place",
        warp_time_slicing=False,
    )

    assert reduce_operation.primitive is reduce_primitive
    assert exchange_operation.primitive is exchange_primitive
    assert scan_operation.primitive is scan_primitive
    assert warp_reduce.call.semantic_key == reduce_primitive.semantic_key
    assert warp_exchange.call.semantic_key == exchange_primitive.semantic_key
    assert block_scan.call.semantic_key == scan_primitive.semantic_key
    assert warp_scan.call.semantic_key == scan_primitive.semantic_key
    assert _plan(this_block(), reduce_operation).semantic_key[1][0] == (
        reduce_primitive.semantic_key
    )
    assert _plan(this_warp(), reduce_operation).semantic_key[1][0] == (
        reduce_primitive.semantic_key
    )
    assert _plan(this_block(), exchange_operation).semantic_key[1] == (
        exchange_primitive.semantic_key
    )
    assert _plan(this_block(), scan_operation).semantic_key[1][0] == (
        scan_primitive.semantic_key
    )
    assert _plan(this_block(), scan_operation).implementation == (
        block_scan.specialization
    )
    assert _plan(this_warp(), scan_operation).implementation == (
        warp_scan.specialization
    )
    assert _plan(this_block(), exchange_operation).implementation == (
        block_exchange.specialization
    )
    assert _plan(this_warp(), exchange_operation).implementation == (
        warp_exchange.specialization
    )


def test_partial_physical_warp_partition_fails_closed():
    plan = _plan(this_warp(), _reduce(), 48)
    mapped_plan = _plan(
        this_warp().group_by(8),
        _reduce(),
        48,
    )

    assert plan.target is GroupLoweringTarget.UNSUPPORTED
    assert plan.unsupported.code is UnsupportedReasonCode.PARTIAL_PHYSICAL_WARP
    assert "complete 32-thread warps" in plan.unsupported.message
    assert mapped_plan.target is GroupLoweringTarget.UNSUPPORTED
    assert mapped_plan.unsupported.code is UnsupportedReasonCode.PARTIAL_PHYSICAL_WARP


@pytest.mark.parametrize(
    ("group", "target", "class_name", "instances", "barrier"),
    [
        (this_block(), GroupLoweringTarget.CUB_BLOCK, "cub::BlockReduce", 1, "block"),
        (this_warp(), GroupLoweringTarget.CUB_WARP, "cub::WarpReduce", 4, "warp"),
    ],
)
def test_cub_only_reduce_is_root_only_with_exact_storage(
    group,
    target,
    class_name,
    instances,
    barrier,
):
    operation = _reduce(
        broadcast=False,
        valid_items=ArgumentBinding.static(17),
    )
    call = make_group_primitive_call(group, operation)
    plan = plan_group_primitive(call, LaunchFacts(exact_block_dim=128))

    assert plan.target is target
    assert isinstance(plan.implementation, AlgorithmSpec)
    assert plan.provenance.cpp_class == class_name
    assert plan.result.visibility is ResultVisibility.GROUP_ROOT
    assert plan.result.primary.ownership is ResultOwnership.GROUP_ROOT
    assert plan.result.primary.root_rank == 0
    assert plan.temp_storage.ownership is StorageOwnership.CALLER
    assert plan.temp_storage.instances == instances
    assert plan.temp_storage.cpp_type == "typename implementation_type::TempStorage"
    assert plan.synchronization.storage_reuse_barrier.value == barrier
    assert call.argument_classifications[1].kind is ArgumentKind.STATIC
    assert call.argument_classifications[1].role is ParameterRole.CONSTANT
    implementation_classifications = plan.implementation.classify_method()
    implementation_valid_items = next(
        item
        for item in implementation_classifications
        if item.name in {"num_valid", "valid_items"}
    )
    assert implementation_valid_items.kind is ArgumentKind.STATIC
    assert implementation_valid_items.role is ParameterRole.CONSTANT
    assert plan.participation.uniform_arguments == ("valid_items",)
    assert plan.participation.valid_member_selection.startswith("first N members")
    precondition = plan.participation.argument_preconditions[0]
    assert precondition.name == "valid_items"
    assert (precondition.minimum, precondition.maximum) == (
        1,
        plan.resolved_group.static_size,
    )
    assert precondition.enforcement is PreconditionEnforcement.PLANNER_VALIDATED


def test_partial_logical_warp_reduce_uses_mapped_width_and_storage_instances():
    plan = _plan(
        this_warp().group_by(8),
        _reduce(
            broadcast=False,
            valid_items=ArgumentBinding.static(5),
        ),
        64,
    )

    assert plan.target is GroupLoweringTarget.CUB_WARP
    assert plan.implementation.template_arguments["VIRTUAL_WARP_THREADS"] == 8
    assert plan.implementation.method_name == "Sum"
    assert plan.temp_storage.instances == 8
    assert plan.temp_storage.instance_index == "linear_thread_rank / 8"
    precondition = plan.participation.argument_preconditions[0]
    assert (precondition.minimum, precondition.maximum) == (1, 8)


def test_cub_reduce_cannot_synthesize_broadcast_and_warp_has_no_algorithm_tag():
    broadcast = _plan(
        this_block(),
        _reduce(valid_items=ArgumentBinding.runtime()),
    )
    warp_algorithm = _plan(
        this_warp(),
        _reduce(
            broadcast=False,
            cub_algorithm=BlockReduceAlgorithm.RAKING,
        ),
    )

    assert broadcast.unsupported.code is UnsupportedReasonCode.CUB_BROADCAST
    assert warp_algorithm.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT

    runtime_call = make_group_primitive_call(
        this_block(),
        _reduce(
            broadcast=False,
            valid_items=ArgumentBinding.runtime(),
        ),
    )
    assert runtime_call.argument_classifications[1].kind is ArgumentKind.RUNTIME
    assert runtime_call.argument_classifications[1].role is ParameterRole.INPUT

    with pytest.raises(ValueError, match="unsupported CUB BlockReduce algorithm"):
        _reduce(broadcast=False, cub_algorithm="::cub::BLOCK_SCAN_RAKING")
    with pytest.raises(ValueError, match="positive integer"):
        _reduce(
            broadcast=False,
            valid_items=ArgumentBinding.static(0),
        )
    with pytest.raises(ValueError, match="exceeds group size"):
        _plan(
            this_warp(),
            _reduce(
                broadcast=False,
                valid_items=ArgumentBinding.static(33),
            ),
            64,
        )


def test_group_block_reduce_algorithms_fail_closed_on_unproven_semantics():
    nondeterministic = _plan(
        this_block(),
        _reduce(
            broadcast=False,
            cub_algorithm=BlockReduceAlgorithm.WARP_REDUCTIONS_NONDETERMINISTIC,
        ),
    )
    assert nondeterministic.target is GroupLoweringTarget.UNSUPPORTED
    assert nondeterministic.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT
    assert "addition-specific" in nondeterministic.unsupported.message

    unproven_commutative = _plan(
        this_block(),
        _reduce(
            operation="reduce",
            reduce_operator=CxxOperator("custom_reduce<T>", Dependency("T")),
            broadcast=False,
            cub_algorithm=BlockReduceAlgorithm.RAKING_COMMUTATIVE_ONLY,
        ),
    )
    assert unproven_commutative.target is GroupLoweringTarget.UNSUPPORTED
    assert "proven commutativity" in unproven_commutative.unsupported.message

    proven_commutative = _plan(
        this_block(),
        _reduce(
            operation="max",
            broadcast=False,
            cub_algorithm=BlockReduceAlgorithm.RAKING_COMMUTATIVE_ONLY,
        ),
    )
    assert proven_commutative.target is GroupLoweringTarget.CUB_BLOCK


def test_runtime_valid_items_is_an_explicit_caller_precondition():
    plan = _plan(
        this_block(),
        _reduce(
            broadcast=False,
            valid_items=ArgumentBinding.runtime(),
        ),
        64,
    )

    precondition = plan.participation.argument_preconditions[0]
    assert precondition.name == "valid_items"
    assert (precondition.minimum, precondition.maximum) == (1, 64)
    assert precondition.enforcement is PreconditionEnforcement.CALLER
    precondition.validate(1)
    precondition.validate(64)
    for invalid in (-1, 0, 65):
        with pytest.raises(ValueError, match="valid_items must be"):
            precondition.validate(invalid)


@pytest.mark.parametrize(
    ("group", "valid_items", "target"),
    [
        (this_block(), 128, GroupLoweringTarget.CUB_BLOCK),
        (this_warp(), 32, GroupLoweringTarget.CUB_WARP),
    ],
)
def test_full_size_valid_items_still_selects_root_only_cub(
    group,
    valid_items,
    target,
):
    plan = _plan(
        group,
        _reduce(
            broadcast=False,
            valid_items=ArgumentBinding.static(valid_items),
        ),
        128,
    )

    assert plan.target is target
    assert plan.result.visibility is ResultVisibility.GROUP_ROOT
    assert plan.participation.valid_member_selection == (
        "first N members by linear group rank"
    )
    precondition = plan.participation.argument_preconditions[0]
    assert precondition.maximum == valid_items
    assert precondition.enforcement is PreconditionEnforcement.PLANNER_VALIDATED


def test_stateful_operator_state_is_uniform_and_remains_in_the_runtime_abi():
    stateful = StatefulOperator(
        op=lambda left, right: left + right,
        state_dtype="state",
        ret_dtype="int",
        arg_dtypes=("int", "int"),
    )
    cudax = _plan(
        this_block(),
        _reduce(operation="reduce", reduce_operator=stateful),
        64,
    )
    direct_cub = _plan(
        this_block(),
        _reduce(
            operation="reduce",
            reduce_operator=stateful,
            broadcast=False,
            cub_algorithm=BlockReduceAlgorithm.RAKING,
        ),
        64,
    )
    scan = _plan(
        this_warp(),
        _scan(mode="inclusive", scan_operator=stateful),
        64,
    )

    assert cudax.participation.uniform_arguments == ("operation",)
    assert [parameter.name for parameter in cudax.implementation.parameters] == [
        "item0",
        "operation",
    ]
    assert direct_cub.participation.uniform_arguments == ("operation",)
    assert scan.participation.uniform_arguments == ("operation",)


@pytest.mark.parametrize(
    ("group", "operand_kind", "items", "target", "struct_name"),
    [
        (
            this_block(),
            GroupOperandKind.SCALAR,
            1,
            GroupLoweringTarget.CUB_BLOCK,
            "BlockScan",
        ),
        (
            this_block(),
            GroupOperandKind.ARRAY,
            4,
            GroupLoweringTarget.CUB_BLOCK,
            "BlockScan",
        ),
        (
            this_warp(),
            GroupOperandKind.SCALAR,
            1,
            GroupLoweringTarget.CUB_WARP,
            "WarpScan",
        ),
    ],
)
def test_scan_selects_exact_block_or_scalar_warp_cub(
    group,
    operand_kind,
    items,
    target,
    struct_name,
):
    operation = _scan(
        dtype="int",
        mode=GroupScanMode.EXCLUSIVE,
        operand_kind=operand_kind,
        items_per_thread=items,
        aggregate=True,
    )
    plan = _plan(group, operation, 128)

    assert plan.target is target
    assert plan.implementation.struct_name == struct_name
    assert plan.result.visibility is ResultVisibility.PER_MEMBER
    assert plan.result.operand_kind is operand_kind
    assert plan.result.result_items_per_thread == items
    assert plan.result.has_aggregate
    assert [value.name for value in plan.result.values] == ["value", "aggregate"]
    value, aggregate = plan.result.values
    assert value.dtype == "int"
    assert value.ownership is ResultOwnership.EACH_MEMBER
    assert value.items_per_member == items
    assert aggregate.dtype == "int"
    assert aggregate.visibility is ResultVisibility.ALL_MEMBERS
    assert aggregate.ownership is ResultOwnership.EACH_MEMBER
    assert aggregate.operand_kind is GroupOperandKind.SCALAR
    assert aggregate.items_per_member == 1
    assert aggregate.root_rank is None
    assert plan.provenance.library == "CUB"
    assert plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert plan.temp_storage.cpp_type is None
    assert plan.temp_storage.instances is None
    assert plan.temp_storage.instance_index is None


def test_logical_warp_partial_scan_uses_mapped_width_and_aggregate_contract():
    operation = _scan(
        mode="inclusive",
        aggregate=True,
        valid_items=ArgumentBinding.static(5),
    )
    call = make_group_primitive_call(this_warp().group_by(8), operation)
    plan = plan_group_primitive(call, LaunchFacts(exact_block_dim=64))

    assert plan.target is GroupLoweringTarget.CUB_WARP
    assert plan.implementation.struct_name == "WarpScan"
    assert plan.implementation.method_name == "InclusiveScanPartial"
    assert plan.implementation.template_arguments["VIRTUAL_WARP_THREADS"] == 8
    assert plan.result.has_aggregate
    assert plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert plan.temp_storage.instances is None
    assert plan.participation.uniform_arguments == ("valid_items",)
    precondition = plan.participation.argument_preconditions[0]
    assert (precondition.minimum, precondition.maximum) == (1, 8)
    assert precondition.enforcement is PreconditionEnforcement.PLANNER_VALIDATED
    assert [item.name for item in call.argument_classifications] == [
        "value",
        "mode",
        "valid_items",
    ]


@pytest.mark.parametrize(
    "operation",
    [
        _reduce(
            broadcast=False,
            valid_items=ArgumentBinding.static(5),
        ),
        _scan(mode="inclusive"),
        _exchange(),
        _load_store("load"),
        _load_store("store"),
    ],
)
def test_cub_backed_logical_warp_rejects_non_power_of_two_width(operation):
    group = this_warp().group_by(12, exhaustive=False)
    plan = _plan(group, operation, 64)

    assert plan.target is GroupLoweringTarget.UNSUPPORTED
    assert plan.implementation is None
    assert plan.unsupported.code is UnsupportedReasonCode.GROUP_KIND
    assert "power-of-two group width" in plan.unsupported.message
    assert "got 12" in plan.unsupported.message


@pytest.mark.parametrize(
    ("operation", "template_parameter"),
    [
        (
            _reduce(
                broadcast=False,
                valid_items=ArgumentBinding.static(5),
            ),
            "VIRTUAL_WARP_THREADS",
        ),
        (_scan(mode="inclusive"), "VIRTUAL_WARP_THREADS"),
        (_exchange(), "LOGICAL_WARP_THREADS"),
        (_load_store("load"), "LOGICAL_WARP_THREADS"),
        (_load_store("store"), "LOGICAL_WARP_THREADS"),
    ],
)
def test_physical_warp_cub_plans_use_architectural_width(
    operation,
    template_parameter,
):
    plan = _plan(this_warp(), operation, 64)

    assert plan.target is GroupLoweringTarget.CUB_WARP
    assert plan.implementation.template_arguments[template_parameter] == 32


def test_block_scan_default_algorithm_canonicalizes_in_lowered_artifact():
    default = _plan(this_block(), _scan())
    explicit = _plan(
        this_block(),
        _scan(cub_algorithm=BlockScanAlgorithm.RAKING),
    )

    assert default.call.semantic_key != explicit.call.semantic_key
    assert default.semantic_key == explicit.semantic_key
    assert default.artifact_key == explicit.artifact_key
    assert default == explicit


def test_warp_scan_rejects_multi_item_and_block_only_algorithm_variants():
    multi_item = _plan(
        this_warp(),
        _scan(
            dtype="int",
            mode="inclusive",
            operand_kind="array",
            items_per_thread=2,
        ),
    )
    algorithm = _plan(
        this_warp(),
        _scan(
            dtype="int",
            mode="inclusive",
            operand_kind="scalar",
            items_per_thread=1,
            cub_algorithm="::cub::BLOCK_SCAN_WARP_SCANS",
        ),
    )

    assert multi_item.unsupported.code is UnsupportedReasonCode.OPERAND_FORM
    assert algorithm.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT

    with pytest.raises(ValueError, match="unsupported CUB BlockScan algorithm"):
        _scan(
            dtype="int",
            mode="inclusive",
            operand_kind="scalar",
            items_per_thread=1,
            cub_algorithm="::cub::BLOCK_REDUCE_RAKING",
        )


@pytest.mark.parametrize(
    ("group", "operand_kind", "items_per_thread"),
    [
        (this_block(), "scalar", 1),
        (this_block(), "array", 2),
        (this_warp(), "scalar", 1),
    ],
)
def test_generic_exclusive_scan_without_initial_value_is_unsupported(
    group,
    operand_kind,
    items_per_thread,
):
    plan = _plan(
        group,
        _scan(
            mode="exclusive",
            operand_kind=operand_kind,
            items_per_thread=items_per_thread,
            scan_operator=CxxOperator("::cuda::maximum<>{}", "int"),
        ),
    )

    assert plan.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT
    assert "require an initial value" in plan.unsupported.message
    assert "rank zero" in plan.unsupported.message


@pytest.mark.parametrize(
    ("block_threads", "is_supported"),
    [(16, False), (48, False), (32, True), (64, True)],
)
def test_block_warp_scans_algorithm_requires_complete_warp_multiple(
    block_threads,
    is_supported,
):
    plan = _plan(
        this_block(),
        _scan(cub_algorithm=BlockScanAlgorithm.WARP_SCANS),
        block_threads,
    )

    if is_supported:
        assert plan.target is GroupLoweringTarget.CUB_BLOCK
        assert plan.unsupported is None
    else:
        assert plan.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT
        assert "BLOCK_SCAN_WARP_SCANS" in plan.unsupported.message
        assert "multiple" in plan.unsupported.message


@pytest.mark.parametrize(
    ("group", "mode", "operand_kind", "scan_operator"),
    [
        (this_block(), "exclusive", "scalar", None),
        (this_block(), "inclusive", "array", None),
        (
            this_block(),
            "inclusive",
            "scalar",
            CxxOperator("::cuda::maximum<>{}", "int"),
        ),
        (this_warp(), "exclusive", "scalar", None),
    ],
)
def test_scan_rejects_initial_value_without_an_exact_cub_overload(
    group,
    mode,
    operand_kind,
    scan_operator,
):
    operation = _scan(
        dtype="int",
        mode=mode,
        operand_kind=operand_kind,
        items_per_thread=2 if operand_kind == "array" else 1,
        scan_operator=scan_operator,
        initial_value=CxxFunction("0", "int"),
    )
    plan = _plan(group, operation)

    assert plan.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT


@pytest.mark.parametrize("group", [this_block(), this_warp()])
def test_custom_exclusive_scan_with_initial_value_is_planned(group):
    operation = _scan(
        dtype="int",
        mode="exclusive",
        operand_kind="scalar",
        items_per_thread=1,
        scan_operator=CxxOperator("::cuda::maximum<>{}", "int"),
        initial_value=CxxFunction("0", "int"),
    )
    plan = _plan(group, operation)

    assert plan.target in {
        GroupLoweringTarget.CUB_BLOCK,
        GroupLoweringTarget.CUB_WARP,
    }
    assert plan.participation.uniform_arguments == ("initial_value",)


def test_scan_prefix_callback_is_a_typed_unsupported_variant():
    from cuda.coop._core import PythonOperator

    operation = _scan(
        dtype="int",
        mode="exclusive",
        operand_kind="scalar",
        items_per_thread=1,
        prefix_callback=PythonOperator(
            ret_dtype=Dependency("T"),
            arg_dtypes=(Dependency("T"),),
            op=lambda value: value,
        ),
    )
    plan = _plan(this_block(), operation)

    assert plan.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT


def test_adjacent_difference_selects_exact_multidimensional_block_cub():
    operation = _adjacent_difference(
        dtype="int",
        items_per_thread=3,
        direction=BlockAdjacentDifferenceDirection.LEFT,
        valid_items=17,
        tile_predecessor_item=0,
    )
    plan = _plan(this_block(), operation, (8, 4, 2))

    assert plan.target is GroupLoweringTarget.CUB_BLOCK
    assert plan.implementation.struct_name == "BlockAdjacentDifference"
    assert plan.implementation.method_name == "SubtractLeftPartialTile"
    assert plan.implementation.template_arguments == {
        "T": "int",
        "BLOCK_DIM_X": 8,
        "BLOCK_DIM_Y": 4,
        "BLOCK_DIM_Z": 2,
        "ITEMS_PER_THREAD": 3,
    }
    assert plan.provenance.library == "CUB"
    assert plan.provenance.header == "cub/block/block_adjacent_difference.cuh"
    assert plan.provenance.cpp_class == "cub::BlockAdjacentDifference"
    assert plan.result.visibility is ResultVisibility.PER_MEMBER
    assert plan.result.operand_kind is GroupOperandKind.ARRAY
    assert plan.result.result_items_per_thread == 3
    assert plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert plan.synchronization.storage_reuse_barrier is SynchronizationScope.BLOCK
    assert plan.participation.uniform_arguments == (
        "valid_items",
        "tile_predecessor_item",
    )


def test_adjacent_difference_runtime_payloads_do_not_fragment_artifacts():
    first = _plan(
        this_block(),
        _adjacent_difference(valid_items=17, tile_predecessor_item=1),
        (64, 1, 1),
    )
    second = _plan(
        this_block(),
        _adjacent_difference(valid_items=31, tile_predecessor_item=999),
        (64, 1, 1),
    )

    assert first.semantic_key == second.semantic_key
    assert first.artifact_key == second.artifact_key


def test_adjacent_difference_rejects_non_block_groups_and_missing_exact_shape():
    warp = _plan(this_warp(), _adjacent_difference(), 64)
    missing = _plan(
        this_block(),
        _adjacent_difference(),
        LaunchFacts(max_block_dim=64),
    )

    assert warp.unsupported.code is UnsupportedReasonCode.GROUP_KIND
    assert missing.unsupported.code is UnsupportedReasonCode.MISSING_EXACT_BLOCK_DIM


def test_radix_sort_selects_one_multidimensional_cub_artifact_and_two_results():
    plan = _plan(
        this_block(),
        _radix_sort(value_dtype="double", items_per_thread=3, descending=True),
        (8, 4, 2),
    )

    assert plan.target is GroupLoweringTarget.CUB_BLOCK
    assert plan.implementation.struct_name == "BlockRadixSort"
    assert plan.implementation.method_name == "SortDescending"
    assert plan.implementation.template_arguments["BLOCK_DIM_X"] == 8
    assert plan.implementation.template_arguments["BLOCK_DIM_Y"] == 4
    assert plan.implementation.template_arguments["BLOCK_DIM_Z"] == 2
    assert plan.implementation.template_arguments["ITEMS_PER_THREAD"] == 3
    assert plan.provenance.header == "cub/block/block_radix_sort.cuh"
    assert plan.provenance.cpp_class == "cub::BlockRadixSort"
    assert [result.name for result in plan.result.values] == ["keys", "values"]
    assert all(
        result.visibility is ResultVisibility.PER_MEMBER
        for result in plan.result.values
    )
    assert plan.participation.uniform_arguments == ("begin_bit", "end_bit")
    assert plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert plan.synchronization.storage_reuse_barrier is SynchronizationScope.BLOCK


def test_radix_sort_runtime_bit_values_do_not_enter_artifact_identity():
    operation = _radix_sort(items_per_thread=2)
    first = _plan(this_block(), operation, (64, 1, 1))
    second = _plan(this_block(), operation, (64, 1, 1))

    assert first.artifact_key == second.artifact_key
    assert first.call.argument_classifications[1].kind.name == "RUNTIME"
    assert operation.primitive.bit_range.begin_bit.value is None
    assert operation.primitive.bit_range.end_bit.value is None


def test_radix_rank_static_width_owns_prefix_result_and_public_cub_plan():
    plan = _plan(
        this_block(),
        _radix_rank(prefix_items=4, items_per_thread=2),
        (64, 1, 1),
    )

    assert plan.target is GroupLoweringTarget.CUB_BLOCK
    assert plan.implementation.struct_name == "BlockRadixRank"
    assert plan.implementation.method_name == "RankKeys"
    assert plan.implementation.template_arguments["RADIX_BITS"] == 8
    assert plan.implementation.template_arguments["BLOCK_DIM_X"] == 64
    assert plan.provenance.header == "cub/block/block_radix_rank.cuh"
    assert plan.provenance.cpp_class == "cub::BlockRadixRank"
    assert [result.name for result in plan.result.values] == [
        "ranks",
        "exclusive_digit_prefix",
    ]
    assert plan.result.values[1].items_per_member == 4

    wide_plan = _plan(
        this_block(),
        _radix_rank(end_bit=12, prefix_items=64, items_per_thread=2),
        (64, 1, 1),
    )
    assert wide_plan.target is GroupLoweringTarget.CUB_BLOCK
    assert wide_plan.implementation.template_arguments["RADIX_BITS"] == 12
    assert wide_plan.result.values[1].items_per_member == 64


def test_radix_plans_reject_wrong_group_missing_exact_shape_and_oversized_tile():
    wrong_group = _plan(this_warp(), _radix_sort(), 64)
    missing = _plan(
        this_block(),
        _radix_rank(block_threads=64),
        LaunchFacts(max_block_dim=64),
    )
    oversized = _plan(
        this_block(),
        _radix_sort(items_per_thread=2),
        (32768, 1, 1),
    )

    assert wrong_group.unsupported.code is UnsupportedReasonCode.GROUP_KIND
    assert missing.unsupported.code is UnsupportedReasonCode.MISSING_EXACT_BLOCK_DIM
    assert oversized.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT
    assert "<= 65535" in oversized.unsupported.message


@pytest.mark.parametrize(
    ("mode", "predecessor", "successor", "result_names"),
    [
        (BlockDiscontinuityMode.HEADS, None, None, ("head_flags",)),
        (BlockDiscontinuityMode.HEADS, 1, None, ("head_flags",)),
        (BlockDiscontinuityMode.TAILS, None, None, ("tail_flags",)),
        (BlockDiscontinuityMode.TAILS, None, 9, ("tail_flags",)),
        (
            BlockDiscontinuityMode.HEADS_AND_TAILS,
            None,
            None,
            ("head_flags", "tail_flags"),
        ),
        (
            BlockDiscontinuityMode.HEADS_AND_TAILS,
            1,
            None,
            ("head_flags", "tail_flags"),
        ),
        (
            BlockDiscontinuityMode.HEADS_AND_TAILS,
            None,
            9,
            ("head_flags", "tail_flags"),
        ),
        (
            BlockDiscontinuityMode.HEADS_AND_TAILS,
            1,
            9,
            ("head_flags", "tail_flags"),
        ),
    ],
)
def test_discontinuity_plans_every_public_cub_boundary_overload(
    mode,
    predecessor,
    successor,
    result_names,
):
    operation = _discontinuity(
        items_per_thread=3,
        mode=mode,
        tile_predecessor_item=predecessor,
        tile_successor_item=successor,
    )
    call = make_group_primitive_call(this_block(), operation)
    plan = plan_group_primitive(call, LaunchFacts(exact_block_dim=(8, 4, 2)))

    assert plan.target is GroupLoweringTarget.CUB_BLOCK
    assert plan.implementation.struct_name == "BlockDiscontinuity"
    assert plan.implementation.method_name == mode.cub_method_name
    assert plan.implementation.template_arguments["BLOCK_DIM_X"] == 8
    assert plan.implementation.template_arguments["BLOCK_DIM_Y"] == 4
    assert plan.implementation.template_arguments["BLOCK_DIM_Z"] == 2
    assert plan.provenance.header == "cub/block/block_discontinuity.cuh"
    assert tuple(result.name for result in plan.result.values) == result_names
    assert all(
        result.dtype == "flag" and result.items_per_member == 3
        for result in plan.result.values
    )
    expected_uniform = (
        *(("tile_predecessor_item",) if predecessor is not None else ()),
        *(("tile_successor_item",) if successor is not None else ()),
    )
    assert plan.participation.uniform_arguments == expected_uniform
    assert plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert plan.synchronization.storage_reuse_barrier is SynchronizationScope.BLOCK


def test_discontinuity_boundary_values_do_not_fragment_artifacts():
    first = _plan(
        this_block(),
        _discontinuity(
            mode="heads_and_tails",
            tile_predecessor_item=1,
            tile_successor_item=9,
        ),
        64,
    )
    second = _plan(
        this_block(),
        _discontinuity(
            mode="heads_and_tails",
            tile_predecessor_item=111,
            tile_successor_item=999,
        ),
        64,
    )

    assert first.artifact_key == second.artifact_key


@pytest.mark.parametrize(
    ("operation", "method", "result_kind", "result_names"),
    [
        (_shuffle(mode="offset"), "Offset", GroupOperandKind.SCALAR, ("value",)),
        (_shuffle(mode="rotate"), "Rotate", GroupOperandKind.SCALAR, ("value",)),
        (
            _shuffle(items_per_thread=3, mode="up"),
            "Up",
            GroupOperandKind.ARRAY,
            ("value",),
        ),
        (
            _shuffle(items_per_thread=3, mode="up", block_suffix=True),
            "Up",
            GroupOperandKind.ARRAY,
            ("value", "block_suffix"),
        ),
        (
            _shuffle(items_per_thread=3, mode="down", block_prefix=True),
            "Down",
            GroupOperandKind.ARRAY,
            ("value", "block_prefix"),
        ),
    ],
)
def test_shuffle_plans_only_public_cub_shapes(
    operation,
    method,
    result_kind,
    result_names,
):
    plan = _plan(this_block(), operation, (8, 4, 2))

    assert plan.target is GroupLoweringTarget.CUB_BLOCK
    assert plan.implementation.struct_name == "BlockShuffle"
    assert plan.implementation.method_name == method
    assert plan.implementation.template_arguments["BLOCK_DIM_Y"] == 4
    assert plan.implementation.template_arguments["BLOCK_DIM_Z"] == 2
    assert plan.provenance.header == "cub/block/block_shuffle.cuh"
    assert plan.result.primary.operand_kind is result_kind
    assert tuple(result.name for result in plan.result.values) == result_names
    assert plan.synchronization.storage_reuse_barrier is SynchronizationScope.BLOCK


def test_shuffle_scalar_distance_is_runtime_and_outside_artifact_identity():
    first = _plan(this_block(), _shuffle(mode="rotate"), 64)
    second = _plan(this_block(), _shuffle(mode="rotate"), 64)

    assert first.participation.uniform_arguments == ("distance",)
    assert first.artifact_key == second.artifact_key
    assert first.call.argument_classifications[-1].name == "distance"
    assert first.call.argument_classifications[-1].kind is ArgumentKind.RUNTIME


def test_shuffle_planner_rejects_non_cub_shapes_and_missing_exact_shape():
    scalar_up = _plan(this_block(), _shuffle(mode="up"), 64)
    array_rotate = _plan(
        this_block(),
        _shuffle(
            items_per_thread=2,
            mode="rotate",
            distance=ArgumentBinding.omitted(),
        ),
        64,
    )
    missing = _plan(
        this_block(),
        _shuffle(mode="offset"),
        LaunchFacts(max_block_dim=64),
    )

    assert scalar_up.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT
    assert array_rotate.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT
    assert missing.unsupported.code is UnsupportedReasonCode.MISSING_EXACT_BLOCK_DIM


def test_block_merge_sort_plans_one_multidimensional_public_cub_artifact():
    plan = _plan(
        this_block(),
        _merge_sort(
            key_dtype="int",
            value_dtype="float",
            items_per_thread=3,
            descending=True,
            valid_items=17,
            oob_default=-1,
        ),
        (8, 4, 2),
    )

    assert plan.target is GroupLoweringTarget.CUB_BLOCK
    assert plan.implementation.struct_name == "BlockMergeSort"
    assert plan.implementation.method_name == "Sort"
    assert plan.implementation.template_arguments == {
        "KeyT": "int",
        "BLOCK_DIM_X": 8,
        "ITEMS_PER_THREAD": 3,
        "ValueT": "float",
        "BLOCK_DIM_Y": 4,
        "BLOCK_DIM_Z": 2,
    }
    assert plan.provenance.header == "cub/block/block_merge_sort.cuh"
    assert plan.provenance.cpp_class == "cub::BlockMergeSort"
    assert [result.name for result in plan.result.values] == ["keys", "values"]
    assert [result.dtype for result in plan.result.values] == ["int", "float"]
    assert all(result.items_per_member == 3 for result in plan.result.values)
    assert plan.participation.uniform_arguments == (
        "valid_items",
        "oob_default",
    )
    assert plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert plan.synchronization.storage_reuse_barrier is SynchronizationScope.BLOCK


def test_warp_merge_sort_plans_physical_and_logical_storage_partitions():
    physical = _plan(this_warp(), _merge_sort(items_per_thread=2), 64)
    logical = _plan(
        this_warp().group_by(16),
        _merge_sort(value_dtype="float", items_per_thread=2),
        64,
    )
    physical_partial = _plan(
        this_warp(),
        _merge_sort(
            value_dtype="float",
            items_per_thread=2,
            valid_items=59,
            oob_default=999,
        ),
        64,
    )

    assert physical.target is GroupLoweringTarget.CUB_WARP
    assert physical.implementation.struct_name == "WarpMergeSort"
    assert physical.implementation.template_arguments["VIRTUAL_WARP_THREADS"] == 32
    assert physical.provenance.header == "cub/warp/warp_merge_sort.cuh"
    assert physical.synchronization.storage_reuse_barrier is SynchronizationScope.WARP
    assert logical.target is GroupLoweringTarget.CUB_WARP
    assert logical.implementation.template_arguments["VIRTUAL_WARP_THREADS"] == 16
    assert [result.name for result in logical.result.values] == ["keys", "values"]
    assert physical_partial.target is GroupLoweringTarget.CUB_WARP
    assert [
        parameter.name for parameter in physical_partial.implementation.parameters[0]
    ] == [
        "temp_storage",
        "keys",
        "values",
        "compare_op",
        "valid_items",
        "oob_default",
    ]
    assert physical_partial.participation.uniform_arguments == (
        "valid_items",
        "oob_default",
    )
    precondition = physical_partial.participation.argument_preconditions[0]
    assert (precondition.minimum, precondition.maximum) == (0, 64)
    precondition.validate(0)
    precondition.validate(64)
    for invalid in (-1, 65):
        with pytest.raises(ValueError, match="valid_items must be"):
            precondition.validate(invalid)


def test_merge_sort_runtime_values_do_not_fragment_artifacts():
    first = _plan(
        this_block(),
        _merge_sort(valid_items=17, oob_default=-1),
        64,
    )
    second = _plan(
        this_block(),
        _merge_sort(valid_items=31, oob_default=999),
        64,
    )

    assert first.artifact_key == second.artifact_key

    warp_first = _plan(
        this_warp(),
        _merge_sort(valid_items=17, oob_default=-1),
        64,
    )
    warp_second = _plan(
        this_warp(),
        _merge_sort(valid_items=31, oob_default=999),
        64,
    )
    assert warp_first.artifact_key == warp_second.artifact_key


def test_merge_sort_rejects_non_power_of_two_blocks_and_incomplete_warps():
    wrong_group = _plan(this_thread(), _merge_sort(), 1)
    missing_exact = _plan(
        this_block(),
        _merge_sort(),
        LaunchFacts(max_block_dim=64),
    )
    block = _plan(this_block(), _merge_sort(), 48)
    physical_warp = _plan(this_warp(), _merge_sort(), 48)
    logical_warp = _plan(this_warp().group_by(16), _merge_sort(), 24)

    assert wrong_group.unsupported.code is UnsupportedReasonCode.GROUP_KIND
    assert (
        missing_exact.unsupported.code is UnsupportedReasonCode.MISSING_EXACT_BLOCK_DIM
    )
    assert block.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT
    assert physical_warp.unsupported.code is UnsupportedReasonCode.PARTIAL_PHYSICAL_WARP
    assert logical_warp.unsupported.code is UnsupportedReasonCode.PARTIAL_PHYSICAL_WARP


@pytest.mark.parametrize(
    ("group", "target", "struct_name", "template_arguments", "barrier"),
    [
        (
            this_block(),
            GroupLoweringTarget.CUB_BLOCK,
            "BlockExchange",
            {
                "T": "int",
                "BLOCK_DIM_X": 128,
                "ITEMS_PER_THREAD": 3,
                "WARP_TIME_SLICING": 0,
                "BLOCK_DIM_Y": 1,
                "BLOCK_DIM_Z": 1,
            },
            SynchronizationScope.BLOCK,
        ),
        (
            this_warp(),
            GroupLoweringTarget.CUB_WARP,
            "WarpExchange",
            {
                "T": "int",
                "ITEMS_PER_THREAD": 3,
                "LOGICAL_WARP_THREADS": 32,
                "WARP_EXCHANGE_ALGORITHM": "::cub::WARP_EXCHANGE_SMEM",
            },
            SynchronizationScope.WARP,
        ),
    ],
)
@pytest.mark.parametrize(
    ("mode", "method_name"),
    [
        (GroupExchangeMode.STRIPED_TO_BLOCKED, "StripedToBlocked"),
        (GroupExchangeMode.BLOCKED_TO_STRIPED, "BlockedToStriped"),
    ],
)
def test_exchange_selects_exact_array_cub_with_implementation_storage(
    group,
    target,
    struct_name,
    template_arguments,
    barrier,
    mode,
    method_name,
):
    operation = _exchange(mode.value, 3)
    plan = _plan(group, operation, 128)

    assert plan.target is target
    assert plan.implementation.struct_name == struct_name
    assert plan.implementation.method_name == method_name
    assert plan.implementation.template_arguments == template_arguments
    assert plan.result.operand_kind is GroupOperandKind.ARRAY
    assert plan.result.result_items_per_thread == 3
    assert plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert plan.temp_storage.address_space is None
    assert plan.temp_storage.cpp_type is None
    assert plan.temp_storage.instances is None
    assert plan.temp_storage.instance_index is None
    assert not plan.temp_storage.exact_layout_required
    assert plan.synchronization.storage_reuse_barrier is barrier


def test_exchange_requires_exact_launch_and_complete_physical_warps():
    missing_exact = _plan(
        this_block(),
        _exchange(),
        LaunchFacts(max_block_dim=128),
    )
    partial_warp = _plan(this_warp(), _exchange(), 48)

    assert (
        missing_exact.unsupported.code is UnsupportedReasonCode.MISSING_EXACT_BLOCK_DIM
    )
    assert "upper bound" in missing_exact.unsupported.message
    assert partial_warp.unsupported.code is UnsupportedReasonCode.PARTIAL_PHYSICAL_WARP
    assert "complete 32-thread warps" in partial_warp.unsupported.message


@pytest.mark.parametrize("group", [this_block(), this_warp()])
def test_group_exchange_leaves_frontend_item_qualification_out_of_core(group):
    plan = _plan(group, _exchange(items_per_thread=6), 64)

    assert plan.target in {
        GroupLoweringTarget.CUB_BLOCK,
        GroupLoweringTarget.CUB_WARP,
    }
    assert plan.result.result_items_per_thread == 6


@pytest.mark.parametrize(
    ("mode", "method_name", "rank_dtype", "valid_flag_dtype"),
    [
        ("warp_striped_to_blocked", "WarpStripedToBlocked", None, None),
        ("blocked_to_warp_striped", "BlockedToWarpStriped", None, None),
        ("scatter_to_blocked", "ScatterToBlocked", "int", None),
        ("scatter_to_striped", "ScatterToStriped", "int", None),
        ("scatter_to_striped_guarded", "ScatterToStripedGuarded", "int", None),
        (
            "scatter_to_striped_flagged",
            "ScatterToStripedFlagged",
            "int",
            "unsigned char",
        ),
    ],
)
def test_group_exchange_plans_every_qualified_block_mode(
    mode,
    method_name,
    rank_dtype,
    valid_flag_dtype,
):
    operation = _exchange(
        mode,
        2,
        rank_dtype=rank_dtype,
        valid_flag_dtype=valid_flag_dtype,
        warp_time_slicing=True,
    )
    call = make_group_primitive_call(this_block(), operation)
    plan = plan_group_primitive(call, LaunchFacts(exact_block_dim=64))

    assert plan.target is GroupLoweringTarget.CUB_BLOCK
    assert plan.implementation.method_name == method_name
    assert plan.implementation.template_arguments["WARP_TIME_SLICING"] == 1
    assert [item.name for item in call.argument_classifications] == [
        "value",
        "mode",
        *(("ranks",) if rank_dtype is not None else ()),
        *(("valid_flags",) if valid_flag_dtype is not None else ()),
    ]


def test_logical_warp_exchange_uses_mapped_width_and_rejects_block_only_modes():
    group = this_warp().group_by(8)
    plan = _plan(
        group,
        _exchange("scatter_to_striped", 2, rank_dtype="int"),
        64,
    )
    block_only = _plan(group, _exchange("warp_striped_to_blocked", 2), 64)

    assert plan.target is GroupLoweringTarget.CUB_WARP
    assert plan.implementation.method_name == "ScatterToStriped"
    assert plan.implementation.template_arguments["LOGICAL_WARP_THREADS"] == 8
    assert plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert plan.temp_storage.instances is None
    assert block_only.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT


def test_reduce_plans_every_static_cudax_group_form():
    cluster_facts = LaunchFacts(
        exact_block_dim=64,
        exact_cluster_dim=2,
        cluster_launch=True,
        provenance=(
            LaunchFactOrigin("exact_cluster_dim", "launch_config", verified=True),
            LaunchFactOrigin("cluster_launch", "launch_config", verified=True),
        ),
    )
    grid_facts = LaunchFacts(
        exact_block_dim=64,
        exact_grid_dim=8,
        exact_cluster_dim=2,
        cluster_launch=True,
        cooperative_launch=True,
        provenance=(
            LaunchFactOrigin("exact_grid_dim", "launch_config", verified=True),
            LaunchFactOrigin("exact_cluster_dim", "launch_config", verified=True),
            LaunchFactOrigin("cluster_launch", "launch_config", verified=True),
            LaunchFactOrigin("cooperative_launch", "launch_config", verified=True),
        ),
    )
    cases = (
        (this_thread(), LaunchFacts()),
        (this_warp(), LaunchFacts(exact_block_dim=64)),
        (this_block(), LaunchFacts(exact_block_dim=64)),
        (
            this_cluster(),
            LaunchFacts(
                exact_block_dim=64,
                cluster_launch=False,
                provenance=(
                    LaunchFactOrigin(
                        "cluster_launch",
                        "launch_config",
                        verified=True,
                    ),
                ),
            ),
        ),
        (this_cluster(), cluster_facts),
        (this_grid(), grid_facts),
        (
            this_warp().group_by(12, exhaustive=False),
            LaunchFacts(exact_block_dim=64),
        ),
        (
            this_block().group_by(3, exhaustive=False),
            LaunchFacts(exact_block_dim=320),
        ),
    )

    plans = [_plan(group, _reduce(), facts) for group, facts in cases]
    assert all(plan.target is GroupLoweringTarget.CUDAX_GROUP for plan in plans)
    assert plans[0].participation.exact_block_dim is None
    assert plans[3].resolved_group.hierarchy.cluster_dim == (1, 1, 1)
    assert plans[4].resolved_group.static_size == 128
    assert plans[5].resolved_group.hierarchy.grid_dim == (4, 1, 1)
    assert plans[5].resolved_group.static_size == 512
    assert plans[6].participation.complete_membership is False
    assert plans[7].resolved_group.groups_per_parent == 3


def test_cluster_resolution_rejects_unknown_launch_state():
    plan = _plan(
        this_cluster(),
        _reduce(),
        LaunchFacts(exact_block_dim=64),
    )

    assert plan.target is GroupLoweringTarget.UNSUPPORTED
    assert plan.unsupported.code is UnsupportedReasonCode.LAUNCH_CAPABILITY
    assert "verified non-cluster launch" in plan.unsupported.message


@pytest.mark.parametrize(
    ("group", "facts"),
    [
        (
            this_cluster(),
            LaunchFacts(
                exact_block_dim=64,
                exact_cluster_dim=2,
                cluster_launch=True,
            ),
        ),
        (
            this_grid(),
            LaunchFacts(
                exact_block_dim=64,
                exact_grid_dim=8,
                cooperative_launch=True,
            ),
        ),
    ],
)
def test_cluster_and_grid_require_backend_verified_launch_capabilities(group, facts):
    plan = _plan(group, _reduce(), facts)

    assert plan.target is GroupLoweringTarget.UNSUPPORTED
    assert plan.unsupported.code is UnsupportedReasonCode.LAUNCH_CAPABILITY
    assert "verified" in plan.unsupported.message


@pytest.mark.parametrize("group", [this_cluster(), this_grid()])
def test_cluster_launch_with_runtime_selected_shape_is_not_specialized(group):
    facts = LaunchFacts(
        exact_block_dim=64,
        exact_grid_dim=8 if group.kind == "grid" else None,
        cluster_launch=True,
        cooperative_launch=True if group.kind == "grid" else None,
        provenance=(
            LaunchFactOrigin("cluster_launch", "launch_config", verified=True),
            *(
                (
                    LaunchFactOrigin(
                        "cooperative_launch",
                        "launch_config",
                        verified=True,
                    ),
                )
                if group.kind == "grid"
                else ()
            ),
        ),
    )

    plan = _plan(group, _reduce(), facts)

    assert plan.target is GroupLoweringTarget.UNSUPPORTED
    assert plan.unsupported.code is UnsupportedReasonCode.LAUNCH_CAPABILITY
    assert "exact static cluster dimensions" in plan.unsupported.message


def test_group_and_launch_markers_are_erased_from_runtime_abi():
    operation = _exchange("striped_to_blocked", 2)
    call = make_group_primitive_call(
        this_block(),
        operation,
        source="root_frontend",
    )
    plan = plan_group_primitive(call, LaunchFacts(exact_block_dim=64))

    assert all(
        classification.name not in {"group", "launch", "launch_facts"}
        for classification in call.argument_classifications
    )
    assert all(
        not isinstance(parameter, (ThreadGroup, LaunchFacts))
        for method in plan.implementation.parameters
        for parameter in method
    )

    cudax_plan = _plan(this_warp(), _reduce(broadcast=False), 64)
    assert cudax_plan.implementation.return_kind is CudaxReturnKind.OPTIONAL_VALUE
    assert all(
        parameter.name not in {"group", "launch", "launch_facts"}
        for parameter in cudax_plan.implementation.parameters
    )


@pytest.mark.parametrize(
    ("group", "kind", "target", "cpp_class"),
    [
        (this_block(), "load", GroupLoweringTarget.CUB_BLOCK, "cub::BlockLoad"),
        (this_block(), "store", GroupLoweringTarget.CUB_BLOCK, "cub::BlockStore"),
        (this_warp(), "load", GroupLoweringTarget.CUB_WARP, "cub::WarpLoad"),
        (this_warp(), "store", GroupLoweringTarget.CUB_WARP, "cub::WarpStore"),
    ],
)
def test_group_load_store_selects_real_cub(group, kind, target, cpp_class):
    plan = _plan(group, _load_store(kind, items_per_thread=3), 64)

    assert plan.target is target
    assert plan.provenance.library == "CUB"
    assert plan.provenance.cpp_class == cpp_class
    if kind == "load":
        assert plan.result.result_items_per_thread == 3
    else:
        assert plan.result is None


def test_group_load_models_partial_tile_and_offset_bindings():
    operation = _load_store(
        valid_items=ArgumentBinding.runtime(),
        oob_default=ArgumentBinding.static(0),
        offset=ArgumentBinding.static(4),
    )
    call = make_group_primitive_call(this_block(), operation)
    plan = plan_group_primitive(call, LaunchFacts(exact_block_dim=64))

    assert plan.target is GroupLoweringTarget.CUB_BLOCK
    assert (
        plan.participation.valid_member_selection == "first valid_items tile elements"
    )
    assert plan.participation.uniform_arguments == (
        "valid_items",
        "oob_default",
        "offset",
    )
    assert [
        classification.name for classification in call.argument_classifications
    ] == [
        "source",
        "valid_items",
        "oob_default",
        "offset",
        "algorithm",
    ]


def test_group_load_store_supports_logical_warps_and_rejects_invalid_algorithms():
    mapped = this_warp().group_by(8)
    mapped_plan = _plan(mapped, _load_store(), 64)
    warp_plan = _plan(
        this_warp(),
        _load_store(algorithm=GroupLoadStoreAlgorithm.WARP_TRANSPOSE),
        64,
    )

    assert mapped_plan.target is GroupLoweringTarget.CUB_WARP
    assert mapped_plan.implementation.template_arguments["LOGICAL_WARP_THREADS"] == 8
    assert mapped_plan.temp_storage.ownership is StorageOwnership.IMPLEMENTATION
    assert mapped_plan.temp_storage.instances is None
    assert warp_plan.unsupported.code is UnsupportedReasonCode.OPERATION_VARIANT


def test_operation_semantics_participate_in_artifact_identity():
    sum_plan = _plan(this_block(), _reduce(operation="sum"), 64)
    max_plan = _plan(this_block(), _reduce(operation="max"), 64)

    assert sum_plan.artifact_key != max_plan.artifact_key
    assert sum_plan != max_plan


def test_explicit_cub_algorithm_participates_in_call_and_plan_identity():
    primitive = make_block_reduce_semantics(
        dtype="int",
        operation="sum",
        value_kind="scalar",
        items_per_thread=1,
    )
    raking_call = make_group_primitive_call(
        this_block(),
        GroupReduceSemantics(
            primitive,
            broadcast=False,
            cub_algorithm=BlockReduceAlgorithm.RAKING,
        ),
    )
    warp_call = make_group_primitive_call(
        this_block(),
        GroupReduceSemantics(
            primitive,
            broadcast=False,
            cub_algorithm=BlockReduceAlgorithm.WARP_REDUCTIONS,
        ),
    )
    raking_plan = plan_group_primitive(raking_call, LaunchFacts(exact_block_dim=64))
    warp_plan = plan_group_primitive(warp_call, LaunchFacts(exact_block_dim=64))

    assert raking_call.semantic_key != warp_call.semantic_key
    assert raking_plan.semantic_key != warp_plan.semantic_key
    assert raking_plan.artifact_key != warp_plan.artifact_key


def test_omitted_block_cub_algorithm_canonicalizes_to_warp_reductions():
    primitive = make_block_reduce_semantics(
        dtype="int",
        operation="sum",
        value_kind="scalar",
        items_per_thread=1,
        valid_items=ArgumentBinding.static(17),
    )
    omitted = plan_group_primitive(
        make_group_primitive_call(
            this_block(),
            GroupReduceSemantics(primitive, broadcast=False),
        ),
        LaunchFacts(exact_block_dim=64),
    )
    explicit = plan_group_primitive(
        make_group_primitive_call(
            this_block(),
            GroupReduceSemantics(
                primitive,
                broadcast=False,
                cub_algorithm=BlockReduceAlgorithm.WARP_REDUCTIONS,
            ),
        ),
        LaunchFacts(exact_block_dim=64),
    )

    assert omitted.call == explicit.call
    assert omitted.semantic_key == explicit.semantic_key
    assert omitted.artifact_key == explicit.artifact_key
    assert omitted == explicit
