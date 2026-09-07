# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402, F401

"""Shared constructors for portable group-planner contract tests."""

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
    GroupHistogramSemantics,
    GroupLoadStoreAlgorithm,
    GroupLoadStoreKind,
    GroupLoadStoreSemantics,
    GroupLoweringTarget,
    GroupMergeSortSemantics,
    GroupOperandKind,
    GroupRadixRankSemantics,
    GroupRadixSortSemantics,
    GroupReduceSemantics,
    GroupRunLengthDecodeSemantics,
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
    make_block_histogram_semantics,
    make_block_merge_sort_semantics,
    make_block_radix_rank_semantics,
    make_block_radix_sort_semantics,
    make_block_reduce_semantics,
    make_block_run_length_decode_semantics,
    make_block_scan_spec,
    make_block_shuffle_semantics,
    make_block_topk_spec,
)
from cuda.coop._core.group import GroupTopKSemantics
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


def _histogram(**overrides):
    bins_per_thread = overrides.pop("bins_per_thread", 1)
    primitive = make_block_histogram_semantics(
        item_dtype=overrides.pop("item_dtype", "int"),
        counter_dtype=overrides.pop("counter_dtype", "unsigned int"),
        items_per_thread=overrides.pop("items_per_thread", 2),
        bins=overrides.pop("bins", 64),
        algorithm=overrides.pop("algorithm", "atomic"),
    )
    assert not overrides
    return GroupHistogramSemantics(
        primitive=primitive,
        bins_per_thread=bins_per_thread,
    )


def _run_length_decode(**overrides):
    primitive = make_block_run_length_decode_semantics(
        item_dtype=overrides.pop("item_dtype", "int"),
        run_length_dtype=overrides.pop("run_length_dtype", "unsigned int"),
        decoded_offset_dtype=overrides.pop(
            "decoded_offset_dtype",
            "unsigned int",
        ),
        total_decoded_size_dtype=overrides.pop(
            "total_decoded_size_dtype",
            "unsigned int",
        ),
        runs_per_thread=overrides.pop("runs_per_thread", 1),
        decoded_items_per_thread=overrides.pop("decoded_items_per_thread", 2),
        with_relative_offsets=overrides.pop("with_relative_offsets", True),
        relative_offset_dtype=overrides.pop(
            "relative_offset_dtype",
            "unsigned int",
        ),
        with_decoded_window_offset=overrides.pop(
            "with_decoded_window_offset",
            True,
        ),
        returns_total_decoded_size=True,
    )
    assert not overrides
    return GroupRunLengthDecodeSemantics(primitive)


def _topk(**overrides):
    primitive = make_block_topk_spec(
        key_dtype=overrides.pop("key_dtype", "unsigned int"),
        value_dtype=overrides.pop("value_dtype", None),
        block_dim=overrides.pop("block_dim", (64, 1, 1)),
        items_per_thread=overrides.pop("items_per_thread", 2),
        selection=overrides.pop("selection", "max"),
        num_valid=overrides.pop("num_valid", ArgumentBinding.runtime()),
        begin_bit=overrides.pop("begin_bit", ArgumentBinding.runtime()),
        end_bit=overrides.pop("end_bit", ArgumentBinding.runtime()),
    )
    assert not overrides
    return GroupTopKSemantics(primitive)


def _plan(group, operation, launch=(64, 1, 1)):
    facts = launch if isinstance(launch, LaunchFacts) else LaunchFacts(launch)
    return plan_group_primitive(
        make_group_primitive_call(group, operation),
        facts,
    )


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
