# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral planning for compile-time CUDA thread-group calls."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from ._algorithm import AlgorithmSpec
from ._bindings import ArgumentBinding, BindingKind
from ._symbols import semantic_token
from ._types import (
    INT32,
    ArgumentKind,
    CxxFunction,
    CxxOperator,
    ParameterClassification,
    ParameterRole,
    PythonOperator,
    Reference,
    RuntimeValue,
    StatefulOperator,
    classify_parameter,
)
from .block.adjacent_difference import (
    BlockAdjacentDifferenceBoundary,
    BlockAdjacentDifferenceDirection,
    BlockAdjacentDifferenceSemantics,
    make_block_adjacent_difference_spec,
)
from .block.discontinuity import (
    BlockDiscontinuityMode,
    BlockDiscontinuitySemantics,
    make_block_discontinuity_spec,
)
from .block.exchange import (
    BlockExchangeMode,
    BlockExchangeSemantics,
    BlockExchangeValueForm,
    make_block_exchange_spec,
)
from .block.histogram import (
    BlockHistogramOperation,
    BlockHistogramSemantics,
    make_block_histogram_spec,
    validate_block_histogram_output_capacity,
)
from .block.load_store import (
    BlockLoadStoreAlgorithm,
    make_block_load_spec,
    make_block_store_spec,
)
from .block.merge_sort import (
    BlockMergeSortSemantics,
    make_block_merge_sort_spec,
)
from .block.radix_rank import (
    BlockRadixRankSemantics,
    make_block_radix_rank_spec,
)
from .block.radix_sort import (
    BlockRadixSortBitPolicy,
    BlockRadixSortOutput,
    BlockRadixSortSemantics,
    make_block_radix_sort_spec,
)
from .block.reduce import (
    BlockReduceAlgorithm,
    make_block_reduce_spec,
    normalize_block_reduce_algorithm,
)
from .block.run_length import (
    BlockRunLengthDecodeSemantics,
    BlockRunLengthDecodeStage,
    make_block_run_length_decode_spec,
)
from .block.scan import (
    BlockScanAlgorithm,
    ScanMode,
    ScanValueKind,
    make_block_scan_spec,
    normalize_block_scan_algorithm,
)
from .block.shuffle import (
    BlockShuffleMode,
    BlockShuffleSemantics,
    BlockShuffleValueKind,
    make_block_shuffle_spec,
)
from .launch import Dim3, LaunchFacts
from .reduce import ReduceOperation, ReduceSemantics, ReduceValueKind
from .scan import ScanSemantics
from .thread_group import (
    COMPLETE_WARP_GROUP_KINDS,
    MAPPED_GROUP_KINDS,
    ThreadGroup,
    ThreadHierarchy,
    normalize_thread_level,
)
from .warp.exchange import (
    WarpExchangeMode,
    WarpExchangeValueForm,
    make_warp_exchange_spec,
)
from .warp.load_store import (
    WarpLoadStoreAlgorithm,
    make_warp_load_spec,
    make_warp_store_spec,
)
from .warp.merge_sort import make_warp_merge_sort_spec
from .warp.reduce import WarpReduceOperation, make_warp_reduce_spec
from .warp.scan import WarpScanMode, make_warp_scan_spec


class GroupLoweringTarget(str, Enum):
    CUDAX_GROUP = "cudax_group"
    CUB_BLOCK = "cub_block"
    CUB_WARP = "cub_warp"
    UNSUPPORTED = "unsupported"


class GroupOperandKind(str, Enum):
    SCALAR = "scalar"
    ARRAY = "array"


GroupScanMode = ScanMode
GroupExchangeMode = BlockExchangeMode


class GroupLoadStoreKind(str, Enum):
    LOAD = "load"
    STORE = "store"


class GroupLoadStoreAlgorithm(str, Enum):
    DIRECT = "direct"
    STRIPED = "striped"
    VECTORIZE = "vectorize"
    TRANSPOSE = "transpose"
    WARP_TRANSPOSE = "warp_transpose"
    WARP_TRANSPOSE_TIMESLICED = "warp_transpose_timesliced"


class ResultVisibility(str, Enum):
    ALL_MEMBERS = "all_members"
    GROUP_ROOT = "group_root"
    PER_MEMBER = "per_member"


class ResultOwnership(str, Enum):
    EACH_MEMBER = "each_member"
    GROUP_ROOT = "group_root"


class PreconditionEnforcement(str, Enum):
    PLANNER_VALIDATED = "planner_validated"
    CALLER = "caller"


class StorageOwnership(str, Enum):
    IMPLEMENTATION = "implementation"
    CALLER = "caller"


class SynchronizationScope(str, Enum):
    NONE = "none"
    WARP = "warp"
    BLOCK = "block"
    GROUP = "group"


class UnsupportedReasonCode(str, Enum):
    MISSING_EXACT_BLOCK_DIM = "missing_exact_block_dim"
    PARTIAL_PHYSICAL_WARP = "partial_physical_warp"
    GROUP_KIND = "group_kind"
    OPERAND_FORM = "operand_form"
    CUB_BROADCAST = "cub_broadcast"
    OPERATION_VARIANT = "operation_variant"
    LAUNCH_CAPABILITY = "launch_capability"


class CudaxReturnKind(str, Enum):
    VALUE = "value"
    OPTIONAL_VALUE = "optional_value"


@dataclass(frozen=True, eq=False)
class GroupReduceSemantics:
    primitive: ReduceSemantics
    broadcast: bool = True
    cub_algorithm: BlockReduceAlgorithm | str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.primitive, ReduceSemantics):
            raise TypeError("primitive must be ReduceSemantics")
        if not isinstance(self.broadcast, bool):
            raise TypeError("broadcast must be a bool")
        if self.cub_algorithm is not None:
            try:
                algorithm = normalize_block_reduce_algorithm(self.cub_algorithm)
            except ValueError as exc:
                raise ValueError(
                    f"unsupported CUB BlockReduce algorithm {self.cub_algorithm!r}"
                ) from exc
            object.__setattr__(self, "cub_algorithm", algorithm)

    @property
    def dtype(self) -> Any:
        return self.primitive.dtype

    @property
    def operation(self) -> ReduceOperation:
        return self.primitive.operation

    @property
    def operand_kind(self) -> GroupOperandKind:
        return GroupOperandKind(self.primitive.value_kind.value)

    @property
    def items_per_thread(self) -> int:
        return self.primitive.items_per_thread

    @property
    def valid_items(self) -> ArgumentBinding:
        return self.primitive.valid_items

    @property
    def reduce_operator(self) -> CxxOperator | PythonOperator | StatefulOperator | None:
        return self.primitive.reduce_operator

    @property
    def requests_cub(self) -> bool:
        return self.cub_algorithm is not None or self.primitive.has_valid_items

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            self.primitive.semantic_key,
            self.broadcast,
            None if self.cub_algorithm is None else self.cub_algorithm.value,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupReduceSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


@dataclass(frozen=True, eq=False)
class GroupScanSemantics:
    primitive: ScanSemantics
    cub_algorithm: BlockScanAlgorithm | str | None = None
    valid_items: ArgumentBinding = field(default_factory=ArgumentBinding.omitted)

    def __post_init__(self) -> None:
        if not isinstance(self.primitive, ScanSemantics):
            raise TypeError("primitive must be ScanSemantics")
        if not isinstance(self.valid_items, ArgumentBinding):
            raise TypeError("valid_items must be an ArgumentBinding")
        if self.cub_algorithm is not None:
            try:
                algorithm = normalize_block_scan_algorithm(self.cub_algorithm)
            except ValueError as exc:
                raise ValueError(
                    f"unsupported CUB BlockScan algorithm {self.cub_algorithm!r}"
                ) from exc
            object.__setattr__(self, "cub_algorithm", algorithm)

    @property
    def dtype(self) -> Any:
        return self.primitive.dtype

    @property
    def mode(self) -> GroupScanMode:
        return GroupScanMode(self.primitive.mode.value)

    @property
    def operand_kind(self) -> GroupOperandKind:
        return GroupOperandKind(self.primitive.value_kind.value)

    @property
    def items_per_thread(self) -> int:
        return self.primitive.items_per_thread

    @property
    def scan_operator(self) -> CxxOperator | PythonOperator | StatefulOperator | None:
        return self.primitive.scan_operator

    @property
    def initial_value(self) -> CxxFunction | Reference | None:
        return self.primitive.initial_value

    @property
    def aggregate(self) -> bool:
        return self.primitive.aggregate

    @property
    def prefix_callback(self) -> PythonOperator | StatefulOperator | None:
        return self.primitive.prefix_callback

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            self.primitive.semantic_key,
            None if self.cub_algorithm is None else self.cub_algorithm.value,
            (
                self.valid_items.kind.value,
                semantic_token(self.valid_items.value),
            ),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupScanSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


@dataclass(frozen=True, eq=False)
class GroupAdjacentDifferenceSemantics:
    """Block-adjacent-difference semantics attached to an explicit group."""

    primitive: BlockAdjacentDifferenceSemantics

    def __post_init__(self) -> None:
        if not isinstance(self.primitive, BlockAdjacentDifferenceSemantics):
            raise TypeError("primitive must be BlockAdjacentDifferenceSemantics")

    @property
    def dtype(self) -> Any:
        return self.primitive.dtype

    @property
    def direction(self) -> BlockAdjacentDifferenceDirection:
        return self.primitive.direction

    @property
    def items_per_thread(self) -> int:
        return self.primitive.items_per_thread

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.primitive.semantic_key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupAdjacentDifferenceSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


@dataclass(frozen=True, eq=False)
class GroupMergeSortSemantics:
    """CUB MergeSort semantics attached to an explicit thread group."""

    primitive: BlockMergeSortSemantics

    def __post_init__(self) -> None:
        if not isinstance(self.primitive, BlockMergeSortSemantics):
            raise TypeError("primitive must be BlockMergeSortSemantics")

    @property
    def dtype(self) -> Any:
        return self.primitive.key_dtype

    @property
    def key_dtype(self) -> Any:
        return self.primitive.key_dtype

    @property
    def value_dtype(self) -> Any | None:
        return self.primitive.value_dtype

    @property
    def items_per_thread(self) -> int:
        return self.primitive.items_per_thread

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.primitive.semantic_key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupMergeSortSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


@dataclass(frozen=True, eq=False)
class GroupDiscontinuitySemantics:
    """Block-discontinuity semantics attached to an explicit group."""

    primitive: BlockDiscontinuitySemantics

    def __post_init__(self) -> None:
        if not isinstance(self.primitive, BlockDiscontinuitySemantics):
            raise TypeError("primitive must be BlockDiscontinuitySemantics")

    @property
    def dtype(self) -> Any:
        return self.primitive.dtype

    @property
    def flag_dtype(self) -> Any:
        return self.primitive.flag_dtype

    @property
    def mode(self) -> BlockDiscontinuityMode:
        return self.primitive.mode

    @property
    def items_per_thread(self) -> int:
        return self.primitive.items_per_thread

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.primitive.semantic_key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupDiscontinuitySemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


@dataclass(frozen=True, eq=False)
class GroupRadixSortSemantics:
    """Block-radix-sort semantics attached to an explicit group."""

    primitive: BlockRadixSortSemantics
    operand_kind: GroupOperandKind = GroupOperandKind.ARRAY

    def __post_init__(self) -> None:
        if not isinstance(self.primitive, BlockRadixSortSemantics):
            raise TypeError("primitive must be BlockRadixSortSemantics")
        object.__setattr__(self, "operand_kind", GroupOperandKind(self.operand_kind))
        if self.primitive.bit_policy is not BlockRadixSortBitPolicy.EXPLICIT:
            raise ValueError("group radix sort requires explicit runtime bit bounds")
        if self.primitive.output is not BlockRadixSortOutput.BLOCKED:
            raise ValueError("group radix sort requires blocked output")
        if (
            self.operand_kind is GroupOperandKind.SCALAR
            and self.primitive.items_per_thread != 1
        ):
            raise ValueError("scalar group radix sort requires one item per thread")

    @property
    def dtype(self) -> Any:
        return self.primitive.key_dtype

    @property
    def items_per_thread(self) -> int:
        return self.primitive.items_per_thread

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.primitive.semantic_key, self.operand_kind.value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupRadixSortSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


@dataclass(frozen=True, eq=False)
class GroupShuffleSemantics:
    """Public-CUB-compatible block-shuffle semantics for an explicit group."""

    primitive: BlockShuffleSemantics

    def __post_init__(self) -> None:
        if not isinstance(self.primitive, BlockShuffleSemantics):
            raise TypeError("primitive must be BlockShuffleSemantics")

    @property
    def dtype(self) -> Any:
        return self.primitive.dtype

    @property
    def mode(self) -> BlockShuffleMode:
        return self.primitive.mode

    @property
    def items_per_thread(self) -> int:
        return self.primitive.items_per_thread or 1

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.primitive.semantic_key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupShuffleSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


@dataclass(frozen=True, eq=False)
class GroupHistogramSemantics:
    """Static-width BlockHistogram semantics attached to an explicit group."""

    primitive: BlockHistogramSemantics
    bins_per_thread: int

    def __post_init__(self) -> None:
        if not isinstance(self.primitive, BlockHistogramSemantics):
            raise TypeError("primitive must be BlockHistogramSemantics")
        if not self.primitive.has_static_bins:
            raise ValueError("group histogram requires a static bin count")
        if self.primitive.operation is not BlockHistogramOperation.HISTOGRAM:
            raise ValueError("group histogram requires the public Histogram operation")
        if (
            not isinstance(self.bins_per_thread, int)
            or isinstance(self.bins_per_thread, bool)
            or self.bins_per_thread < 1
        ):
            raise ValueError("bins_per_thread must be a positive integer")

    @property
    def dtype(self) -> Any:
        """Result counter dtype consumed by the common result contract."""

        return self.primitive.counter_dtype

    @property
    def items_per_thread(self) -> int:
        """Number of striped histogram counters returned to each member."""

        return self.bins_per_thread

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.primitive.semantic_key, self.bins_per_thread

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupHistogramSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


@dataclass(frozen=True, eq=False)
class GroupRunLengthDecodeSemantics:
    """Fused public-CUB run-length decode semantics for one block group."""

    primitive: BlockRunLengthDecodeSemantics

    def __post_init__(self) -> None:
        if not isinstance(self.primitive, BlockRunLengthDecodeSemantics):
            raise TypeError("primitive must be BlockRunLengthDecodeSemantics")
        if self.primitive.run_length_dtype is None:
            raise ValueError("group run-length decode requires a run-length dtype")
        if self.primitive.total_decoded_size_dtype is None:
            raise ValueError(
                "group run-length decode requires a total decoded-size dtype"
            )
        if not self.primitive.returns_total_decoded_size:
            raise ValueError(
                "group run-length decode requires the fused total-size result"
            )

    @property
    def dtype(self) -> Any:
        return self.primitive.item_dtype

    @property
    def items_per_thread(self) -> int:
        return self.primitive.decoded_items_per_thread

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.primitive.semantic_key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupRunLengthDecodeSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


@dataclass(frozen=True, eq=False)
class GroupRadixRankSemantics:
    """Block-radix-rank semantics attached to an explicit group.

    ``primitive.key_dtype`` is the unsigned bit-ordered type consumed by CUB.
    ``input_dtype`` records the public key type before CUTLASS adapts signed
    keys to that representation.
    """

    primitive: BlockRadixRankSemantics
    input_dtype: Any
    operand_kind: GroupOperandKind = GroupOperandKind.ARRAY

    def __post_init__(self) -> None:
        if not isinstance(self.primitive, BlockRadixRankSemantics):
            raise TypeError("primitive must be BlockRadixRankSemantics")
        if self.input_dtype is None:
            raise ValueError("input_dtype must be provided")
        object.__setattr__(self, "operand_kind", GroupOperandKind(self.operand_kind))
        if not self.primitive.bit_range.is_static:
            raise ValueError("group radix rank requires a static radix bit range")
        if (
            self.operand_kind is GroupOperandKind.SCALAR
            and self.primitive.items_per_thread != 1
        ):
            raise ValueError("scalar group radix rank requires one item per thread")

    @property
    def dtype(self) -> Any:
        return self.input_dtype

    @property
    def items_per_thread(self) -> int:
        return self.primitive.items_per_thread

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            self.primitive.semantic_key,
            semantic_token(self.input_dtype),
            self.operand_kind.value,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupRadixRankSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


@dataclass(frozen=True, eq=False)
class GroupExchangeSemantics:
    primitive: BlockExchangeSemantics

    def __post_init__(self) -> None:
        if not isinstance(self.primitive, BlockExchangeSemantics):
            raise TypeError("primitive must be BlockExchangeSemantics")
        if self.primitive.value_form is not BlockExchangeValueForm.OUT_OF_PLACE:
            raise ValueError("group exchange requires out-of-place value form")

    @property
    def dtype(self) -> Any:
        return self.primitive.dtype

    @property
    def mode(self) -> GroupExchangeMode:
        return GroupExchangeMode(self.primitive.mode.value)

    @property
    def items_per_thread(self) -> int:
        return self.primitive.items_per_thread

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.primitive.semantic_key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupExchangeSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


@dataclass(frozen=True, eq=False)
class GroupLoadStoreSemantics:
    kind: GroupLoadStoreKind
    dtype: Any
    items_per_thread: int
    algorithm: GroupLoadStoreAlgorithm = GroupLoadStoreAlgorithm.DIRECT
    valid_items: ArgumentBinding = field(default_factory=ArgumentBinding.omitted)
    oob_default: ArgumentBinding = field(default_factory=ArgumentBinding.omitted)
    offset: ArgumentBinding = field(default_factory=ArgumentBinding.omitted)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", GroupLoadStoreKind(self.kind))
        object.__setattr__(
            self,
            "algorithm",
            GroupLoadStoreAlgorithm(self.algorithm),
        )
        if (
            not isinstance(self.items_per_thread, int)
            or isinstance(self.items_per_thread, bool)
            or self.items_per_thread <= 0
        ):
            raise ValueError("items_per_thread must be a positive integer")
        for name in ("valid_items", "oob_default", "offset"):
            if not isinstance(getattr(self, name), ArgumentBinding):
                raise TypeError(f"{name} must be an ArgumentBinding")
        if self.kind is GroupLoadStoreKind.STORE and (
            self.oob_default.kind is not BindingKind.OMITTED
        ):
            raise ValueError("oob_default is valid only for group load")
        if (
            self.oob_default.kind is not BindingKind.OMITTED
            and self.valid_items.kind is BindingKind.OMITTED
        ):
            raise ValueError("oob_default requires valid_items")

    @property
    def has_valid_items(self) -> bool:
        return self.valid_items.kind is not BindingKind.OMITTED

    @property
    def has_oob_default(self) -> bool:
        return self.oob_default.kind is not BindingKind.OMITTED

    @property
    def has_offset(self) -> bool:
        return self.offset.kind is not BindingKind.OMITTED

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            f"group_{self.kind.value}",
            semantic_token(self.dtype),
            self.items_per_thread,
            self.algorithm.value,
            self.valid_items.semantic_key,
            self.oob_default.semantic_key,
            self.offset.semantic_key,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupLoadStoreSemantics):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


GroupOperationSemantics = (
    GroupReduceSemantics
    | GroupScanSemantics
    | GroupAdjacentDifferenceSemantics
    | GroupMergeSortSemantics
    | GroupDiscontinuitySemantics
    | GroupShuffleSemantics
    | GroupHistogramSemantics
    | GroupRunLengthDecodeSemantics
    | GroupRadixSortSemantics
    | GroupRadixRankSemantics
    | GroupExchangeSemantics
    | GroupLoadStoreSemantics
)


def _primitive_semantic_key(operation: GroupOperationSemantics) -> tuple[Any, ...]:
    return operation.semantic_key


def _lowered_operation_semantic_key(
    operation: GroupOperationSemantics,
    target: GroupLoweringTarget,
) -> tuple[Any, ...]:
    if (
        isinstance(operation, GroupScanSemantics)
        and target is GroupLoweringTarget.CUB_BLOCK
    ):
        algorithm = operation.cub_algorithm or BlockScanAlgorithm.RAKING
        return operation.primitive.semantic_key, algorithm.value
    return _primitive_semantic_key(operation)


def _requested_result_visibility(
    operation: GroupOperationSemantics,
) -> ResultVisibility:
    if isinstance(operation, GroupReduceSemantics):
        return (
            ResultVisibility.ALL_MEMBERS
            if operation.broadcast
            else ResultVisibility.GROUP_ROOT
        )
    return ResultVisibility.PER_MEMBER


def _requested_group_key(group: ThreadGroup) -> tuple[Any, ...]:
    hierarchy = group.hierarchy
    assert hierarchy is not None
    if group.kind == "warp":
        return "warp", "physical", 32
    if group.kind == "block":
        return "block", hierarchy.block_dim
    if group.kind == "cluster":
        return "cluster", hierarchy.block_dim, hierarchy.cluster_dim
    if group.kind == "grid":
        return (
            "grid",
            hierarchy.block_dim,
            hierarchy.cluster_dim,
            hierarchy.grid_dim,
        )
    if group.kind in MAPPED_GROUP_KINDS:
        return group.semantic_key
    return (group.kind,)


@dataclass(frozen=True, eq=False)
class GroupPrimitiveCall:
    group: ThreadGroup
    operation: GroupOperationSemantics
    source: str = field(default="canonical", compare=False, hash=False)
    argument_classifications: tuple[ParameterClassification, ...] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.group, ThreadGroup):
            raise TypeError("GroupPrimitiveCall group must be a ThreadGroup")
        if not isinstance(
            self.operation,
            (
                GroupReduceSemantics,
                GroupScanSemantics,
                GroupAdjacentDifferenceSemantics,
                GroupMergeSortSemantics,
                GroupDiscontinuitySemantics,
                GroupShuffleSemantics,
                GroupHistogramSemantics,
                GroupRunLengthDecodeSemantics,
                GroupRadixSortSemantics,
                GroupRadixRankSemantics,
                GroupExchangeSemantics,
                GroupLoadStoreSemantics,
            ),
        ):
            raise TypeError("unsupported GroupPrimitiveCall operation")
        object.__setattr__(
            self,
            "argument_classifications",
            _call_classifications(self.operation),
        )

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            _requested_group_key(self.group),
            _primitive_semantic_key(self.operation),
            _requested_result_visibility(self.operation).value,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupPrimitiveCall):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


@dataclass(frozen=True)
class CudaxCallDescription:
    primitive: str
    header: str
    namespace: str
    overload: str | None = None
    parameters: tuple[ParameterClassification, ...] = ()
    return_kind: CudaxReturnKind = CudaxReturnKind.VALUE

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(self, "return_kind", CudaxReturnKind(self.return_kind))
        if any(
            not isinstance(parameter, ParameterClassification)
            for parameter in self.parameters
        ):
            raise TypeError("CUDAX parameters must be ParameterClassification records")
        forbidden = {"group", "launch", "launch_facts"}
        if any(parameter.name in forbidden for parameter in self.parameters):
            raise ValueError("CUDAX runtime ABI cannot contain group or launch markers")

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            self.primitive,
            self.header,
            self.namespace,
            self.overload,
            self.parameters,
            self.return_kind.value,
        )


@dataclass(frozen=True)
class ArgumentPrecondition:
    name: str
    minimum: int | None
    maximum: int | None
    enforcement: PreconditionEnforcement

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("argument precondition name must not be empty")
        object.__setattr__(
            self,
            "enforcement",
            PreconditionEnforcement(self.enforcement),
        )
        for bound_name, bound in (
            ("minimum", self.minimum),
            ("maximum", self.maximum),
        ):
            if bound is not None and (
                not isinstance(bound, int) or isinstance(bound, bool)
            ):
                raise TypeError(f"{bound_name} must be an integer or None")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("argument precondition minimum exceeds maximum")

    def validate(self, value: int) -> None:
        """Validate a concrete value when a caller can inspect it."""

        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{self.name} must be an integer")
        if self.minimum is not None and value < self.minimum:
            raise ValueError(f"{self.name} must be at least {self.minimum}")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"{self.name} must be at most {self.maximum}")


@dataclass(frozen=True)
class ParticipationContract:
    group_kind: str
    exact_group_size: int
    exact_block_dim: Dim3 | None
    complete_membership: bool
    contiguous: bool
    aligned: bool
    converged_entry: bool
    complete_parent_partition: bool
    uniform_arguments: tuple[str, ...] = ()
    valid_member_selection: str | None = None
    argument_preconditions: tuple[ArgumentPrecondition, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "uniform_arguments", tuple(self.uniform_arguments))
        object.__setattr__(
            self,
            "argument_preconditions",
            tuple(self.argument_preconditions),
        )
        if any(
            not isinstance(precondition, ArgumentPrecondition)
            for precondition in self.argument_preconditions
        ):
            raise TypeError(
                "argument_preconditions must contain ArgumentPrecondition records"
            )
        names = [precondition.name for precondition in self.argument_preconditions]
        if len(names) != len(set(names)):
            raise ValueError("argument precondition names must be unique")


@dataclass(frozen=True, eq=False)
class LogicalResultContract:
    name: str
    dtype: Any
    visibility: ResultVisibility
    ownership: ResultOwnership
    operand_kind: GroupOperandKind
    items_per_member: int
    root_rank: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("logical result name must not be empty")
        object.__setattr__(self, "visibility", ResultVisibility(self.visibility))
        object.__setattr__(self, "ownership", ResultOwnership(self.ownership))
        object.__setattr__(self, "operand_kind", GroupOperandKind(self.operand_kind))
        if (
            not isinstance(self.items_per_member, int)
            or isinstance(self.items_per_member, bool)
            or self.items_per_member < 1
        ):
            raise ValueError("items_per_member must be a positive integer")
        if self.operand_kind is GroupOperandKind.SCALAR and self.items_per_member != 1:
            raise ValueError("scalar logical results contain exactly one item")
        is_root_result = self.ownership is ResultOwnership.GROUP_ROOT
        if is_root_result != (self.visibility is ResultVisibility.GROUP_ROOT):
            raise ValueError("group-root visibility and ownership must agree")
        if is_root_result:
            if (
                not isinstance(self.root_rank, int)
                or isinstance(self.root_rank, bool)
                or self.root_rank != 0
            ):
                raise ValueError("group-root results require root rank 0")
        elif self.root_rank is not None:
            raise ValueError("non-root results cannot define a root rank")

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            self.name,
            semantic_token(self.dtype),
            self.visibility.value,
            self.ownership.value,
            self.operand_kind.value,
            self.items_per_member,
            self.root_rank,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LogicalResultContract):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


@dataclass(frozen=True)
class ResultContract:
    values: tuple[LogicalResultContract, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))
        if not self.values:
            raise ValueError("result contract requires at least one logical result")
        if any(not isinstance(value, LogicalResultContract) for value in self.values):
            raise TypeError("values must contain LogicalResultContract records")
        names = [value.name for value in self.values]
        if len(names) != len(set(names)):
            raise ValueError("logical result names must be unique")

    @property
    def primary(self) -> LogicalResultContract:
        return self.values[0]

    @property
    def visibility(self) -> ResultVisibility:
        return self.primary.visibility

    @property
    def operand_kind(self) -> GroupOperandKind:
        return self.primary.operand_kind

    @property
    def result_items_per_thread(self) -> int:
        return self.primary.items_per_member

    @property
    def has_aggregate(self) -> bool:
        return any(value.name == "aggregate" for value in self.values)


@dataclass(frozen=True)
class SynchronizationContract:
    converged_entry: bool
    storage_reuse_barrier: SynchronizationScope


@dataclass(frozen=True)
class TempStorageContract:
    ownership: StorageOwnership
    address_space: str | None
    cpp_type: str | None
    instances: int | None
    instance_index: str | None
    exact_layout_required: bool


@dataclass(frozen=True)
class ImplementationProvenance:
    library: str
    header: str
    cpp_class: str
    method: str
    note: str = field(default="", compare=False, hash=False)

    @property
    def semantic_key(self) -> tuple[str, str, str, str]:
        return self.library, self.header, self.cpp_class, self.method


@dataclass(frozen=True)
class UnsupportedReason:
    code: UnsupportedReasonCode
    message: str = field(compare=False, hash=False)


@dataclass(frozen=True)
class ThreadGroupResolution:
    """One launch-reconciled static group or a typed unsupported reason."""

    group: ThreadGroup
    unsupported: UnsupportedReason | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.group, ThreadGroup):
            raise TypeError("ThreadGroupResolution group must be a ThreadGroup")
        if self.unsupported is not None and not isinstance(
            self.unsupported, UnsupportedReason
        ):
            raise TypeError(
                "ThreadGroupResolution unsupported must be an UnsupportedReason"
            )

    def require_supported(self) -> ThreadGroup:
        if self.unsupported is not None:
            raise NotImplementedError(self.unsupported.message)
        return self.group


def _canonical_group_key(group: ThreadGroup) -> tuple[Any, ...]:
    hierarchy = group.hierarchy
    assert hierarchy is not None
    if group.kind == "warp":
        return "warp", "physical", 32
    if group.kind == "block":
        return "block", hierarchy.block_dim
    if group.kind == "cluster":
        return "cluster", hierarchy.block_dim, hierarchy.cluster_dim
    if group.kind == "grid":
        return (
            "grid",
            hierarchy.block_dim,
            hierarchy.cluster_dim,
            hierarchy.grid_dim,
        )
    if group.kind in MAPPED_GROUP_KINDS:
        return group.semantic_key
    return (group.kind,)


@dataclass(frozen=True, eq=False)
class GroupLoweringPlan:
    target: GroupLoweringTarget
    call: GroupPrimitiveCall
    resolved_group: ThreadGroup
    implementation: CudaxCallDescription | AlgorithmSpec | None
    participation: ParticipationContract | None
    result: ResultContract | None
    synchronization: SynchronizationContract | None
    temp_storage: TempStorageContract | None
    provenance: ImplementationProvenance | None
    unsupported: UnsupportedReason | None = None

    def __post_init__(self) -> None:
        is_unsupported = self.target is GroupLoweringTarget.UNSUPPORTED
        if is_unsupported != (self.unsupported is not None):
            raise ValueError("unsupported plans require exactly one reason")
        result_required = not (
            isinstance(self.call.operation, GroupLoadStoreSemantics)
            and self.call.operation.kind is GroupLoadStoreKind.STORE
        )
        if not is_unsupported and (
            self.implementation is None
            or self.participation is None
            or self.synchronization is None
            or self.temp_storage is None
            or self.provenance is None
            or (result_required and self.result is None)
        ):
            raise ValueError("supported plans require complete lowering contracts")

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        result_visibility = (
            None if self.result is None else self.result.visibility.value
        )
        return (
            _canonical_group_key(self.resolved_group),
            _lowered_operation_semantic_key(self.call.operation, self.target),
            result_visibility,
        )

    @property
    def artifact_key(self) -> tuple[Any, ...] | None:
        if self.unsupported is not None:
            return None
        implementation_key = (
            None
            if self.implementation is None
            else getattr(self.implementation, "semantic_key")
        )
        return (
            self.target.value,
            _canonical_group_key(self.resolved_group),
            self.resolved_group.hierarchy.block_dim,
            _lowered_operation_semantic_key(self.call.operation, self.target),
            implementation_key,
            self.participation,
            self.result,
            self.synchronization,
            self.temp_storage,
            None if self.provenance is None else self.provenance.semantic_key,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupLoweringPlan):
            return NotImplemented
        return self._identity_key == other._identity_key

    def __hash__(self) -> int:
        return hash(self._identity_key)

    @property
    def _identity_key(self) -> tuple[Any, ...]:
        if self.artifact_key is not None:
            return "artifact", self.artifact_key
        assert self.unsupported is not None
        return "unsupported", self.semantic_key, self.unsupported.code.value

    def require_supported(self) -> "GroupLoweringPlan":
        if self.unsupported is not None:
            raise NotImplementedError(self.unsupported.message)
        return self


def _call_classifications(
    operation: GroupOperationSemantics,
) -> tuple[ParameterClassification, ...]:
    if isinstance(operation, GroupLoadStoreSemantics):
        classifications = [
            ParameterClassification(
                "source"
                if operation.kind is GroupLoadStoreKind.LOAD
                else "destination",
                ArgumentKind.RUNTIME,
                (
                    ParameterRole.INPUT
                    if operation.kind is GroupLoadStoreKind.LOAD
                    else ParameterRole.OUTPUT
                ),
            )
        ]
        if operation.kind is GroupLoadStoreKind.STORE:
            classifications.append(
                ParameterClassification(
                    "value",
                    ArgumentKind.RUNTIME,
                    ParameterRole.INPUT,
                )
            )
        for name, binding in (
            ("valid_items", operation.valid_items),
            ("oob_default", operation.oob_default),
            ("offset", operation.offset),
        ):
            if binding.argument_kind is None:
                continue
            classifications.append(
                ParameterClassification(
                    name,
                    binding.argument_kind,
                    (
                        ParameterRole.CONSTANT
                        if binding.kind is BindingKind.STATIC
                        else ParameterRole.INPUT
                    ),
                )
            )
        classifications.append(
            ParameterClassification(
                "algorithm",
                ArgumentKind.STATIC,
                ParameterRole.CONSTANT,
            )
        )
        return tuple(classifications)

    if isinstance(operation, GroupMergeSortSemantics):
        primitive = operation.primitive
        classifications = [
            ParameterClassification("keys", ArgumentKind.RUNTIME, ParameterRole.INOUT)
        ]
        if primitive.has_values:
            classifications.append(
                ParameterClassification(
                    "values", ArgumentKind.RUNTIME, ParameterRole.INOUT
                )
            )
        compare = classify_parameter(primitive.compare_operator)
        classifications.append(
            ParameterClassification("compare_op", compare.kind, compare.role)
        )
        if primitive.has_partial_tile:
            classifications.extend(
                (
                    ParameterClassification(
                        "valid_items", ArgumentKind.RUNTIME, ParameterRole.INPUT
                    ),
                    ParameterClassification(
                        "oob_default", ArgumentKind.RUNTIME, ParameterRole.INPUT
                    ),
                )
            )
        return tuple(classifications)

    if isinstance(operation, GroupRunLengthDecodeSemantics):
        primitive = operation.primitive
        classifications = [
            ParameterClassification(
                "run_values", ArgumentKind.RUNTIME, ParameterRole.INPUT
            ),
            ParameterClassification(
                "run_lengths", ArgumentKind.RUNTIME, ParameterRole.INPUT
            ),
            ParameterClassification(
                "decoded_items_per_thread",
                ArgumentKind.STATIC,
                ParameterRole.CONSTANT,
            ),
            ParameterClassification(
                "decoded_window_offset",
                ArgumentKind.RUNTIME,
                ParameterRole.INPUT,
            ),
        ]
        if primitive.has_relative_offsets:
            classifications.append(
                ParameterClassification(
                    "relative_offsets",
                    ArgumentKind.RUNTIME,
                    ParameterRole.OUTPUT,
                )
            )
        classifications.append(
            ParameterClassification(
                "total_decoded_size",
                ArgumentKind.RUNTIME,
                ParameterRole.OUTPUT,
            )
        )
        return tuple(classifications)

    if isinstance(operation, GroupRadixSortSemantics):
        classifications = [
            ParameterClassification("keys", ArgumentKind.RUNTIME, ParameterRole.INPUT)
        ]
        if operation.primitive.has_values:
            classifications.append(
                ParameterClassification(
                    "values", ArgumentKind.RUNTIME, ParameterRole.INPUT
                )
            )
        classifications.extend(
            (
                ParameterClassification(
                    "begin_bit", ArgumentKind.RUNTIME, ParameterRole.INPUT
                ),
                ParameterClassification(
                    "end_bit", ArgumentKind.RUNTIME, ParameterRole.INPUT
                ),
                ParameterClassification(
                    "order", ArgumentKind.STATIC, ParameterRole.CONSTANT
                ),
                ParameterClassification(
                    "payload", ArgumentKind.STATIC, ParameterRole.CONSTANT
                ),
                ParameterClassification(
                    "output", ArgumentKind.STATIC, ParameterRole.CONSTANT
                ),
            )
        )
        return tuple(classifications)

    if isinstance(operation, GroupRadixRankSemantics):
        classifications = [
            ParameterClassification("keys", ArgumentKind.RUNTIME, ParameterRole.INPUT),
            ParameterClassification(
                "begin_bit", ArgumentKind.STATIC, ParameterRole.CONSTANT
            ),
            ParameterClassification(
                "end_bit", ArgumentKind.STATIC, ParameterRole.CONSTANT
            ),
            ParameterClassification(
                "order", ArgumentKind.STATIC, ParameterRole.CONSTANT
            ),
        ]
        if operation.primitive.has_exclusive_digit_prefix:
            classifications.append(
                ParameterClassification(
                    "exclusive_digit_prefix",
                    ArgumentKind.RUNTIME,
                    ParameterRole.OUTPUT,
                )
            )
        return tuple(classifications)

    primary_argument = (
        "samples" if isinstance(operation, GroupHistogramSemantics) else "value"
    )
    classifications = [
        ParameterClassification(
            primary_argument,
            ArgumentKind.RUNTIME,
            ParameterRole.INPUT,
        )
    ]
    if isinstance(operation, GroupHistogramSemantics):
        classifications.extend(
            (
                ParameterClassification(
                    "bins", ArgumentKind.STATIC, ParameterRole.CONSTANT
                ),
                ParameterClassification(
                    "bins_per_thread", ArgumentKind.STATIC, ParameterRole.CONSTANT
                ),
                ParameterClassification(
                    "algorithm", ArgumentKind.STATIC, ParameterRole.CONSTANT
                ),
            )
        )
    elif isinstance(operation, GroupAdjacentDifferenceSemantics):
        classifications.extend(
            (
                ParameterClassification(
                    "direction", ArgumentKind.STATIC, ParameterRole.CONSTANT
                ),
                ParameterClassification(
                    "operation", ArgumentKind.STATIC, ParameterRole.OPERATOR
                ),
            )
        )
        if operation.primitive.has_partial_tile:
            classifications.append(
                ParameterClassification(
                    "valid_items", ArgumentKind.RUNTIME, ParameterRole.INPUT
                )
            )
        if operation.primitive.boundary is not BlockAdjacentDifferenceBoundary.NONE:
            boundary_name = (
                "tile_predecessor_item"
                if operation.primitive.boundary
                is BlockAdjacentDifferenceBoundary.PREDECESSOR
                else "tile_successor_item"
            )
            classifications.append(
                ParameterClassification(
                    boundary_name, ArgumentKind.RUNTIME, ParameterRole.INPUT
                )
            )
    elif isinstance(operation, GroupDiscontinuitySemantics):
        classifications.extend(
            (
                ParameterClassification(
                    "mode", ArgumentKind.STATIC, ParameterRole.CONSTANT
                ),
                ParameterClassification(
                    "operation", ArgumentKind.STATIC, ParameterRole.OPERATOR
                ),
            )
        )
        if operation.primitive.has_tile_predecessor:
            classifications.append(
                ParameterClassification(
                    "tile_predecessor_item",
                    ArgumentKind.RUNTIME,
                    ParameterRole.INPUT,
                )
            )
        if operation.primitive.has_tile_successor:
            classifications.append(
                ParameterClassification(
                    "tile_successor_item",
                    ArgumentKind.RUNTIME,
                    ParameterRole.INPUT,
                )
            )
    elif isinstance(operation, GroupShuffleSemantics):
        classifications.append(
            ParameterClassification("mode", ArgumentKind.STATIC, ParameterRole.CONSTANT)
        )
        if operation.primitive.distance.argument_kind is not None:
            classifications.append(
                ParameterClassification(
                    "distance",
                    operation.primitive.distance.argument_kind,
                    (
                        ParameterRole.CONSTANT
                        if operation.primitive.distance.kind is BindingKind.STATIC
                        else ParameterRole.INPUT
                    ),
                )
            )
        for boundary_name, enabled in (
            ("block_prefix", operation.primitive.block_prefix),
            ("block_suffix", operation.primitive.block_suffix),
        ):
            if enabled:
                classifications.append(
                    ParameterClassification(
                        boundary_name,
                        ArgumentKind.RUNTIME,
                        ParameterRole.OUTPUT,
                    )
                )
    elif isinstance(operation, GroupExchangeSemantics):
        classifications.append(
            ParameterClassification("mode", ArgumentKind.STATIC, ParameterRole.CONSTANT)
        )
        if operation.primitive.uses_ranks:
            classifications.append(
                ParameterClassification(
                    "ranks",
                    ArgumentKind.RUNTIME,
                    ParameterRole.INPUT,
                )
            )
        if operation.primitive.uses_valid_flags:
            classifications.append(
                ParameterClassification(
                    "valid_flags",
                    ArgumentKind.RUNTIME,
                    ParameterRole.INPUT,
                )
            )
    elif isinstance(operation, GroupReduceSemantics):
        if operation.valid_items.argument_kind is not None:
            classifications.append(
                ParameterClassification(
                    "valid_items",
                    operation.valid_items.argument_kind,
                    (
                        ParameterRole.CONSTANT
                        if operation.valid_items.kind is BindingKind.STATIC
                        else ParameterRole.INPUT
                    ),
                )
            )
        if operation.reduce_operator is not None:
            classification = classify_parameter(operation.reduce_operator)
            classifications.append(
                ParameterClassification(
                    "operation",
                    classification.kind,
                    classification.role,
                )
            )
        else:
            classifications.append(
                ParameterClassification(
                    "operation", ArgumentKind.STATIC, ParameterRole.CONSTANT
                )
            )
    elif isinstance(operation, GroupScanSemantics):
        classifications.append(
            ParameterClassification("mode", ArgumentKind.STATIC, ParameterRole.CONSTANT)
        )
        for name, parameter in (
            ("initial_value", operation.initial_value),
            ("operation", operation.scan_operator),
            ("prefix_callback", operation.prefix_callback),
        ):
            if parameter is None:
                continue
            classification = classify_parameter(parameter)
            classifications.append(
                ParameterClassification(
                    name,
                    classification.kind,
                    classification.role,
                )
            )
        if operation.valid_items.argument_kind is not None:
            classifications.append(
                ParameterClassification(
                    "valid_items",
                    operation.valid_items.argument_kind,
                    (
                        ParameterRole.CONSTANT
                        if operation.valid_items.kind is BindingKind.STATIC
                        else ParameterRole.INPUT
                    ),
                )
            )
    else:
        classifications.append(
            ParameterClassification("mode", ArgumentKind.STATIC, ParameterRole.CONSTANT)
        )
    return tuple(classifications)


def make_group_primitive_call(
    group: ThreadGroup,
    operation: GroupOperationSemantics,
    *,
    source: str = "canonical",
) -> GroupPrimitiveCall:
    return GroupPrimitiveCall(
        group=group,
        operation=operation,
        source=source,
    )


def _unsupported(
    call: GroupPrimitiveCall,
    resolved_group: ThreadGroup,
    code: UnsupportedReasonCode,
    message: str,
) -> GroupLoweringPlan:
    return GroupLoweringPlan(
        target=GroupLoweringTarget.UNSUPPORTED,
        call=call,
        resolved_group=resolved_group,
        implementation=None,
        participation=None,
        result=None,
        synchronization=None,
        temp_storage=None,
        provenance=None,
        unsupported=UnsupportedReason(code=code, message=message),
    )


_THREAD_LEVEL_ORDER = {
    "thread": 0,
    "warp": 1,
    "block": 2,
    "cluster": 3,
    "grid": 4,
}

_MAPPED_PARENT_LEVEL = {
    "threads_within_warp": "warp",
    "warps_within_block": "block",
}


def _resolution_failure(
    group: ThreadGroup,
    code: UnsupportedReasonCode,
    message: str,
) -> ThreadGroupResolution:
    return ThreadGroupResolution(
        group=group,
        unsupported=UnsupportedReason(code=code, message=message),
    )


def resolve_thread_group(
    group: ThreadGroup,
    launch: LaunchFacts,
    *,
    through_level: str | None = None,
) -> ThreadGroupResolution:
    """Resolve a group against exact launch facts through a hierarchy level.

    ``through_level`` requests the enclosing hierarchy needed by group queries.
    Collective planners omit it because the group's own level is sufficient.
    Exact dimensions remain distinct from upper bounds, and cluster state must
    be verified before a missing cluster extent can be treated as one block.
    """

    if not isinstance(group, ThreadGroup):
        raise TypeError("group must be a ThreadGroup")
    if not isinstance(launch, LaunchFacts):
        raise TypeError("launch must be LaunchFacts")

    group_level = _MAPPED_PARENT_LEVEL.get(group.kind, group.kind)
    if through_level is not None:
        through_level = normalize_thread_level(
            through_level,
            scope="resolve_thread_group",
            feature="through_level",
        )
    required_level = max(
        (group_level, through_level or group_level),
        key=_THREAD_LEVEL_ORDER.__getitem__,
    )
    needs_complete_warp = (
        group.kind in COMPLETE_WARP_GROUP_KINDS or through_level == "warp"
    )
    if required_level == "thread":
        return ThreadGroupResolution(group)

    exact_block_dim = launch.exact_block_dim
    if exact_block_dim is None:
        return _resolution_failure(
            group,
            UnsupportedReasonCode.MISSING_EXACT_BLOCK_DIM,
            "group operation requires exact block dimensions; max_block_dim "
            "is only an upper bound",
        )
    assert group.hierarchy is not None
    if group.hierarchy.block_dim is not None:
        if group.hierarchy.block_dim != exact_block_dim:
            raise ValueError(
                f"group block dimensions {group.hierarchy.block_dim!r} do not "
                f"match the exact kernel launch dimensions {exact_block_dim!r}",
            )
    needs_cluster = (
        _THREAD_LEVEL_ORDER[required_level] >= _THREAD_LEVEL_ORDER["cluster"]
    )
    exact_cluster_dim = launch.exact_cluster_dim if needs_cluster else None
    if needs_cluster:
        cluster_launch_verified = launch.is_verified("cluster_launch")
        if exact_cluster_dim is None:
            if launch.cluster_launch is not False or not cluster_launch_verified:
                return _resolution_failure(
                    group,
                    UnsupportedReasonCode.LAUNCH_CAPABILITY,
                    "cluster and grid group operations require exact static "
                    "cluster dimensions, or a backend-verified non-cluster launch",
                )
            exact_cluster_dim = (1, 1, 1)
        elif launch.cluster_launch is None or not cluster_launch_verified:
            return _resolution_failure(
                group,
                UnsupportedReasonCode.LAUNCH_CAPABILITY,
                "cluster and grid group operations require backend-verified "
                "cluster launch state",
            )
        elif exact_cluster_dim != (1, 1, 1) and launch.cluster_launch is not True:
            return _resolution_failure(
                group,
                UnsupportedReasonCode.LAUNCH_CAPABILITY,
                "multi-block cluster operations require verified cluster launch "
                "capability",
            )

    hierarchy_grid_dim = None
    needs_grid = required_level == "grid"
    if needs_grid:
        exact_grid_dim = launch.exact_grid_dim
        if exact_grid_dim is None:
            return _resolution_failure(
                group,
                UnsupportedReasonCode.LAUNCH_CAPABILITY,
                "grid group operations require exact static grid dimensions",
            )
        assert exact_cluster_dim is not None
        if any(
            grid_extent % cluster_extent != 0
            for grid_extent, cluster_extent in zip(
                exact_grid_dim,
                exact_cluster_dim,
            )
        ):
            return _resolution_failure(
                group,
                UnsupportedReasonCode.LAUNCH_CAPABILITY,
                "physical CTA grid dimensions must be divisible by the cluster "
                "dimensions",
            )
        hierarchy_grid_dim = tuple(
            grid_extent // cluster_extent
            for grid_extent, cluster_extent in zip(
                exact_grid_dim,
                exact_cluster_dim,
            )
        )

    resolved_hierarchy = ThreadHierarchy._resolved(
        block_dim=exact_block_dim,
        cluster_dim=(exact_cluster_dim if needs_cluster else None),
        grid_dim=hierarchy_grid_dim,
    )
    if (
        needs_cluster
        and group.hierarchy.cluster_dim is not None
        and (group.hierarchy.cluster_dim != resolved_hierarchy.cluster_dim)
    ):
        raise ValueError(
            f"group cluster dimensions {group.hierarchy.cluster_dim!r} do not "
            f"match exact launch dimensions {resolved_hierarchy.cluster_dim!r}"
        )
    if (
        needs_grid
        and group.hierarchy.grid_dim is not None
        and (group.hierarchy.grid_dim != resolved_hierarchy.grid_dim)
    ):
        raise ValueError(
            f"group grid dimensions {group.hierarchy.grid_dim!r} do not match "
            f"exact hierarchy dimensions {resolved_hierarchy.grid_dim!r}"
        )
    resolved = group.with_hierarchy(
        resolved_hierarchy,
        source="launch_facts",
    )
    if needs_complete_warp and launch.exact_block_threads % 32 != 0:  # type: ignore[operator]
        return _resolution_failure(
            resolved,
            UnsupportedReasonCode.PARTIAL_PHYSICAL_WARP,
            "physical-warp operation requires complete 32-thread warps and "
            "every physical warp in the enclosing CTA to be complete; got "
            f"{launch.exact_block_threads} block threads",
        )
    if resolved.mapping is not None:
        parent_units = resolved.parent_unit_count
        assert parent_units is not None
        if resolved.mapping.count > parent_units:
            return _resolution_failure(
                resolved,
                UnsupportedReasonCode.GROUP_KIND,
                "mapped group count exceeds the resolved parent unit count",
            )
        if resolved.mapping.exhaustive and resolved.remainder_count != 0:
            return _resolution_failure(
                resolved,
                UnsupportedReasonCode.GROUP_KIND,
                "exhaustive mapped group count must divide the resolved parent "
                "unit count",
            )
    return ThreadGroupResolution(resolved)


def _resolve_group(
    call: GroupPrimitiveCall,
    launch: LaunchFacts,
) -> tuple[ThreadGroup, GroupLoweringPlan | None]:
    resolution = resolve_thread_group(call.group, launch)
    if resolution.unsupported is None:
        return resolution.group, None
    return resolution.group, _unsupported(
        call,
        resolution.group,
        resolution.unsupported.code,
        resolution.unsupported.message,
    )


def _contracts(
    resolved_group: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupOperationSemantics,
    *,
    visibility: ResultVisibility,
    storage_ownership: StorageOwnership,
    cpp_type: str | None,
    uniform_arguments: tuple[str, ...] = (),
    valid_member_selection: str | None = None,
    argument_preconditions: tuple[ArgumentPrecondition, ...] = (),
    returns_value: bool = True,
) -> tuple[
    ParticipationContract,
    ResultContract | None,
    SynchronizationContract,
    TempStorageContract,
]:
    group_size = resolved_group.static_size
    assert group_size is not None
    if resolved_group.kind in {"warp", "threads_within_warp"}:
        logical_width = (
            32 if resolved_group.kind == "warp" else resolved_group.static_size
        )
        assert logical_width is not None
        instances = launch.exact_block_threads // logical_width  # type: ignore[operator]
        index = f"linear_thread_rank / {logical_width}"
        barrier = SynchronizationScope.WARP
    elif resolved_group.kind == "block":
        instances = 1
        index = "cta"
        barrier = SynchronizationScope.BLOCK
    elif resolved_group.kind == "thread":
        instances = 1
        index = "thread"
        barrier = SynchronizationScope.NONE
    else:
        instances = resolved_group.groups_per_parent or 1
        index = "group.rank(parent)"
        barrier = SynchronizationScope.GROUP
    if isinstance(operation, GroupReduceSemantics):
        result_kind = GroupOperandKind.SCALAR
        result_items_per_member = 1
    elif isinstance(operation, GroupScanSemantics):
        result_kind = operation.operand_kind
        result_items_per_member = operation.items_per_thread
    elif isinstance(operation, GroupShuffleSemantics):
        result_kind = (
            GroupOperandKind.ARRAY
            if operation.primitive.value_kind is BlockShuffleValueKind.ARRAY
            else GroupOperandKind.SCALAR
        )
        result_items_per_member = operation.items_per_thread
    else:
        result_kind = GroupOperandKind.ARRAY
        result_items_per_member = operation.items_per_thread
    ownership = (
        ResultOwnership.GROUP_ROOT
        if visibility is ResultVisibility.GROUP_ROOT
        else ResultOwnership.EACH_MEMBER
    )
    results = []
    if isinstance(operation, GroupMergeSortSemantics):
        results.append(
            LogicalResultContract(
                name="keys",
                dtype=operation.key_dtype,
                visibility=visibility,
                ownership=ownership,
                operand_kind=GroupOperandKind.ARRAY,
                items_per_member=operation.items_per_thread,
            )
        )
        if operation.value_dtype is not None:
            results.append(
                LogicalResultContract(
                    name="values",
                    dtype=operation.value_dtype,
                    visibility=visibility,
                    ownership=ownership,
                    operand_kind=GroupOperandKind.ARRAY,
                    items_per_member=operation.items_per_thread,
                )
            )
    elif isinstance(operation, GroupDiscontinuitySemantics):
        for name, enabled in (
            ("head_flags", operation.primitive.has_heads),
            ("tail_flags", operation.primitive.has_tails),
        ):
            if enabled:
                results.append(
                    LogicalResultContract(
                        name=name,
                        dtype=operation.flag_dtype,
                        visibility=visibility,
                        ownership=ownership,
                        operand_kind=GroupOperandKind.ARRAY,
                        items_per_member=operation.items_per_thread,
                        root_rank=(
                            0 if ownership is ResultOwnership.GROUP_ROOT else None
                        ),
                    )
                )
    elif returns_value:
        results.append(
            LogicalResultContract(
                name="value",
                dtype=operation.dtype,
                visibility=visibility,
                ownership=ownership,
                operand_kind=result_kind,
                items_per_member=result_items_per_member,
                root_rank=(0 if ownership is ResultOwnership.GROUP_ROOT else None),
            )
        )
    if isinstance(operation, GroupShuffleSemantics):
        for name, enabled in (
            ("block_prefix", operation.primitive.block_prefix),
            ("block_suffix", operation.primitive.block_suffix),
        ):
            if enabled:
                results.append(
                    LogicalResultContract(
                        name=name,
                        dtype=operation.dtype,
                        visibility=ResultVisibility.ALL_MEMBERS,
                        ownership=ResultOwnership.EACH_MEMBER,
                        operand_kind=GroupOperandKind.SCALAR,
                        items_per_member=1,
                    )
                )
    if isinstance(operation, GroupScanSemantics) and operation.aggregate:
        results.append(
            LogicalResultContract(
                name="aggregate",
                dtype=operation.dtype,
                visibility=ResultVisibility.ALL_MEMBERS,
                ownership=ResultOwnership.EACH_MEMBER,
                operand_kind=GroupOperandKind.SCALAR,
                items_per_member=1,
            )
        )
    return (
        ParticipationContract(
            group_kind=resolved_group.kind,
            exact_group_size=group_size,
            exact_block_dim=launch.exact_block_dim,
            complete_membership=resolved_group.complete_membership is not False,
            contiguous=True,
            aligned=True,
            converged_entry=True,
            complete_parent_partition=(
                resolved_group.kind == "warp"
                or resolved_group.complete_membership is True
            ),
            uniform_arguments=uniform_arguments,
            valid_member_selection=valid_member_selection,
            argument_preconditions=argument_preconditions,
        ),
        ResultContract(tuple(results)) if results else None,
        SynchronizationContract(
            converged_entry=True,
            storage_reuse_barrier=barrier,
        ),
        TempStorageContract(
            ownership=storage_ownership,
            address_space=(
                None
                if storage_ownership is StorageOwnership.IMPLEMENTATION
                else "shared"
            ),
            cpp_type=cpp_type,
            instances=(
                None
                if storage_ownership is StorageOwnership.IMPLEMENTATION
                else instances
            ),
            instance_index=(
                None if storage_ownership is StorageOwnership.IMPLEMENTATION else index
            ),
            exact_layout_required=storage_ownership is StorageOwnership.CALLER,
        ),
    )


def _stateful_operator_uniformity(operator: Any) -> tuple[str, ...]:
    return ("operation",) if isinstance(operator, StatefulOperator) else ()


def _cub_warp_width(group: ThreadGroup) -> int:
    """Return a CUB-legal physical or logical warp width.

    The physical-warp descriptor always lowers at the architectural width.
    Mapped thread groups are more permissive than CUB, so reject widths that
    CUB cannot instantiate before constructing a low-level specialization.
    """

    if group.kind == "warp":
        return 32
    if group.kind != "threads_within_warp":
        raise ValueError("CUB warp primitives require a warp-based group")
    width = group.static_size
    if (
        not isinstance(width, int)
        or isinstance(width, bool)
        or width < 1
        or width > 32
        or width & (width - 1)
        or 32 % width != 0
    ):
        raise ValueError(
            "CUB-backed logical-warp operations require a power-of-two group "
            "width in [1, 32] that divides the 32-thread physical warp; "
            f"got {width!r}"
        )
    return width


def _unsupported_cub_warp_width(
    call: GroupPrimitiveCall,
    resolved: ThreadGroup,
) -> tuple[int | None, GroupLoweringPlan | None]:
    try:
        return _cub_warp_width(resolved), None
    except ValueError as exc:
        return None, _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.GROUP_KIND,
            str(exc),
        )


_KNOWN_COMMUTATIVE_REDUCE_OPERATORS = frozenset(
    {
        "::cuda::std::plus<>",
        "::cuda::std::multiplies<>",
        "::cuda::minimum<>",
        "::cuda::maximum<>",
        "::cuda::std::bit_and<>",
        "::cuda::std::bit_or<>",
        "::cuda::std::bit_xor<>",
    }
)


def _has_proven_commutative_reduce_operator(
    operation: GroupReduceSemantics,
) -> bool:
    if operation.operation is ReduceOperation.SUM:
        return True
    operator = operation.reduce_operator
    if not isinstance(operator, CxxOperator):
        return False
    cpp = operator.cpp.replace("<T>", "<>").removesuffix("{}")
    return cpp in _KNOWN_COMMUTATIVE_REDUCE_OPERATORS


def _plan_reduce(
    call: GroupPrimitiveCall,
    resolved: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupReduceSemantics,
) -> GroupLoweringPlan:
    if operation.requests_cub and resolved.kind not in {
        "block",
        "warp",
        "threads_within_warp",
    }:
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.OPERATION_VARIANT,
            "valid_items and explicit CUB algorithms are supported only for "
            "physical block, physical-warp, and logical-warp groups",
        )
    if (
        resolved.kind == "block"
        and operation.requests_cub
        and operation.cub_algorithm is None
    ):
        operation = replace(
            operation,
            cub_algorithm=BlockReduceAlgorithm.WARP_REDUCTIONS,
        )
        call = replace(call, operation=operation)
    if operation.requests_cub and operation.broadcast:
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.CUB_BROADCAST,
            "direct CUB reduce returns a defined value only at the group root; "
            "it cannot satisfy broadcast=True",
        )

    if not operation.requests_cub:
        implementation = CudaxCallDescription(
            primitive="reduce",
            header="cuda/experimental/coop.cuh",
            namespace="cuda::experimental::coop",
            overload="broadcasted" if operation.broadcast else "root_only",
            parameters=(
                *(
                    ParameterClassification(
                        f"item{index}",
                        ArgumentKind.RUNTIME,
                        ParameterRole.INPUT,
                    )
                    for index in range(operation.items_per_thread)
                ),
                *(
                    classification
                    for classification in call.argument_classifications
                    if classification.kind is ArgumentKind.RUNTIME
                    and classification.name != "value"
                ),
            ),
            return_kind=(
                CudaxReturnKind.VALUE
                if operation.broadcast
                else CudaxReturnKind.OPTIONAL_VALUE
            ),
        )
        contracts = _contracts(
            resolved,
            launch,
            operation,
            visibility=(
                ResultVisibility.ALL_MEMBERS
                if operation.broadcast
                else ResultVisibility.GROUP_ROOT
            ),
            storage_ownership=StorageOwnership.IMPLEMENTATION,
            cpp_type=None,
            uniform_arguments=_stateful_operator_uniformity(operation.reduce_operator),
        )
        return GroupLoweringPlan(
            target=GroupLoweringTarget.CUDAX_GROUP,
            call=call,
            resolved_group=resolved,
            implementation=implementation,
            participation=contracts[0],
            result=contracts[1],
            synchronization=contracts[2],
            temp_storage=contracts[3],
            provenance=ImplementationProvenance(
                library="CUDAX",
                header=implementation.header,
                cpp_class=implementation.namespace,
                method="reduce",
            ),
        )

    assert launch.exact_block_dim is not None
    operation_name: Any
    if operation.operation is ReduceOperation.SUM:
        operation_name = (
            ReduceOperation.SUM if resolved.kind == "block" else WarpReduceOperation.SUM
        )
    else:
        operation_name = (
            ReduceOperation.REDUCE
            if resolved.kind == "block"
            else WarpReduceOperation.REDUCE
        )
    reduce_operator = operation.reduce_operator

    if operation.valid_items.kind is BindingKind.STATIC:
        valid_items = operation.valid_items.value
        assert isinstance(valid_items, int)
        group_size = resolved.static_size
        assert group_size is not None
        if valid_items < 1:
            raise ValueError("static valid_items must be at least 1")
        if valid_items > group_size:
            raise ValueError(
                f"static valid_items {valid_items} exceeds group size {group_size}"
            )
    if resolved.kind == "block":
        algorithm = operation.cub_algorithm or BlockReduceAlgorithm.WARP_REDUCTIONS
        if algorithm is BlockReduceAlgorithm.WARP_REDUCTIONS_NONDETERMINISTIC:
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.OPERATION_VARIANT,
                "group BlockReduce does not expose "
                "BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC because its "
                "current CUB implementation is addition-specific",
            )
        if (
            algorithm is BlockReduceAlgorithm.RAKING_COMMUTATIVE_ONLY
            and not _has_proven_commutative_reduce_operator(operation)
        ):
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.OPERATION_VARIANT,
                "BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY requires a reduction "
                "operator with proven commutativity",
            )
        spec = make_block_reduce_spec(
            dtype=operation.dtype,
            block_dim=launch.exact_block_dim,
            items_per_thread=operation.items_per_thread,
            operation=operation_name,
            algorithm=algorithm,
            value_kind=ReduceValueKind(operation.operand_kind.value),
            reduce_operator=reduce_operator,
            valid_items=operation.valid_items,
        ).specialization
        target = GroupLoweringTarget.CUB_BLOCK
        cpp_class = "cub::BlockReduce"
        header = "cub/block/block_reduce.cuh"
    else:
        warp_width, width_error = _unsupported_cub_warp_width(call, resolved)
        if width_error is not None:
            return width_error
        assert warp_width is not None
        if operation.operand_kind is GroupOperandKind.ARRAY:
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.OPERAND_FORM,
                "direct CUB WarpReduce planning currently supports scalar operands",
            )
        if operation.cub_algorithm is not None:
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.OPERATION_VARIANT,
                "CUB algorithm selection applies to BlockReduce, not WarpReduce",
            )
        spec = make_warp_reduce_spec(
            dtype=operation.dtype,
            threads_in_warp=warp_width,
            operation=operation_name,
            reduce_operator=reduce_operator,
            valid_items=operation.valid_items,
            include_full_warp=False,
        ).specialization
        target = GroupLoweringTarget.CUB_WARP
        cpp_class = "cub::WarpReduce"
        header = "cub/warp/warp_reduce.cuh"
    contracts = _contracts(
        resolved,
        launch,
        operation,
        visibility=ResultVisibility.GROUP_ROOT,
        storage_ownership=StorageOwnership.CALLER,
        cpp_type="typename implementation_type::TempStorage",
        uniform_arguments=(
            *_stateful_operator_uniformity(operation.reduce_operator),
            *(("valid_items",) if operation.primitive.has_valid_items else ()),
        ),
        valid_member_selection=(
            "first N members by linear group rank"
            if operation.primitive.has_valid_items
            else None
        ),
        argument_preconditions=(
            (
                ArgumentPrecondition(
                    name="valid_items",
                    minimum=1,
                    maximum=resolved.static_size,
                    enforcement=(
                        PreconditionEnforcement.PLANNER_VALIDATED
                        if operation.valid_items.kind is BindingKind.STATIC
                        else PreconditionEnforcement.CALLER
                    ),
                ),
            )
            if operation.primitive.has_valid_items
            else ()
        ),
    )
    return GroupLoweringPlan(
        target=target,
        call=call,
        resolved_group=resolved,
        implementation=spec,
        participation=contracts[0],
        result=contracts[1],
        synchronization=contracts[2],
        temp_storage=contracts[3],
        provenance=ImplementationProvenance(
            library="CUB",
            header=header,
            cpp_class=cpp_class,
            method=spec.method_name,
        ),
    )


def _plan_scan(
    call: GroupPrimitiveCall,
    resolved: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupScanSemantics,
) -> GroupLoweringPlan:
    if resolved.kind not in {"block", "warp", "threads_within_warp"}:
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.GROUP_KIND,
            "group scan supports physical block, physical-warp, and "
            "logical-warp groups",
        )
    if operation.prefix_callback is not None:
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.OPERATION_VARIANT,
            "group scan prefix callbacks are not supported in the initial slice",
        )
    if (
        operation.valid_items.kind is not BindingKind.OMITTED
        and resolved.kind == "block"
    ):
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.OPERATION_VARIANT,
            "valid_items applies to WarpScan, not BlockScan",
        )
    if operation.initial_value is not None and operation.scan_operator is None:
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.OPERATION_VARIANT,
            "CUB sum scan overloads do not accept an explicit initial value; "
            "provide a scan operator",
        )
    if (
        operation.mode is GroupScanMode.EXCLUSIVE
        and operation.scan_operator is not None
        and operation.initial_value is None
    ):
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.OPERATION_VARIANT,
            "group exclusive scans with an explicit operator require an initial "
            "value because the CUB no-initial overload leaves group rank zero "
            "undefined",
        )
    assert launch.exact_block_dim is not None
    block_threads = launch.exact_block_threads
    assert block_threads is not None
    if resolved.kind in {"warp", "threads_within_warp"}:
        warp_width, width_error = _unsupported_cub_warp_width(call, resolved)
        if width_error is not None:
            return width_error
        assert warp_width is not None
        if operation.valid_items.kind is BindingKind.STATIC:
            valid_items = operation.valid_items.value
            if isinstance(valid_items, bool) or not isinstance(valid_items, int):
                raise TypeError("static valid_items must be an integer")
            if not 1 <= valid_items <= warp_width:
                raise ValueError(
                    "static valid_items must be between 1 and the logical warp size"
                )
        if operation.operand_kind is GroupOperandKind.ARRAY:
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.OPERAND_FORM,
                "CUB WarpScan is scalar-per-lane; multi-item warp scan is unsupported",
            )
        if operation.cub_algorithm is not None:
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.OPERATION_VARIANT,
                "CUB algorithm selection applies to BlockScan, not WarpScan",
            )
        spec = make_warp_scan_spec(
            dtype=operation.dtype,
            threads_in_warp=warp_width,
            mode=WarpScanMode(operation.mode.value),
            scan_operator=operation.scan_operator,
            initial_value=operation.initial_value,
            warp_aggregate=operation.aggregate,
            valid_items=operation.valid_items.kind is not BindingKind.OMITTED,
        ).specialization
        target = GroupLoweringTarget.CUB_WARP
        cpp_class = "cub::WarpScan"
        header = "cub/warp/warp_scan.cuh"
    else:
        if (
            operation.cub_algorithm is BlockScanAlgorithm.WARP_SCANS
            and block_threads % 32 != 0
        ):
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.OPERATION_VARIANT,
                "BLOCK_SCAN_WARP_SCANS requires a block size that is a multiple "
                "of the 32-thread architectural warp; CUB otherwise substitutes "
                "BLOCK_SCAN_RAKING",
            )
        if (
            operation.mode is GroupScanMode.INCLUSIVE
            and operation.operand_kind is GroupOperandKind.SCALAR
            and operation.initial_value is not None
        ):
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.OPERATION_VARIANT,
                "scalar CUB BlockScan InclusiveScan has no initial-value overload",
            )
        spec = make_block_scan_spec(
            dtype=operation.dtype,
            block_dim=launch.exact_block_dim,
            items_per_thread=operation.items_per_thread,
            mode=ScanMode(operation.mode.value),
            algorithm=operation.cub_algorithm or BlockScanAlgorithm.RAKING,
            value_kind=ScanValueKind(operation.operand_kind.value),
            scan_operator=operation.scan_operator,
            initial_value=operation.initial_value,
            block_aggregate=operation.aggregate,
        ).specialization
        target = GroupLoweringTarget.CUB_BLOCK
        cpp_class = "cub::BlockScan"
        header = "cub/block/block_scan.cuh"
    contracts = _contracts(
        resolved,
        launch,
        operation,
        visibility=ResultVisibility.PER_MEMBER,
        storage_ownership=StorageOwnership.IMPLEMENTATION,
        cpp_type=None,
        uniform_arguments=(
            *_stateful_operator_uniformity(operation.scan_operator),
            *(("initial_value",) if operation.initial_value is not None else ()),
            *(
                ("valid_items",)
                if operation.valid_items.kind is not BindingKind.OMITTED
                else ()
            ),
        ),
        valid_member_selection=(
            "first valid_items lanes"
            if operation.valid_items.kind is not BindingKind.OMITTED
            else None
        ),
        argument_preconditions=(
            (
                ArgumentPrecondition(
                    name="valid_items",
                    minimum=1,
                    maximum=resolved.static_size,
                    enforcement=(
                        PreconditionEnforcement.PLANNER_VALIDATED
                        if operation.valid_items.kind is BindingKind.STATIC
                        else PreconditionEnforcement.CALLER
                    ),
                ),
            )
            if operation.valid_items.kind is not BindingKind.OMITTED
            else ()
        ),
    )
    return GroupLoweringPlan(
        target=target,
        call=call,
        resolved_group=resolved,
        implementation=spec,
        participation=contracts[0],
        result=contracts[1],
        synchronization=contracts[2],
        temp_storage=contracts[3],
        provenance=ImplementationProvenance(
            library="CUB",
            header=header,
            cpp_class=cpp_class,
            method=spec.method_name,
        ),
    )


def _plan_exchange(
    call: GroupPrimitiveCall,
    resolved: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupExchangeSemantics,
) -> GroupLoweringPlan:
    if resolved.kind not in {"block", "warp", "threads_within_warp"}:
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.GROUP_KIND,
            "group exchange supports physical block, physical-warp, and "
            "logical-warp groups",
        )
    assert launch.exact_block_dim is not None
    if resolved.kind == "block":
        spec = make_block_exchange_spec(
            dtype=operation.dtype,
            block_dim=launch.exact_block_dim,
            items_per_thread=operation.items_per_thread,
            mode=BlockExchangeMode(operation.mode.value),
            value_form=BlockExchangeValueForm.OUT_OF_PLACE,
            warp_time_slicing=operation.primitive.warp_time_slicing,
            rank_dtype=operation.primitive.rank_dtype,
            valid_flag_dtype=operation.primitive.valid_flag_dtype,
        ).specialization
        target = GroupLoweringTarget.CUB_BLOCK
        cpp_class = "cub::BlockExchange"
        header = "cub/block/block_exchange.cuh"
    else:
        warp_width, width_error = _unsupported_cub_warp_width(call, resolved)
        if width_error is not None:
            return width_error
        assert warp_width is not None
        try:
            warp_mode = WarpExchangeMode(operation.mode.value)
        except ValueError:
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.OPERATION_VARIANT,
                f"cub::WarpExchange does not support mode {operation.mode.value!r}",
            )
        if operation.primitive.warp_time_slicing:
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.OPERATION_VARIANT,
                "warp_time_slicing applies only to BlockExchange",
            )
        if operation.primitive.valid_flag_dtype is not None:
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.OPERATION_VARIANT,
                "cub::WarpExchange does not accept valid_flags",
            )
        spec = make_warp_exchange_spec(
            dtype=operation.dtype,
            items_per_thread=operation.items_per_thread,
            threads_in_warp=warp_width,
            mode=warp_mode,
            value_form=WarpExchangeValueForm.OUT_OF_PLACE,
            rank_dtype=operation.primitive.rank_dtype,
        ).specialization
        target = GroupLoweringTarget.CUB_WARP
        cpp_class = "cub::WarpExchange"
        header = "cub/warp/warp_exchange.cuh"
    contracts = _contracts(
        resolved,
        launch,
        operation,
        visibility=ResultVisibility.PER_MEMBER,
        storage_ownership=StorageOwnership.IMPLEMENTATION,
        cpp_type=None,
    )
    return GroupLoweringPlan(
        target=target,
        call=call,
        resolved_group=resolved,
        implementation=spec,
        participation=contracts[0],
        result=contracts[1],
        synchronization=contracts[2],
        temp_storage=contracts[3],
        provenance=ImplementationProvenance(
            library="CUB",
            header=header,
            cpp_class=cpp_class,
            method=spec.method_name,
        ),
    )


def _plan_adjacent_difference(
    call: GroupPrimitiveCall,
    resolved: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupAdjacentDifferenceSemantics,
) -> GroupLoweringPlan:
    if resolved.kind != "block":
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.GROUP_KIND,
            "group adjacent_difference supports complete physical block groups",
        )
    assert launch.exact_block_dim is not None
    primitive = operation.primitive
    valid_items = RuntimeValue("valid_items") if primitive.has_partial_tile else None
    tile_predecessor_item = (
        RuntimeValue("tile_predecessor_item")
        if primitive.boundary is BlockAdjacentDifferenceBoundary.PREDECESSOR
        else None
    )
    tile_successor_item = (
        RuntimeValue("tile_successor_item")
        if primitive.boundary is BlockAdjacentDifferenceBoundary.SUCCESSOR
        else None
    )
    spec = make_block_adjacent_difference_spec(
        dtype=operation.dtype,
        block_dim=launch.exact_block_dim,
        items_per_thread=operation.items_per_thread,
        direction=operation.direction,
        difference_operator=primitive.difference_operator,
        valid_items=valid_items,
        tile_predecessor_item=tile_predecessor_item,
        tile_successor_item=tile_successor_item,
    ).specialization
    uniform_arguments = []
    if primitive.has_partial_tile:
        uniform_arguments.append("valid_items")
    if primitive.boundary is BlockAdjacentDifferenceBoundary.PREDECESSOR:
        uniform_arguments.append("tile_predecessor_item")
    elif primitive.boundary is BlockAdjacentDifferenceBoundary.SUCCESSOR:
        uniform_arguments.append("tile_successor_item")
    contracts = _contracts(
        resolved,
        launch,
        operation,
        visibility=ResultVisibility.PER_MEMBER,
        storage_ownership=StorageOwnership.IMPLEMENTATION,
        cpp_type=None,
        uniform_arguments=tuple(uniform_arguments),
        valid_member_selection=(
            "first valid_items tile elements" if primitive.has_partial_tile else None
        ),
    )
    return GroupLoweringPlan(
        target=GroupLoweringTarget.CUB_BLOCK,
        call=call,
        resolved_group=resolved,
        implementation=spec,
        participation=contracts[0],
        result=contracts[1],
        synchronization=contracts[2],
        temp_storage=contracts[3],
        provenance=ImplementationProvenance(
            library="CUB",
            header="cub/block/block_adjacent_difference.cuh",
            cpp_class="cub::BlockAdjacentDifference",
            method=spec.method_name,
        ),
    )


def _plan_merge_sort(
    call: GroupPrimitiveCall,
    resolved: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupMergeSortSemantics,
) -> GroupLoweringPlan:
    if resolved.kind not in {"block", "warp", "threads_within_warp"}:
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.GROUP_KIND,
            "group merge_sort supports complete block, physical-warp, and "
            "logical-warp groups",
        )

    primitive = operation.primitive
    assert launch.exact_block_dim is not None
    block_threads = launch.exact_block_threads
    assert block_threads is not None
    if resolved.kind == "block":
        if block_threads & (block_threads - 1):
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.OPERATION_VARIANT,
                "cub::BlockMergeSort requires a power-of-two block thread count",
            )
        runtime_valid_items = (
            RuntimeValue("valid_items") if primitive.has_partial_tile else None
        )
        runtime_oob_default = (
            RuntimeValue("oob_default") if primitive.has_partial_tile else None
        )
        spec = make_block_merge_sort_spec(
            key_dtype=primitive.key_dtype,
            value_dtype=primitive.value_dtype,
            block_dim=launch.exact_block_dim,
            items_per_thread=primitive.items_per_thread,
            compare_operator=primitive.compare_operator,
            valid_items=runtime_valid_items,
            oob_default=runtime_oob_default,
        ).specialization
        target = GroupLoweringTarget.CUB_BLOCK
        cpp_class = "cub::BlockMergeSort"
        header = "cub/block/block_merge_sort.cuh"
        tile_threads = block_threads
    else:
        logical_width = resolved.static_size
        assert logical_width is not None
        if block_threads % logical_width:
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.PARTIAL_PHYSICAL_WARP,
                "warp merge_sort requires the exact block thread count to be "
                "a multiple of the logical warp width",
            )
        spec = make_warp_merge_sort_spec(
            key_dtype=primitive.key_dtype,
            value_dtype=primitive.value_dtype,
            items_per_thread=primitive.items_per_thread,
            threads_in_warp=logical_width,
            compare_operator=primitive.compare_operator,
            valid_items=(
                RuntimeValue("valid_items") if primitive.has_partial_tile else None
            ),
            oob_default=(
                RuntimeValue("oob_default") if primitive.has_partial_tile else None
            ),
        ).specialization
        target = GroupLoweringTarget.CUB_WARP
        cpp_class = "cub::WarpMergeSort"
        header = "cub/warp/warp_merge_sort.cuh"
        tile_threads = logical_width

    argument_preconditions = ()
    uniform_arguments = ()
    valid_member_selection = None
    if primitive.has_partial_tile:
        uniform_arguments = ("valid_items", "oob_default")
        valid_member_selection = "first valid_items tile elements"
        argument_preconditions = (
            ArgumentPrecondition(
                name="valid_items",
                minimum=0,
                maximum=tile_threads * primitive.items_per_thread,
                enforcement=PreconditionEnforcement.CALLER,
            ),
        )
    contracts = _contracts(
        resolved,
        launch,
        operation,
        visibility=ResultVisibility.PER_MEMBER,
        storage_ownership=StorageOwnership.IMPLEMENTATION,
        cpp_type=None,
        uniform_arguments=uniform_arguments,
        valid_member_selection=valid_member_selection,
        argument_preconditions=argument_preconditions,
    )
    return GroupLoweringPlan(
        target=target,
        call=call,
        resolved_group=resolved,
        implementation=spec,
        participation=contracts[0],
        result=contracts[1],
        synchronization=contracts[2],
        temp_storage=contracts[3],
        provenance=ImplementationProvenance(
            library="CUB",
            header=header,
            cpp_class=cpp_class,
            method=spec.method_name,
        ),
    )


def _plan_discontinuity(
    call: GroupPrimitiveCall,
    resolved: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupDiscontinuitySemantics,
) -> GroupLoweringPlan:
    if resolved.kind != "block":
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.GROUP_KIND,
            "group discontinuity supports complete physical block groups",
        )
    assert launch.exact_block_dim is not None
    primitive = operation.primitive
    tile_predecessor_item = (
        RuntimeValue("tile_predecessor_item")
        if primitive.has_tile_predecessor
        else None
    )
    tile_successor_item = (
        RuntimeValue("tile_successor_item") if primitive.has_tile_successor else None
    )
    spec = make_block_discontinuity_spec(
        dtype=operation.dtype,
        flag_dtype=operation.flag_dtype,
        block_dim=launch.exact_block_dim,
        items_per_thread=operation.items_per_thread,
        mode=operation.mode,
        flag_operator=primitive.flag_operator,
        tile_predecessor_item=tile_predecessor_item,
        tile_successor_item=tile_successor_item,
    ).specialization
    contracts = _contracts(
        resolved,
        launch,
        operation,
        visibility=ResultVisibility.PER_MEMBER,
        storage_ownership=StorageOwnership.IMPLEMENTATION,
        cpp_type=None,
        uniform_arguments=(
            *(("tile_predecessor_item",) if primitive.has_tile_predecessor else ()),
            *(("tile_successor_item",) if primitive.has_tile_successor else ()),
        ),
    )
    return GroupLoweringPlan(
        target=GroupLoweringTarget.CUB_BLOCK,
        call=call,
        resolved_group=resolved,
        implementation=spec,
        participation=contracts[0],
        result=contracts[1],
        synchronization=contracts[2],
        temp_storage=contracts[3],
        provenance=ImplementationProvenance(
            library="CUB",
            header="cub/block/block_discontinuity.cuh",
            cpp_class="cub::BlockDiscontinuity",
            method=spec.method_name,
        ),
    )


def _plan_shuffle(
    call: GroupPrimitiveCall,
    resolved: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupShuffleSemantics,
) -> GroupLoweringPlan:
    if resolved.kind != "block":
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.GROUP_KIND,
            "group shuffle supports complete physical block groups",
        )
    primitive = operation.primitive
    is_array = primitive.value_kind is BlockShuffleValueKind.ARRAY
    if is_array and primitive.mode not in {
        BlockShuffleMode.UP,
        BlockShuffleMode.DOWN,
    }:
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.OPERATION_VARIANT,
            "public CUB ThreadData shuffle supports only unit-shift Up and Down",
        )
    if not is_array and primitive.mode not in {
        BlockShuffleMode.OFFSET,
        BlockShuffleMode.ROTATE,
    }:
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.OPERATION_VARIANT,
            "public CUB scalar shuffle supports only Offset and Rotate",
        )
    assert launch.exact_block_dim is not None
    spec = make_block_shuffle_spec(
        dtype=operation.dtype,
        block_dim=launch.exact_block_dim,
        mode=primitive.mode,
        items_per_thread=primitive.items_per_thread,
        distance=primitive.distance,
        block_prefix=primitive.block_prefix,
        block_suffix=primitive.block_suffix,
    ).specialization
    contracts = _contracts(
        resolved,
        launch,
        operation,
        visibility=ResultVisibility.PER_MEMBER,
        storage_ownership=StorageOwnership.IMPLEMENTATION,
        cpp_type=None,
        uniform_arguments=(
            ("distance",) if primitive.distance.kind is BindingKind.RUNTIME else ()
        ),
    )
    return GroupLoweringPlan(
        target=GroupLoweringTarget.CUB_BLOCK,
        call=call,
        resolved_group=resolved,
        implementation=spec,
        participation=contracts[0],
        result=contracts[1],
        synchronization=contracts[2],
        temp_storage=contracts[3],
        provenance=ImplementationProvenance(
            library="CUB",
            header="cub/block/block_shuffle.cuh",
            cpp_class="cub::BlockShuffle",
            method=spec.method_name,
        ),
    )


def _plan_histogram(
    call: GroupPrimitiveCall,
    resolved: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupHistogramSemantics,
) -> GroupLoweringPlan:
    if resolved.kind != "block":
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.GROUP_KIND,
            "group histogram supports complete physical block groups",
        )
    assert launch.exact_block_dim is not None
    primitive = operation.primitive
    assert primitive.bins is not None
    assert resolved.static_size is not None
    validate_block_histogram_output_capacity(
        bins=primitive.bins,
        bins_per_thread=operation.bins_per_thread,
        block_threads=resolved.static_size,
    )
    spec = make_block_histogram_spec(
        item_dtype=primitive.item_dtype,
        counter_dtype=primitive.counter_dtype,
        block_dim=launch.exact_block_dim,
        items_per_thread=primitive.items_per_thread,
        bins=primitive.bins,
        algorithm=primitive.algorithm,
        operation=BlockHistogramOperation.HISTOGRAM,
    ).specialization
    contracts = _contracts(
        resolved,
        launch,
        operation,
        visibility=ResultVisibility.PER_MEMBER,
        storage_ownership=StorageOwnership.IMPLEMENTATION,
        cpp_type=None,
        argument_preconditions=(
            ArgumentPrecondition(
                name="samples",
                minimum=0,
                maximum=primitive.bins - 1,
                enforcement=PreconditionEnforcement.CALLER,
            ),
        ),
    )
    return GroupLoweringPlan(
        target=GroupLoweringTarget.CUB_BLOCK,
        call=call,
        resolved_group=resolved,
        implementation=spec,
        participation=contracts[0],
        result=contracts[1],
        synchronization=contracts[2],
        temp_storage=contracts[3],
        provenance=ImplementationProvenance(
            library="CUB",
            header="cub/block/block_histogram.cuh",
            cpp_class="cub::BlockHistogram",
            method=spec.method_name,
        ),
    )


def _plan_run_length_decode(
    call: GroupPrimitiveCall,
    resolved: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupRunLengthDecodeSemantics,
) -> GroupLoweringPlan:
    if resolved.kind != "block":
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.GROUP_KIND,
            "group run-length decode supports complete physical block groups",
        )
    assert launch.exact_block_dim is not None
    primitive = operation.primitive
    assert primitive.run_length_dtype is not None
    assert primitive.total_decoded_size_dtype is not None
    spec = make_block_run_length_decode_spec(
        item_dtype=primitive.item_dtype,
        run_length_dtype=primitive.run_length_dtype,
        decoded_offset_dtype=primitive.decoded_offset_dtype,
        total_decoded_size_dtype=primitive.total_decoded_size_dtype,
        relative_offset_dtype=primitive.relative_offset_dtype,
        block_dim=launch.exact_block_dim,
        runs_per_thread=primitive.runs_per_thread,
        decoded_items_per_thread=primitive.decoded_items_per_thread,
        stage=BlockRunLengthDecodeStage.FUSED,
        with_relative_offsets=primitive.has_relative_offsets,
        with_decoded_window_offset=True,
        returns_total_decoded_size=True,
    ).specialization
    contracts = _contracts(
        resolved,
        launch,
        operation,
        visibility=ResultVisibility.PER_MEMBER,
        storage_ownership=StorageOwnership.IMPLEMENTATION,
        cpp_type=None,
        uniform_arguments=("decoded_window_offset",),
        argument_preconditions=(
            ArgumentPrecondition(
                name="run_lengths",
                minimum=0,
                maximum=None,
                enforcement=PreconditionEnforcement.CALLER,
            ),
            ArgumentPrecondition(
                name="sum(run_lengths)",
                minimum=1,
                maximum=None,
                enforcement=PreconditionEnforcement.CALLER,
            ),
            ArgumentPrecondition(
                name="decoded_window_offset",
                minimum=0,
                maximum=None,
                enforcement=PreconditionEnforcement.CALLER,
            ),
        ),
    )
    results = [
        LogicalResultContract(
            name="decoded_items",
            dtype=primitive.item_dtype,
            visibility=ResultVisibility.PER_MEMBER,
            ownership=ResultOwnership.EACH_MEMBER,
            operand_kind=GroupOperandKind.ARRAY,
            items_per_member=primitive.decoded_items_per_thread,
        )
    ]
    if primitive.has_relative_offsets:
        results.append(
            LogicalResultContract(
                name="relative_offsets",
                dtype=primitive.relative_offset_dtype,
                visibility=ResultVisibility.PER_MEMBER,
                ownership=ResultOwnership.EACH_MEMBER,
                operand_kind=GroupOperandKind.ARRAY,
                items_per_member=primitive.decoded_items_per_thread,
            )
        )
    results.append(
        LogicalResultContract(
            name="total_decoded_size",
            dtype=primitive.total_decoded_size_dtype,
            visibility=ResultVisibility.ALL_MEMBERS,
            ownership=ResultOwnership.EACH_MEMBER,
            operand_kind=GroupOperandKind.SCALAR,
            items_per_member=1,
        )
    )
    return GroupLoweringPlan(
        target=GroupLoweringTarget.CUB_BLOCK,
        call=call,
        resolved_group=resolved,
        implementation=spec,
        participation=contracts[0],
        result=ResultContract(tuple(results)),
        synchronization=contracts[2],
        temp_storage=contracts[3],
        provenance=ImplementationProvenance(
            library="CUB",
            header="cub/block/block_run_length_decode.cuh",
            cpp_class="cub::BlockRunLengthDecode",
            method=primitive.decode_method_name,
        ),
    )


_CUB_RADIX_MAX_TILE_ITEMS = (1 << 16) - 1


def _radix_result(
    *,
    name: str,
    dtype: Any,
    operand_kind: GroupOperandKind,
    items_per_member: int,
) -> LogicalResultContract:
    return LogicalResultContract(
        name=name,
        dtype=dtype,
        visibility=ResultVisibility.PER_MEMBER,
        ownership=ResultOwnership.EACH_MEMBER,
        operand_kind=operand_kind,
        items_per_member=items_per_member,
    )


def _radix_contracts(
    resolved: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupRadixSortSemantics | GroupRadixRankSemantics,
    *,
    uniform_arguments: tuple[str, ...] = (),
) -> tuple[
    ParticipationContract,
    SynchronizationContract,
    TempStorageContract,
]:
    contracts = _contracts(
        resolved,
        launch,
        operation,
        visibility=ResultVisibility.PER_MEMBER,
        storage_ownership=StorageOwnership.IMPLEMENTATION,
        cpp_type=None,
        uniform_arguments=uniform_arguments,
        returns_value=False,
    )
    return contracts[0], contracts[2], contracts[3]


def _radix_tile_failure(
    call: GroupPrimitiveCall,
    resolved: ThreadGroup,
    *,
    block_threads: int,
    items_per_thread: int,
) -> GroupLoweringPlan | None:
    tile_items = block_threads * items_per_thread
    if tile_items <= _CUB_RADIX_MAX_TILE_ITEMS:
        return None
    return _unsupported(
        call,
        resolved,
        UnsupportedReasonCode.OPERATION_VARIANT,
        "CUB block radix collectives require block_threads * items_per_thread "
        f"<= {_CUB_RADIX_MAX_TILE_ITEMS}; received {tile_items}",
    )


def _plan_radix_sort(
    call: GroupPrimitiveCall,
    resolved: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupRadixSortSemantics,
) -> GroupLoweringPlan:
    if resolved.kind != "block":
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.GROUP_KIND,
            "group radix sort supports complete physical block groups",
        )
    assert launch.exact_block_dim is not None
    assert launch.exact_block_threads is not None
    failure = _radix_tile_failure(
        call,
        resolved,
        block_threads=launch.exact_block_threads,
        items_per_thread=operation.items_per_thread,
    )
    if failure is not None:
        return failure
    primitive = operation.primitive
    bit_width = None if primitive.bit_range is None else primitive.bit_range.bit_width
    spec = make_block_radix_sort_spec(
        key_dtype=primitive.key_dtype,
        value_dtype=primitive.value_dtype,
        block_dim=launch.exact_block_dim,
        items_per_thread=primitive.items_per_thread,
        descending=primitive.order,
        blocked_to_striped=False,
        begin_bit=RuntimeValue("begin_bit"),
        end_bit=RuntimeValue("end_bit"),
        key_bit_width=bit_width,
        bit_policy=BlockRadixSortBitPolicy.EXPLICIT,
    ).specialization
    participation, synchronization, temp_storage = _radix_contracts(
        resolved,
        launch,
        operation,
        uniform_arguments=("begin_bit", "end_bit"),
    )
    result_values = [
        _radix_result(
            name="keys",
            dtype=primitive.key_dtype,
            operand_kind=operation.operand_kind,
            items_per_member=primitive.items_per_thread,
        )
    ]
    if primitive.has_values:
        result_values.append(
            _radix_result(
                name="values",
                dtype=primitive.value_dtype,
                operand_kind=operation.operand_kind,
                items_per_member=primitive.items_per_thread,
            )
        )
    return GroupLoweringPlan(
        target=GroupLoweringTarget.CUB_BLOCK,
        call=call,
        resolved_group=resolved,
        implementation=spec,
        participation=participation,
        result=ResultContract(tuple(result_values)),
        synchronization=synchronization,
        temp_storage=temp_storage,
        provenance=ImplementationProvenance(
            library="CUB",
            header="cub/block/block_radix_sort.cuh",
            cpp_class="cub::BlockRadixSort",
            method=spec.method_name,
        ),
    )


def _plan_radix_rank(
    call: GroupPrimitiveCall,
    resolved: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupRadixRankSemantics,
) -> GroupLoweringPlan:
    if resolved.kind != "block":
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.GROUP_KIND,
            "group radix rank supports complete physical block groups",
        )
    assert launch.exact_block_dim is not None
    assert launch.exact_block_threads is not None
    failure = _radix_tile_failure(
        call,
        resolved,
        block_threads=launch.exact_block_threads,
        items_per_thread=operation.items_per_thread,
    )
    if failure is not None:
        return failure
    primitive = operation.primitive
    begin_bit = primitive.bit_range.static_begin_bit
    end_bit = primitive.bit_range.static_end_bit
    assert begin_bit is not None and end_bit is not None
    spec = make_block_radix_rank_spec(
        key_dtype=primitive.key_dtype,
        block_dim=launch.exact_block_dim,
        items_per_thread=primitive.items_per_thread,
        begin_bit=begin_bit,
        end_bit=end_bit,
        key_bit_width=primitive.bit_range.bit_width,
        descending=primitive.order,
        with_exclusive_digit_prefix=primitive.has_exclusive_digit_prefix,
    ).specialization
    participation, synchronization, temp_storage = _radix_contracts(
        resolved,
        launch,
        operation,
    )
    result_values = [
        _radix_result(
            name="ranks",
            dtype=INT32,
            operand_kind=operation.operand_kind,
            items_per_member=primitive.items_per_thread,
        )
    ]
    if primitive.has_exclusive_digit_prefix:
        prefix_items = primitive.exclusive_digit_prefix_items_per_thread
        assert prefix_items is not None
        result_values.append(
            _radix_result(
                name="exclusive_digit_prefix",
                dtype=INT32,
                operand_kind=GroupOperandKind.ARRAY,
                items_per_member=prefix_items,
            )
        )
    return GroupLoweringPlan(
        target=GroupLoweringTarget.CUB_BLOCK,
        call=call,
        resolved_group=resolved,
        implementation=spec,
        participation=participation,
        result=ResultContract(tuple(result_values)),
        synchronization=synchronization,
        temp_storage=temp_storage,
        provenance=ImplementationProvenance(
            library="CUB",
            header="cub/block/block_radix_rank.cuh",
            cpp_class="cub::BlockRadixRank",
            method=spec.method_name,
        ),
    )


def _plan_load_store(
    call: GroupPrimitiveCall,
    resolved: ThreadGroup,
    launch: LaunchFacts,
    operation: GroupLoadStoreSemantics,
) -> GroupLoweringPlan:
    if resolved.kind not in {"block", "warp", "threads_within_warp"}:
        return _unsupported(
            call,
            resolved,
            UnsupportedReasonCode.GROUP_KIND,
            "group load/store supports complete physical block, physical-warp, "
            "and logical-warp groups",
        )
    assert launch.exact_block_dim is not None
    if resolved.kind == "block":
        algorithm = BlockLoadStoreAlgorithm(operation.algorithm.value)
        make_spec = (
            make_block_load_spec
            if operation.kind is GroupLoadStoreKind.LOAD
            else make_block_store_spec
        )
        spec = make_spec(
            dtype=operation.dtype,
            block_dim=launch.exact_block_dim,
            items_per_thread=operation.items_per_thread,
            algorithm=algorithm,
            valid_items=operation.has_valid_items,
            oob_default=operation.has_oob_default,
            include_full_tile=False,
            include_pointer_offset=operation.has_offset,
        ).specialization
        target = GroupLoweringTarget.CUB_BLOCK
        cpp_class = (
            "cub::BlockLoad"
            if operation.kind is GroupLoadStoreKind.LOAD
            else "cub::BlockStore"
        )
        header = f"cub/block/block_{operation.kind.value}.cuh"
    else:
        warp_width, width_error = _unsupported_cub_warp_width(call, resolved)
        if width_error is not None:
            return width_error
        assert warp_width is not None
        try:
            algorithm = WarpLoadStoreAlgorithm(operation.algorithm.value)
        except ValueError:
            return _unsupported(
                call,
                resolved,
                UnsupportedReasonCode.OPERATION_VARIANT,
                f"cub::Warp{operation.kind.value.title()} does not support "
                f"algorithm {operation.algorithm.value!r}",
            )
        make_spec = (
            make_warp_load_spec
            if operation.kind is GroupLoadStoreKind.LOAD
            else make_warp_store_spec
        )
        spec = make_spec(
            dtype=operation.dtype,
            items_per_thread=operation.items_per_thread,
            threads_in_warp=warp_width,
            algorithm=algorithm,
            valid_items=operation.has_valid_items,
            oob_default=operation.has_oob_default,
            include_full_tile=False,
        ).specialization
        target = GroupLoweringTarget.CUB_WARP
        cpp_class = (
            "cub::WarpLoad"
            if operation.kind is GroupLoadStoreKind.LOAD
            else "cub::WarpStore"
        )
        header = f"cub/warp/warp_{operation.kind.value}.cuh"
    contracts = _contracts(
        resolved,
        launch,
        operation,
        visibility=ResultVisibility.PER_MEMBER,
        storage_ownership=StorageOwnership.IMPLEMENTATION,
        cpp_type=None,
        uniform_arguments=(
            *(("valid_items",) if operation.has_valid_items else ()),
            *(("oob_default",) if operation.has_oob_default else ()),
            *(("offset",) if operation.has_offset else ()),
        ),
        valid_member_selection=(
            "first valid_items tile elements" if operation.has_valid_items else None
        ),
        returns_value=operation.kind is GroupLoadStoreKind.LOAD,
    )
    return GroupLoweringPlan(
        target=target,
        call=call,
        resolved_group=resolved,
        implementation=spec,
        participation=contracts[0],
        result=contracts[1],
        synchronization=contracts[2],
        temp_storage=contracts[3],
        provenance=ImplementationProvenance(
            library="CUB",
            header=header,
            cpp_class=cpp_class,
            method=spec.method_name,
        ),
    )


def plan_group_primitive(
    call: GroupPrimitiveCall,
    launch: LaunchFacts,
) -> GroupLoweringPlan:
    """Resolve a compile-time group call to an official CUDAX/CUB target."""

    if not isinstance(call, GroupPrimitiveCall):
        raise TypeError("call must be a GroupPrimitiveCall")
    if not isinstance(launch, LaunchFacts):
        raise TypeError("launch must be LaunchFacts")
    cluster_dim = launch.exact_cluster_dim
    uses_multi_block_cluster = cluster_dim is not None and cluster_dim != (1, 1, 1)
    if call.group.kind in {"cluster", "grid"} and uses_multi_block_cluster:
        if launch.cluster_launch is not True or not launch.is_verified(
            "cluster_launch"
        ):
            return _unsupported(
                call,
                call.group,
                UnsupportedReasonCode.LAUNCH_CAPABILITY,
                "multi-block cluster lowering requires verified cluster launch "
                f"capability; observed {launch.cluster_launch!r} with verified="
                f"{launch.is_verified('cluster_launch')!r}",
            )
    if call.group.kind == "grid":
        if launch.cooperative_launch is not True or not launch.is_verified(
            "cooperative_launch"
        ):
            return _unsupported(
                call,
                call.group,
                UnsupportedReasonCode.LAUNCH_CAPABILITY,
                "grid group lowering requires verified cooperative launch "
                f"capability; observed {launch.cooperative_launch!r} with "
                f"verified={launch.is_verified('cooperative_launch')!r}",
            )
    resolved, failure = _resolve_group(call, launch)
    if failure is not None:
        return failure
    operation = call.operation
    if isinstance(operation, GroupReduceSemantics):
        return _plan_reduce(call, resolved, launch, operation)
    if isinstance(operation, GroupScanSemantics):
        return _plan_scan(call, resolved, launch, operation)
    if isinstance(operation, GroupAdjacentDifferenceSemantics):
        return _plan_adjacent_difference(call, resolved, launch, operation)
    if isinstance(operation, GroupMergeSortSemantics):
        return _plan_merge_sort(call, resolved, launch, operation)
    if isinstance(operation, GroupDiscontinuitySemantics):
        return _plan_discontinuity(call, resolved, launch, operation)
    if isinstance(operation, GroupShuffleSemantics):
        return _plan_shuffle(call, resolved, launch, operation)
    if isinstance(operation, GroupHistogramSemantics):
        return _plan_histogram(call, resolved, launch, operation)
    if isinstance(operation, GroupRunLengthDecodeSemantics):
        return _plan_run_length_decode(call, resolved, launch, operation)
    if isinstance(operation, GroupRadixSortSemantics):
        return _plan_radix_sort(call, resolved, launch, operation)
    if isinstance(operation, GroupRadixRankSemantics):
        return _plan_radix_rank(call, resolved, launch, operation)
    if isinstance(operation, GroupExchangeSemantics):
        return _plan_exchange(call, resolved, launch, operation)
    return _plan_load_store(call, resolved, launch, operation)


__all__ = [
    "ArgumentPrecondition",
    "CudaxCallDescription",
    "CudaxReturnKind",
    "GroupAdjacentDifferenceSemantics",
    "GroupDiscontinuitySemantics",
    "GroupExchangeMode",
    "GroupExchangeSemantics",
    "GroupHistogramSemantics",
    "GroupRunLengthDecodeSemantics",
    "GroupLoweringPlan",
    "GroupLoweringTarget",
    "GroupLoadStoreAlgorithm",
    "GroupLoadStoreKind",
    "GroupLoadStoreSemantics",
    "GroupMergeSortSemantics",
    "GroupOperandKind",
    "GroupOperationSemantics",
    "GroupPrimitiveCall",
    "GroupReduceSemantics",
    "GroupRadixRankSemantics",
    "GroupRadixSortSemantics",
    "GroupScanMode",
    "GroupScanSemantics",
    "GroupShuffleSemantics",
    "ImplementationProvenance",
    "LogicalResultContract",
    "ParticipationContract",
    "PreconditionEnforcement",
    "ResultContract",
    "ResultOwnership",
    "ResultVisibility",
    "StorageOwnership",
    "SynchronizationContract",
    "SynchronizationScope",
    "TempStorageContract",
    "ThreadGroupResolution",
    "UnsupportedReason",
    "UnsupportedReasonCode",
    "make_group_primitive_call",
    "plan_group_primitive",
    "resolve_thread_group",
]
