# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable cooperative primitives shared by certified CUDA Python DSLs.

This is one backend-neutral static contract. Automatic DSL registration or a
qualified backend import selects the lowering while an editor continues to show
this portable intersection. Import ``cuda.coop.cutlass`` or
``cuda.coop.numba_mlir`` for backend-only completion.
"""

from typing import Any, Generic, Literal, TypeAlias, overload

from typing_extensions import Self, TypeVar

from ._typing import ExchangeMode as _ExchangeMode
from ._typing import HistogramAlgorithm as _HistogramAlgorithm
from ._typing import PortableShuffleMode as _PortableShuffleMode
from ._typing import ReduceAlgorithm as _ReduceAlgorithm
from ._typing import ReduceOperator as _ReduceOperator
from ._typing import ScanAlgorithm as _ScanAlgorithm
from ._typing import ScanOperator as _ScanOperator
from ._typing import TempStorageLike as TempStorageLike
from ._typing import TempStorageSharing as _TempStorageSharing
from ._typing import ThreadDataLike as ThreadDataLike
from ._typing import ThreadGroupKind as _ThreadGroupKind
from ._typing import ThreadLevel as _ThreadLevel
from ._typing import _BlockLoadStoreAlgorithm as _BlockLoadStoreAlgorithm
from ._typing import _IntegerValue as _IntegerValue
from ._typing import _NonSumScanOperator as _NonSumScanOperator
from ._typing import _PortableIntegerKey as _PortableIntegerKey
from ._typing import _PortableIntegerValue as _PortableIntegerValue
from ._typing import _PortableNumericScalar as _PortableNumericScalar
from ._typing import _PortableRunLength as _PortableRunLength
from ._typing import _PortableRunValue as _PortableRunValue
from ._typing import _PortableThreadDataLike as _PortableThreadDataLike
from ._typing import _SumScanOperator as _SumScanOperator
from ._typing import _SynchronizableGroupKind as _SynchronizableGroupKind
from ._typing import _TraceInteger as _TraceInteger
from ._typing import _ValidItems as _ValidItems
from ._typing import _WarpLoadStoreAlgorithm as _WarpLoadStoreAlgorithm

_ItemT = TypeVar("_ItemT", bound=_PortableNumericScalar)
_PortableNumericT = TypeVar("_PortableNumericT", bound=_PortableNumericScalar)
_ScalarT = TypeVar("_ScalarT", bound=_PortableNumericScalar)
_CounterT = TypeVar("_CounterT", bound=_PortableIntegerKey)
_GroupKindT_co = TypeVar(
    "_GroupKindT_co",
    bound=_ThreadGroupKind,
    covariant=True,
    default=_ThreadGroupKind,
)
_ScalarValueT = TypeVar("_ScalarValueT", bound=_PortableNumericScalar)
_IntegerKeyT = TypeVar("_IntegerKeyT", bound=_PortableIntegerKey)
_HistogramSampleT = TypeVar("_HistogramSampleT", bound=_PortableIntegerValue)
_RunValueT = TypeVar("_RunValueT", bound=_PortableRunValue)
_RunLengthT = TypeVar("_RunLengthT", bound=_PortableRunLength)

__version__: str

class ThreadHierarchy:
    """CUDA hierarchy descriptor resolved from the current kernel launch."""

    block_dim: tuple[int, int, int] | None
    grid_dim: tuple[int, int, int] | None
    cluster_dim: tuple[int, int, int] | None
    implicit: bool

    def __init__(self) -> None:
        """Describe the compiler's current launch hierarchy."""

    @classmethod
    def current(cls) -> Self:
        """Describe the compiler's current launch hierarchy."""

    @property
    def is_static(self) -> bool:
        """Return whether the hierarchy carries static dimensions."""

    @property
    def block_thread_count(self) -> int | None:
        """Return the statically known CTA size, if any."""

Hierarchy: TypeAlias = ThreadHierarchy

class ThreadGroup(Generic[_GroupKindT_co]):
    """Common compiler-facing contract for one CUDA thread group."""

    hierarchy: ThreadHierarchy
    block_dim: tuple[int, int, int] | None
    parent: ThreadGroup | None
    source: str

    @property
    def kind(self) -> _GroupKindT_co:
        """Return the kind carried by this group descriptor."""

    @property
    def static_size(self) -> int | None:
        """Return the statically known group extent, if any."""

    @property
    def is_current(self) -> bool:
        """Return whether this descriptor refers to the current hierarchy."""

    @property
    def is_static(self) -> bool:
        """Return whether dimensions needed by this group are static."""

    def rank(self, level: _ThreadLevel = "thread") -> _IntegerValue:
        """Return this group's rank relative to another hierarchy level.

        Outside compilation, raises ``CoopCompilerContextRequiredError`` with
        ``cuda.coop.ThreadGroup.rank requires a Python DSL compiler context
        (compiler-owned activation) or a qualified backend import before
        compilation; for example, import cuda.coop.cutlass or
        cuda.coop.numba_mlir``.
        """

    def count(self, level: _ThreadLevel = "thread") -> _IntegerValue:
        """Return this group's count relative to another hierarchy level.

        Outside compilation, raises ``CoopCompilerContextRequiredError`` with
        ``cuda.coop.ThreadGroup.count requires a Python DSL compiler context
        (compiler-owned activation) or a qualified backend import before
        compilation; for example, import cuda.coop.cutlass or
        cuda.coop.numba_mlir``.
        """

    @overload
    def rank_as(self, dtype: type[_ItemT], level: _ThreadLevel = "thread") -> _ItemT:
        """Return rank converted to a dtype.

        Outside compilation, raises ``CoopCompilerContextRequiredError`` with
        ``cuda.coop.ThreadGroup.rank_as requires a Python DSL compiler context
        (compiler-owned activation) or a qualified backend import before
        compilation; for example, import cuda.coop.cutlass or
        cuda.coop.numba_mlir``.
        """
    @overload
    def rank_as(self, dtype: None = None, level: _ThreadLevel = "thread") -> Any:
        """Omit dtype or use an ``Any``-typed external compiler dtype token.

        Outside compilation, raises ``CoopCompilerContextRequiredError`` with
        ``cuda.coop.ThreadGroup.rank_as requires a Python DSL compiler context
        (compiler-owned activation) or a qualified backend import before
        compilation; for example, import cuda.coop.cutlass or
        cuda.coop.numba_mlir``.
        """

    @overload
    def count_as(self, dtype: type[_ItemT], level: _ThreadLevel = "thread") -> _ItemT:
        """Return count converted to a dtype.

        Outside compilation, raises ``CoopCompilerContextRequiredError`` with
        ``cuda.coop.ThreadGroup.count_as requires a Python DSL compiler context
        (compiler-owned activation) or a qualified backend import before
        compilation; for example, import cuda.coop.cutlass or
        cuda.coop.numba_mlir``.
        """
    @overload
    def count_as(self, dtype: None = None, level: _ThreadLevel = "thread") -> Any:
        """Omit dtype or use an ``Any``-typed external compiler dtype token.

        Outside compilation, raises ``CoopCompilerContextRequiredError`` with
        ``cuda.coop.ThreadGroup.count_as requires a Python DSL compiler context
        (compiler-owned activation) or a qualified backend import before
        compilation; for example, import cuda.coop.cutlass or
        cuda.coop.numba_mlir``.
        """

    def sync(self: ThreadGroup[_SynchronizableGroupKind]) -> None:
        """Synchronize the group's members.

        Grid synchronization is unavailable through the common V1 profile.

        Outside compilation, raises ``CoopCompilerContextRequiredError`` with
        ``cuda.coop.ThreadGroup.sync requires a Python DSL compiler context
        (compiler-owned activation) or a qualified backend import before
        compilation; for example, import cuda.coop.cutlass or
        cuda.coop.numba_mlir``.
        """

    def sync_aligned(self: ThreadGroup[_SynchronizableGroupKind]) -> None:
        """Synchronize an aligned group.

        Grid synchronization is unavailable through the common V1 profile.

        Outside compilation, raises ``CoopCompilerContextRequiredError`` with
        ``cuda.coop.ThreadGroup.sync_aligned requires a Python DSL compiler
        context (compiler-owned activation) or a qualified backend import before
        compilation; for example, import cuda.coop.cutlass or
        cuda.coop.numba_mlir``.
        """

    @overload
    def group_by(
        self: ThreadGroup[Literal["warp"]],
        count: int,
        *,
        exhaustive: bool = True,
    ) -> ThreadGroup[Literal["threads_within_warp"]]:
        """Partition a physical warp into groups of threads."""
    @overload
    def group_by(
        self: ThreadGroup[Literal["block"]],
        count: int,
        *,
        exhaustive: bool = True,
    ) -> ThreadGroup[Literal["warps_within_block"]]:
        """Partition a block into groups of physical warps."""
    def is_member(self) -> _IntegerValue:
        """Return whether the current thread belongs to this group.

        Outside compilation, raises ``CoopCompilerContextRequiredError`` with
        ``cuda.coop.ThreadGroup.is_member requires a Python DSL compiler context
        (compiler-owned activation) or a qualified backend import before
        compilation; for example, import cuda.coop.cutlass or
        cuda.coop.numba_mlir``.
        """

_MemoryGroup: TypeAlias = ThreadGroup[Literal["warp", "threads_within_warp", "block"]]
_ReductionGroup: TypeAlias = ThreadGroup[
    Literal["thread", "warp", "threads_within_warp", "block", "cluster"]
]
_BlockGroup: TypeAlias = ThreadGroup[Literal["block"]]
_WarpGroup: TypeAlias = ThreadGroup[Literal["warp", "threads_within_warp"]]

def this_thread() -> ThreadGroup[Literal["thread"]]:
    """Describe the current thread."""

def this_warp() -> ThreadGroup[Literal["warp"]]:
    """Describe the current physical warp."""

def this_block() -> ThreadGroup[Literal["block"]]:
    """Describe the current CUDA thread block."""

def this_cluster() -> ThreadGroup[Literal["cluster"]]:
    """Describe the current thread-block cluster."""

def this_grid() -> ThreadGroup[Literal["grid"]]:
    """Describe the current grid; grid collectives are not in common V1."""

@overload
def ThreadData(
    items_per_thread: int,
    dtype: type[_PortableNumericT],
) -> ThreadDataLike[_PortableNumericT]:
    """Construct a payload with a portable or structural compiler numeric dtype."""

@overload
def ThreadData(
    items_per_thread: int,
    dtype: None = None,
) -> ThreadDataLike[Any]:
    """Omit dtype or use an ``Any``-typed external compiler dtype token."""

def TempStorage(
    size_in_bytes: int | None = None,
    alignment: int | None = None,
    auto_sync: bool | None = None,
    sharing: _TempStorageSharing = "shared",
) -> TempStorageLike:
    """Construct the selected backend's explicit scratch descriptor."""

@overload
def load(
    group: _BlockGroup,
    source: object,
    output: ThreadDataLike[_PortableNumericT],
    /,
    *,
    algorithm: _BlockLoadStoreAlgorithm = "direct",
    valid_items: _ValidItems | None = None,
    oob_default: None = None,
    offset: _IntegerValue | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Populate and return ``output`` with one cooperative block tile."""

@overload
def load(
    group: _BlockGroup,
    source: object,
    output: ThreadDataLike[_PortableNumericT],
    /,
    *,
    algorithm: _BlockLoadStoreAlgorithm = "direct",
    valid_items: _ValidItems,
    oob_default: _PortableNumericT,
    offset: _IntegerValue | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Populate a partial block tile and fill invalid items."""

@overload
def load(
    group: _WarpGroup,
    source: object,
    output: ThreadDataLike[_PortableNumericT],
    /,
    *,
    algorithm: _WarpLoadStoreAlgorithm = "direct",
    valid_items: _ValidItems | None = None,
    oob_default: None = None,
    offset: _IntegerValue | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Populate and return ``output`` with a physical- or logical-warp tile."""

@overload
def load(
    group: _WarpGroup,
    source: object,
    output: ThreadDataLike[_PortableNumericT],
    /,
    *,
    algorithm: _WarpLoadStoreAlgorithm = "direct",
    valid_items: _ValidItems,
    oob_default: _PortableNumericT,
    offset: _IntegerValue | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Populate a partial physical- or logical-warp tile and fill invalid items."""

@overload
def store(
    group: _BlockGroup,
    destination: object,
    value: _PortableNumericScalar | _PortableThreadDataLike,
    /,
    *,
    algorithm: _BlockLoadStoreAlgorithm = "direct",
    valid_items: _ValidItems | None = None,
    offset: _IntegerValue | None = None,
    temp_storage: TempStorageLike | None = None,
) -> None:
    """Store one scalar or per-thread payload cooperatively across a block."""

@overload
def store(
    group: _WarpGroup,
    destination: object,
    value: _PortableNumericScalar | _PortableThreadDataLike,
    /,
    *,
    algorithm: _WarpLoadStoreAlgorithm = "direct",
    valid_items: _ValidItems | None = None,
    offset: _IntegerValue | None = None,
    temp_storage: TempStorageLike | None = None,
) -> None:
    """Store one scalar or per-thread payload across a physical or logical warp."""

@overload
def reduce(
    group: _ReductionGroup,
    value: ThreadDataLike[_ItemT],
    /,
    *,
    binary_op: _ReduceOperator | None = None,
    broadcast: bool = True,
    valid_items: None = None,
    algorithm: None = None,
) -> _ItemT:
    """Reduce a full-group payload and return one item of its element type."""

@overload
def reduce(
    group: _ReductionGroup,
    value: _ScalarValueT,
    /,
    *,
    binary_op: _ReduceOperator | None = None,
    broadcast: bool = True,
    valid_items: None = None,
    algorithm: None = None,
) -> _ScalarValueT:
    """Reduce full-group scalar values while preserving their static type."""

@overload
def reduce(
    group: _BlockGroup,
    value: _ScalarValueT,
    /,
    *,
    binary_op: _ReduceOperator | None = None,
    broadcast: Literal[False],
    valid_items: _ValidItems,
    algorithm: _ReduceAlgorithm | None = None,
) -> _ScalarValueT:
    """Reduce a scalar through direct CUB BlockReduce at the block root.

    ``valid_items`` accepts Python, NumPy, and structural compiler integers.
    """

@overload
def reduce(
    group: _BlockGroup,
    value: _ScalarValueT,
    /,
    *,
    binary_op: _ReduceOperator | None = None,
    broadcast: Literal[False],
    valid_items: None = None,
    algorithm: _ReduceAlgorithm,
) -> _ScalarValueT:
    """Reduce a scalar with an explicit direct CUB BlockReduce algorithm."""

@overload
def reduce(
    group: _WarpGroup,
    value: _ScalarValueT,
    /,
    *,
    binary_op: _ReduceOperator | None = None,
    broadcast: Literal[False],
    valid_items: _ValidItems,
    algorithm: None = None,
) -> _ScalarValueT:
    """Reduce a valid scalar prefix through direct CUB WarpReduce.

    ``valid_items`` accepts Python, NumPy, and structural compiler integers.
    """

@overload
def sum(
    group: _ReductionGroup,
    value: ThreadDataLike[_ItemT],
    /,
    *,
    broadcast: bool = True,
    valid_items: None = None,
    algorithm: None = None,
) -> _ItemT:
    """Sum a full-group payload and return one item of its element type."""

@overload
def sum(
    group: _ReductionGroup,
    value: _ScalarValueT,
    /,
    *,
    broadcast: bool = True,
    valid_items: None = None,
    algorithm: None = None,
) -> _ScalarValueT:
    """Sum full-group scalar values while preserving their static type."""

@overload
def sum(
    group: _BlockGroup,
    value: _ScalarValueT,
    /,
    *,
    broadcast: Literal[False],
    valid_items: _ValidItems,
    algorithm: _ReduceAlgorithm | None = None,
) -> _ScalarValueT:
    """Sum a scalar through direct CUB BlockReduce at the block root.

    ``valid_items`` accepts Python, NumPy, and structural compiler integers.
    """

@overload
def sum(
    group: _BlockGroup,
    value: _ScalarValueT,
    /,
    *,
    broadcast: Literal[False],
    valid_items: None = None,
    algorithm: _ReduceAlgorithm,
) -> _ScalarValueT:
    """Sum a scalar with an explicit direct CUB BlockReduce algorithm."""

@overload
def sum(
    group: _WarpGroup,
    value: _ScalarValueT,
    /,
    *,
    broadcast: Literal[False],
    valid_items: _ValidItems,
    algorithm: None = None,
) -> _ScalarValueT:
    """Sum a valid scalar prefix through direct CUB WarpReduce.

    ``valid_items`` accepts Python, NumPy, and structural compiler integers.
    """

@overload
def scan(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: _SumScanOperator | None = None,
    initial_value: _PortableNumericScalar | None = None,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ScalarT:
    """Block-exclusive sum a scalar, optionally from an initial value.

    External compiler scalar values typed as ``Any`` necessarily return
    ``Any`` in the backend-neutral static contract.
    """

@overload
def scan(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: _NonSumScanOperator,
    initial_value: _PortableNumericScalar,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ScalarT:
    """Block-exclusive scan a scalar with a required non-sum initial value."""

@overload
def scan(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["inclusive"],
    scan_op: _ScanOperator | None = None,
    initial_value: None = None,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ScalarT:
    """Block-inclusive scan a portable or structural compiler scalar."""

@overload
def scan(
    group: _BlockGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: _SumScanOperator | None = None,
    initial_value: _PortableNumericScalar | None = None,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Return block-exclusive sums without mutating the input payload."""

@overload
def scan(
    group: _BlockGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: _NonSumScanOperator,
    initial_value: _PortableNumericScalar,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Return non-sum block prefixes from a required initial value."""

@overload
def scan(
    group: _BlockGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    mode: Literal["inclusive"],
    scan_op: _ScanOperator | None = None,
    initial_value: None = None,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Return block-inclusive prefixes without mutating the input payload."""

@overload
def scan(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: _SumScanOperator | None = None,
    initial_value: _PortableNumericScalar | None = None,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ScalarT:
    """Physical- or logical-warp-exclusive sum a scalar from an optional initial value.

    Physical- and logical-warp scans have no algorithm selector or caller-owned scratch.
    """

@overload
def scan(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["exclusive"] = "exclusive",
    scan_op: _NonSumScanOperator,
    initial_value: _PortableNumericScalar,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ScalarT:
    """Physical- or logical-warp-exclusive scan with a non-sum initial value."""

@overload
def scan(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["inclusive"],
    scan_op: _ScanOperator | None = None,
    initial_value: None = None,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ScalarT:
    """Physical- or logical-warp-inclusive scan without an initial value."""

@overload
def exclusive_sum(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ScalarT:
    """Preserve a scalar type through block-exclusive sum."""

@overload
def exclusive_sum(
    group: _BlockGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Return block-exclusive sums with the input payload shape."""

@overload
def exclusive_sum(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ScalarT:
    """Preserve a scalar type through physical- or logical-warp exclusive sum."""

@overload
def inclusive_sum(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ScalarT:
    """Preserve a scalar type through block-inclusive sum."""

@overload
def inclusive_sum(
    group: _BlockGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Return block-inclusive sums with the input payload shape."""

@overload
def inclusive_sum(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ScalarT:
    """Preserve a scalar type through physical- or logical-warp inclusive sum."""

@overload
def exclusive_scan(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: _SumScanOperator | None = None,
    initial_value: _PortableNumericScalar | None = None,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ScalarT:
    """Block-exclusive sum a scalar, optionally from an initial value."""

@overload
def exclusive_scan(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: _NonSumScanOperator,
    initial_value: _PortableNumericScalar,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ScalarT:
    """Block-exclusive scan a scalar with a required non-sum initial value."""

@overload
def exclusive_scan(
    group: _BlockGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    scan_op: _SumScanOperator | None = None,
    initial_value: _PortableNumericScalar | None = None,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Return block-exclusive sums with the input payload shape."""

@overload
def exclusive_scan(
    group: _BlockGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    scan_op: _NonSumScanOperator,
    initial_value: _PortableNumericScalar,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Return non-sum block prefixes from a required initial value."""

@overload
def exclusive_scan(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: _SumScanOperator | None = None,
    initial_value: _PortableNumericScalar | None = None,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ScalarT:
    """Physical- or logical-warp-exclusive sum from an optional initial value."""

@overload
def exclusive_scan(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: _NonSumScanOperator,
    initial_value: _PortableNumericScalar,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ScalarT:
    """Physical- or logical-warp-exclusive scan with a non-sum initial value."""

@overload
def inclusive_scan(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: _ScanOperator | None = None,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ScalarT:
    """Preserve a scalar type through block-inclusive Scan."""

@overload
def inclusive_scan(
    group: _BlockGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    scan_op: _ScanOperator | None = None,
    algorithm: _ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Return block-inclusive prefixes with the input payload shape."""

@overload
def inclusive_scan(
    group: _WarpGroup,
    value: _ScalarT,
    /,
    *,
    scan_op: _ScanOperator | None = None,
    algorithm: None = None,
    temp_storage: None = None,
) -> _ScalarT:
    """Preserve a scalar type through physical- or logical-warp inclusive Scan."""

def exchange(
    group: _MemoryGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    mode: _ExchangeMode = "striped_to_blocked",
) -> ThreadDataLike[_PortableNumericT]:
    """Return a layout-rearranged ``ThreadData`` payload without mutation.

    ``group`` may be a complete block, physical warp, or logical warp. The
    portable modes are ``"striped_to_blocked"`` and ``"blocked_to_striped"``.
    ``value`` must own one through five items per participant; scalar inputs are
    not supported. The result preserves the input payload's shape and item type.
    """

@overload
def adjacent_difference(
    group: _BlockGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    direction: Literal["left"] = "left",
    valid_items: _ValidItems | None = None,
    tile_predecessor_item: _PortableNumericT | None = None,
    tile_successor_item: None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Return left differences for ``value`` across ``group``.

    ``direction`` is left, ``valid_items`` may limit the tile,
    ``tile_predecessor_item`` supplies its boundary,
    ``tile_successor_item`` stays ``None``, and ``temp_storage`` supplies
    optional scratch.
    """

@overload
def adjacent_difference(
    group: _BlockGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    direction: Literal["right"],
    valid_items: _ValidItems | None = None,
    tile_predecessor_item: None = None,
    tile_successor_item: None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Return right differences for ``value`` across ``group``.

    ``direction`` is right, ``valid_items`` may limit the tile, both
    ``tile_predecessor_item`` and ``tile_successor_item`` stay ``None``, and
    ``temp_storage`` supplies optional scratch.
    """

@overload
def adjacent_difference(
    group: _BlockGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    direction: Literal["right"],
    valid_items: None = None,
    tile_predecessor_item: None = None,
    tile_successor_item: _PortableNumericT,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Return full-tile right differences for ``value`` across ``group``.

    ``direction`` is right, ``valid_items`` and ``tile_predecessor_item`` stay
    ``None``, ``tile_successor_item`` supplies the boundary, and
    ``temp_storage`` supplies optional scratch.
    """

@overload
def discontinuity(
    group: _BlockGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    mode: Literal["heads"] = "heads",
    tile_predecessor_item: _PortableNumericT | None = None,
    tile_successor_item: None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[int]:
    """Return block-wide signed 32-bit head flags.

    ``group`` must be a complete physical block and ``value`` must be a
    fixed-size per-thread payload. ``mode`` is ``"heads"`` and compares each
    item with its predecessor. The returned flag payload has the same shape as
    ``value``, and ``value`` is not mutated.

    ``tile_predecessor_item`` supplies the head boundary and must match the
    payload item type; without it, the first head is set.
    ``tile_successor_item`` stays ``None``. ``temp_storage`` supplies optional
    caller-owned scratch. The portable root uses built-in inequality.
    """

@overload
def discontinuity(
    group: _BlockGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    mode: Literal["tails"],
    tile_predecessor_item: None = None,
    tile_successor_item: _PortableNumericT | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[int]:
    """Return block-wide signed 32-bit tail flags.

    ``group`` must be a complete physical block and ``value`` must be a
    fixed-size per-thread payload. ``mode`` is ``"tails"`` and compares each
    item with its successor. The returned flag payload has the same shape as
    ``value``, and ``value`` is not mutated.

    ``tile_predecessor_item`` stays ``None``. ``tile_successor_item`` supplies
    the tail boundary and must match the payload item type; without it, the
    final tail is set. ``temp_storage`` supplies optional caller-owned scratch.
    The portable root uses built-in inequality and intentionally excludes the
    qualified ``"heads_and_tails"`` pair-returning mode.
    """

def shuffle(
    group: _BlockGroup,
    value: ThreadDataLike[_PortableNumericT],
    /,
    *,
    mode: _PortableShuffleMode = "down",
    distance: Literal[1] = 1,
) -> ThreadDataLike[_PortableNumericT]:
    """Return a unit-shifted block payload without mutating ``value``.

    ``group`` must be a complete physical block and ``value`` must be a
    fixed-size per-thread payload. ``mode="up"`` shifts the flattened blocked
    tile toward higher item ranks and leaves its first item undefined;
    ``mode="down"`` shifts toward lower ranks and leaves its last item
    undefined. ``distance`` is fixed at the portable value ``1``. Use a
    backend-qualified import for scalar Offset/Rotate or boundary outputs.
    """

@overload
def merge_sort_keys(
    group: _MemoryGroup,
    keys: ThreadDataLike[_IntegerKeyT],
    /,
    *,
    descending: bool = False,
    valid_items: None = None,
    oob_default: None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_IntegerKeyT]:
    """Return a fully merge-sorted integral payload without mutating ``keys``.

    ``group`` must be a complete physical block, physical warp, or logical warp;
    a block must contain a power-of-two number of threads. ``keys`` must be a fixed-size
    ``ThreadDataLike`` payload of Python, NumPy, or compiler integer values. The
    returned payload preserves the input item type and item count.
    """

@overload
def merge_sort_keys(
    group: _MemoryGroup,
    keys: ThreadDataLike[_IntegerKeyT],
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems,
    oob_default: _IntegerKeyT,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_IntegerKeyT]:
    """Return a partial-tile merge-sorted integral payload.

    ``valid_items`` and the key-typed ``oob_default`` must be supplied
    together. A block must contain a power-of-two number of threads. The
    sentinel must sort after every valid key: greater for ascending order and
    less for descending order. Only the valid sorted prefix is defined; the
    returned payload still preserves the input item type and item count. Use a
    qualified import for custom comparators or backend-specific group and
    payload forms.
    """

@overload
def merge_sort_pairs(
    group: _MemoryGroup,
    keys: ThreadDataLike[_IntegerKeyT],
    values: ThreadDataLike[_PortableNumericT],
    /,
    *,
    descending: bool = False,
    valid_items: None = None,
    oob_default: None = None,
    temp_storage: TempStorageLike | None = None,
) -> tuple[
    ThreadDataLike[_IntegerKeyT],
    ThreadDataLike[_PortableNumericT],
]:
    """Return fully merge-sorted key/value payloads without mutation.

    Keys determine ordering and each numeric value remains attached to its key.
    Both result payloads preserve their independent item types and common item
    count. Equal-key ordering is unspecified.
    """

@overload
def merge_sort_pairs(
    group: _MemoryGroup,
    keys: ThreadDataLike[_IntegerKeyT],
    values: ThreadDataLike[_PortableNumericT],
    /,
    *,
    descending: bool = False,
    valid_items: _ValidItems,
    oob_default: _IntegerKeyT,
    temp_storage: TempStorageLike | None = None,
) -> tuple[
    ThreadDataLike[_IntegerKeyT],
    ThreadDataLike[_PortableNumericT],
]:
    """Return a partial-tile merge-sorted key/value prefix.

    ``valid_items`` and the key-typed ``oob_default`` are supplied together.
    Only the valid sorted prefix is defined; key/value association is preserved.
    """

def radix_sort_keys(
    group: _BlockGroup,
    keys: ThreadDataLike[_IntegerKeyT],
    /,
    *,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    descending: bool = False,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_IntegerKeyT]:
    """Return a full-tile radix-sorted block payload without mutating ``keys``.

    ``group`` must be a complete physical block. ``keys`` must be a fixed-size
    ``ThreadDataLike`` payload of signed or unsigned 32- or 64-bit Python,
    NumPy, or compiler integer values. ``begin_bit`` and ``end_bit`` select a
    half-open interval in CUB's bit-ordered key representation. ``end_bit``
    defaults to the key width, including when only ``begin_bit`` is supplied.
    ``descending`` selects descending order. ``temp_storage`` supplies optional
    caller-owned scratch. The returned payload preserves the input item type
    and item count.
    """

def radix_sort_pairs(
    group: _BlockGroup,
    keys: ThreadDataLike[_IntegerKeyT],
    values: ThreadDataLike[_PortableNumericT],
    /,
    *,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    descending: bool = False,
    temp_storage: TempStorageLike | None = None,
) -> tuple[
    ThreadDataLike[_IntegerKeyT],
    ThreadDataLike[_PortableNumericT],
]:
    """Return radix-sorted block key/value payloads without mutation.

    Keys and values have matching item counts and retain their independent
    types. Values move with keys across the selected bit-ordered range.
    """

def radix_rank(
    group: _BlockGroup,
    keys: ThreadDataLike[_IntegerKeyT],
    /,
    *,
    begin_bit: _TraceInteger = 0,
    end_bit: _TraceInteger | None = None,
    radix_bits: _TraceInteger | None = None,
    descending: bool = False,
) -> ThreadDataLike[int]:
    """Return shape-preserving signed 32-bit ranks for one radix digit.

    ``group`` must be a complete physical block. ``keys`` must be a fixed-size
    payload of signed or unsigned 32- or 64-bit Python, NumPy, or compiler
    integer values. ``begin_bit``, ``end_bit``, and ``radix_bits`` are
    trace-time Python or NumPy integers. The selected half-open CUB bit-ordered
    interval defaults to four bits and may contain at most eight bits. Equal
    digits retain flattened blocked input order. The operation leaves ``keys``
    unchanged. Use a qualified import for scalar/register inputs or an
    exclusive digit-prefix side output.
    """

@overload
def histogram(
    group: _BlockGroup,
    samples: ThreadDataLike[_HistogramSampleT],
    /,
    *,
    bins: int,
    bins_per_thread: int = 1,
    counter_dtype: type[_CounterT],
    algorithm: _HistogramAlgorithm = "atomic",
) -> ThreadDataLike[_CounterT]:
    """Return striped counters typed by a portable dtype class.

    The complete block leaves compiler-produced ``samples`` unchanged.
    Positive static dimensions must cover every bin; excess slots are zero.
    Every sample must satisfy ``0 <= sample < bins``; violating this CUB
    precondition is undefined behavior.
    """

@overload
def histogram(
    group: _BlockGroup,
    samples: ThreadDataLike[_HistogramSampleT],
    /,
    *,
    bins: int,
    bins_per_thread: int = 1,
    counter_dtype: None = None,
    algorithm: _HistogramAlgorithm = "atomic",
) -> ThreadDataLike[int]:
    """Return default signed-integer striped counters.

    The complete block leaves compiler-produced ``samples`` unchanged.
    Positive static dimensions must cover every bin; excess slots are zero.
    Every sample must satisfy ``0 <= sample < bins``; violating this CUB
    precondition is undefined behavior.
    """

def run_length_decode(
    group: _BlockGroup,
    run_values: ThreadDataLike[_RunValueT],
    run_lengths: ThreadDataLike[_RunLengthT],
    /,
    *,
    decoded_items_per_thread: _TraceInteger,
    decoded_window_offset: _IntegerValue = 0,
) -> ThreadDataLike[_RunValueT]:
    """Return one fixed-size window from a blockwise run-length stream.

    ``group`` is the complete physical block. ``run_values`` and
    ``run_lengths`` are matching positive-size payloads whose flattened runs
    use blocked ownership. ``decoded_items_per_thread`` is a positive
    trace-static count. ``decoded_window_offset`` is a uniform nonnegative
    integer representable in the run-length dtype; callers providing a dynamic
    offset guarantee that range. Out-of-range decoded positions are zero. The
    new result preserves the run-value dtype and leaves both inputs unchanged.
    Actual runs have positive lengths; zero lengths are allowed only as one
    trailing padding suffix, and the block-wide sum is positive and
    representable in the run-length dtype. Relative offsets and total decoded
    size are available only from a qualified backend import.
    """

def topk_max_keys(
    group: _BlockGroup,
    keys: ThreadDataLike[_IntegerKeyT],
    k: _IntegerValue,
    /,
    *,
    valid_items: _ValidItems | None = None,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_IntegerKeyT]:
    """Return the largest keys in an unordered, shape-preserving prefix.

    ``group`` must be a complete one-dimensional physical block. ``keys`` is
    a fixed-size payload of Python integers or signed or unsigned 32- or
    64-bit NumPy or compiler integers. ``k`` and ``valid_items`` are uniform
    integer values satisfying ``1 <= k <= valid_items``; omitting
    ``valid_items`` selects the full block tile. ``begin_bit`` and ``end_bit``
    select a nonempty half-open interval in CUB's bit-ordered key
    representation, with ``end_bit=None`` selecting the key width. Only the
    first ``k`` flattened blocked positions are defined. That prefix is not
    sorted, ties do not expand it, and the remaining positions are undefined.
    The result preserves the input item type and count without mutating
    ``keys``. ``temp_storage`` supplies optional reusable scratch.
    """

def topk_min_keys(
    group: _BlockGroup,
    keys: ThreadDataLike[_IntegerKeyT],
    k: _IntegerValue,
    /,
    *,
    valid_items: _ValidItems | None = None,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_IntegerKeyT]:
    """Return the smallest keys in an unordered, shape-preserving prefix.

    ``group`` must be a complete one-dimensional physical block. ``keys`` is
    a fixed-size payload of Python integers or signed or unsigned 32- or
    64-bit NumPy or compiler integers. ``k`` and ``valid_items`` are uniform
    integer values satisfying ``1 <= k <= valid_items``; omitting
    ``valid_items`` selects the full block tile. ``begin_bit`` and ``end_bit``
    select a nonempty half-open interval in CUB's bit-ordered key
    representation, with ``end_bit=None`` selecting the key width. Only the
    first ``k`` flattened blocked positions are defined. That prefix is not
    sorted, ties do not expand it, and the remaining positions are undefined.
    The result preserves the input item type and count without mutating
    ``keys``. ``temp_storage`` supplies optional reusable scratch.
    """

def topk_max_pairs(
    group: _BlockGroup,
    keys: ThreadDataLike[_IntegerKeyT],
    values: ThreadDataLike[_PortableNumericT],
    k: _IntegerValue,
    /,
    *,
    valid_items: _ValidItems | None = None,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    temp_storage: TempStorageLike | None = None,
) -> tuple[
    ThreadDataLike[_IntegerKeyT],
    ThreadDataLike[_PortableNumericT],
]:
    """Return the largest-key pairs in an unordered prefix.

    Exactly the first ``k`` flattened blocked pairs are defined. Values remain
    attached to their keys and the remaining positions are undefined.
    ``temp_storage`` supplies optional reusable scratch.
    """

def topk_min_pairs(
    group: _BlockGroup,
    keys: ThreadDataLike[_IntegerKeyT],
    values: ThreadDataLike[_PortableNumericT],
    k: _IntegerValue,
    /,
    *,
    valid_items: _ValidItems | None = None,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    temp_storage: TempStorageLike | None = None,
) -> tuple[
    ThreadDataLike[_IntegerKeyT],
    ThreadDataLike[_PortableNumericT],
]:
    """Return the smallest-key pairs in an unordered prefix.

    Exactly the first ``k`` flattened blocked pairs are defined. Values remain
    attached to their keys and the remaining positions are undefined.
    ``temp_storage`` supplies optional reusable scratch.
    """

__all__ = [
    "Hierarchy",
    "TempStorage",
    "TempStorageLike",
    "ThreadData",
    "ThreadDataLike",
    "ThreadGroup",
    "ThreadHierarchy",
    "__version__",
    "adjacent_difference",
    "discontinuity",
    "exchange",
    "exclusive_scan",
    "exclusive_sum",
    "histogram",
    "inclusive_scan",
    "inclusive_sum",
    "load",
    "merge_sort_keys",
    "merge_sort_pairs",
    "radix_rank",
    "radix_sort_keys",
    "radix_sort_pairs",
    "reduce",
    "run_length_decode",
    "scan",
    "shuffle",
    "store",
    "sum",
    "this_block",
    "this_cluster",
    "this_grid",
    "this_thread",
    "this_warp",
    "topk_max_keys",
    "topk_max_pairs",
    "topk_min_keys",
    "topk_min_pairs",
]
