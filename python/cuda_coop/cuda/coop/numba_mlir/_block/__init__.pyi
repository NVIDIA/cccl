# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typed block-scoped Numba-CUDA-MLIR cooperative primitives."""

from collections.abc import Callable, Mapping
from enum import IntEnum
from typing import Any, Protocol, TypeVar, overload

from ..._typing import (
    LoadStoreAlgorithm,
    ReduceAlgorithm,
    ScanAlgorithm,
    ThreadDataLike,
)
from ..._typing import _ValidItems as _ValidItems
from .. import (
    BlockHistogramAlgorithm,
    BlockLoadAlgorithm,
    BlockScanAlgorithm,
    BlockStoreAlgorithm,
    TempStorage,
)

_T = TypeVar("_T")
_K = TypeVar("_K")
_V = TypeVar("_V")
_Dim = int | tuple[int, ...] | list[int]
_Methods = Mapping[str, object]

class _CompilerDTypeToken(Protocol):
    """Structural Numba-CUDA-MLIR dtype token used by factory overloads."""

    name: str

_DTypeToken = type[object] | _CompilerDTypeToken

class _Invocable(Protocol):
    """Metadata shared by generated Numba-CUDA-MLIR device callables."""

    temp_storage_bytes: int
    temp_storage_alignment: int
    files: list[str]

class _LoadInvocable(_Invocable, Protocol[_T]):
    """Generated block-load callable."""

    @overload
    def __call__(
        self,
        source: object,
        output: ThreadDataLike[_T],
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Load one block tile into ``output``."""
    @overload
    def __call__(
        self,
        source: object,
        output: ThreadDataLike[_T],
        offset_or_valid_items: object,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Load an offset or partial block tile into ``output``."""
    @overload
    def __call__(
        self,
        source: object,
        output: ThreadDataLike[_T],
        valid_items: object,
        oob_default: _T,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Load a partial block tile and fill invalid items."""

class _StoreInvocable(_Invocable, Protocol[_T]):
    """Generated block-store callable."""

    @overload
    def __call__(
        self,
        destination: object,
        value: ThreadDataLike[_T],
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Store one block tile from ``value``."""
    @overload
    def __call__(
        self,
        destination: object,
        value: ThreadDataLike[_T],
        offset_or_valid_items: object,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Store an offset or partial block tile from ``value``."""

class _ReduceInvocable(_Invocable, Protocol[_T]):
    """Generated block-reduction callable."""

    @overload
    def __call__(
        self,
        value: _T | ThreadDataLike[_T],
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> _T:
        """Reduce one scalar or per-thread payload."""
    @overload
    def __call__(
        self,
        value: _T,
        valid_items: object,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> _T:
        """Reduce the valid prefix of scalar block inputs."""

class _TransformInvocable(_Invocable, Protocol[_T]):
    """Generated one-payload block transform."""

    @overload
    def __call__(
        self,
        value: ThreadDataLike[_T],
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Transform ``value`` in place."""
    @overload
    def __call__(
        self,
        value: ThreadDataLike[_T],
        first_runtime_operand: object,
        second_runtime_operand: object,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Transform ``value`` with a configured runtime range."""

class _OutTransformInvocable(_Invocable, Protocol[_T]):
    """Generated out-of-place block transform."""

    def __call__(
        self,
        value: ThreadDataLike[_T],
        output: ThreadDataLike[_T],
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Transform ``value`` into ``output``."""

class _ScanInvocable(_Invocable, Protocol[_T]):
    """Generated block scan callable."""

    @overload
    def __call__(
        self,
        value: ThreadDataLike[_T],
        output: ThreadDataLike[_T],
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Scan ``value`` into ``output``."""
    @overload
    def __call__(
        self,
        value: ThreadDataLike[_T],
        output: ThreadDataLike[_T],
        aggregate_or_prefix: object,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Scan and expose one configured runtime result."""

class _AdjacentDifferenceInvocable(_Invocable, Protocol[_T]):
    """Generated block adjacent-difference callable."""

    @overload
    def __call__(
        self,
        value: ThreadDataLike[_T],
        output: ThreadDataLike[_T],
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Write adjacent differences to ``output``."""
    @overload
    def __call__(
        self,
        value: ThreadDataLike[_T],
        output: ThreadDataLike[_T],
        first_runtime_operand: object,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Write adjacent differences with one runtime operand."""
    @overload
    def __call__(
        self,
        value: ThreadDataLike[_T],
        output: ThreadDataLike[_T],
        first_runtime_operand: object,
        second_runtime_operand: object,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Write adjacent differences with two runtime operands."""

class _DiscontinuityInvocable(_Invocable, Protocol[_T]):
    """Generated block discontinuity callable."""

    @overload
    def __call__(
        self,
        value: ThreadDataLike[_T],
        flags: object,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Write one discontinuity flag payload."""
    @overload
    def __call__(
        self,
        value: ThreadDataLike[_T],
        flags: object,
        third: object,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Write discontinuity flags with three runtime operands."""
    @overload
    def __call__(
        self,
        value: ThreadDataLike[_T],
        flags: object,
        third: object,
        fourth: object,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Write discontinuity flags with four runtime operands."""
    @overload
    def __call__(
        self,
        value: ThreadDataLike[_T],
        head_flags: object,
        tail_flags: object,
        tile_predecessor_item: object,
        tile_successor_item: object,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Write both flag payloads with tile boundary items."""

class _ShuffleInvocable(_Invocable, Protocol[_T]):
    """Generated scalar or array block shuffle callable."""

    @overload
    def __call__(
        self,
        value: _T | ThreadDataLike[_T],
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> _T | None:
        """Shuffle a scalar or array payload."""
    @overload
    def __call__(
        self,
        value: ThreadDataLike[_T],
        output: ThreadDataLike[_T],
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Shuffle an array payload into ``output``."""
    @overload
    def __call__(
        self,
        value: ThreadDataLike[_T],
        output: ThreadDataLike[_T],
        boundary: object,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Shuffle an array payload and expose its boundary item."""

class _RadixRankInvocable(_Invocable, Protocol[_T]):
    """Generated block radix-rank callable."""

    @overload
    def __call__(
        self,
        keys: ThreadDataLike[_T],
        ranks: object,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Write stable ranks for ``keys``."""
    @overload
    def __call__(
        self,
        keys: ThreadDataLike[_T],
        ranks: object,
        exclusive_digit_prefix: object,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Write ranks and the exclusive digit prefix."""

class _ExchangeInvocable(_Invocable, Protocol[_T]):
    """Generated in-place or out-of-place block exchange callable."""

    @overload
    def __call__(
        self,
        value: ThreadDataLike[_T],
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Exchange ``value`` in place."""
    @overload
    def __call__(
        self,
        value: ThreadDataLike[_T],
        operand: object,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Exchange into an output payload or use one scatter operand."""
    @overload
    def __call__(
        self,
        value: ThreadDataLike[_T],
        output: ThreadDataLike[_T],
        ranks: object,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Scatter ``value`` into ``output`` according to ``ranks``."""
    @overload
    def __call__(
        self,
        value: ThreadDataLike[_T],
        output: ThreadDataLike[_T],
        ranks: object,
        valid_flags: object,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Flagged-scatter ``value`` into ``output``."""

class _PairTransformInvocable(_Invocable, Protocol[_K, _V]):
    """Generated key-value block transform."""

    @overload
    def __call__(
        self,
        keys: ThreadDataLike[_K],
        values: ThreadDataLike[_V],
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Transform key-value payloads in place."""
    @overload
    def __call__(
        self,
        keys: ThreadDataLike[_K],
        values: ThreadDataLike[_V],
        first_runtime_operand: object,
        second_runtime_operand: object,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Transform key-value payloads with a runtime range."""

class _TopKKeysInvocable(_Invocable, Protocol[_K]):
    """Generated keys-only block TopK callable."""

    @overload
    def __call__(
        self,
        keys: ThreadDataLike[_K],
        k: object,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Select ``k`` keys in place."""
    @overload
    def __call__(
        self,
        keys: ThreadDataLike[_K],
        k: object,
        num_valid: object,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Select ``k`` keys from a valid prefix."""
    @overload
    def __call__(
        self,
        keys: ThreadDataLike[_K],
        k: object,
        begin_bit: object,
        end_bit: object,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Select ``k`` keys using a runtime bit range."""
    @overload
    def __call__(
        self,
        keys: ThreadDataLike[_K],
        k: object,
        num_valid: object,
        begin_bit: object,
        end_bit: object,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Select ``k`` valid keys using a runtime bit range."""

class _TopKPairsInvocable(_Invocable, Protocol[_K, _V]):
    """Generated key-value block TopK callable."""

    @overload
    def __call__(
        self,
        keys: ThreadDataLike[_K],
        values: ThreadDataLike[_V],
        k: object,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Select ``k`` key-value pairs in place."""
    @overload
    def __call__(
        self,
        keys: ThreadDataLike[_K],
        values: ThreadDataLike[_V],
        k: object,
        num_valid: object,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Select ``k`` pairs from a valid prefix."""
    @overload
    def __call__(
        self,
        keys: ThreadDataLike[_K],
        values: ThreadDataLike[_V],
        k: object,
        begin_bit: object,
        end_bit: object,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Select ``k`` pairs using a runtime bit range."""
    @overload
    def __call__(
        self,
        keys: ThreadDataLike[_K],
        values: ThreadDataLike[_V],
        k: object,
        num_valid: object,
        begin_bit: object,
        end_bit: object,
        /,
        *,
        temp_storage: TempStorage | None = None,
    ) -> None:
        """Select ``k`` valid pairs using a runtime bit range."""

class _Histogram(Protocol):
    """Parent object returned by the qualified histogram intrinsic."""

    def init(self, histogram: object = None, /) -> None:
        """Initialize the shared counters."""

    def composite(
        self,
        samples: object = None,
        histogram: object = None,
        /,
    ) -> None:
        """Accumulate one per-thread sample payload."""

class _RunLengthDecoder(Protocol):
    """Parent object returned by the qualified run-length intrinsic."""

    def decode(
        self,
        decoded_items: object,
        decoded_window_offset: object = None,
        relative_offsets: object = None,
        /,
    ) -> None:
        """Decode one output window and optional relative offsets."""

class _RunLengthInvocable(Protocol):
    """Generated run-length parent factory."""

    temp_storage_bytes: int
    temp_storage_alignment: int

    def __call__(
        self,
        run_values: object,
        run_lengths: object,
        runs_per_thread: int | None = None,
        decoded_items_per_thread: int | None = None,
        total_decoded_size: object = None,
        decoded_offset_dtype: object = None,
        temp_storage: TempStorage | None = None,
    ) -> _RunLengthDecoder:
        """Bind encoded runs and return a decoder."""

    def decode(
        self,
        run_length_dtype: object = None,
        total_decoded_size_dtype: object = None,
        *,
        with_relative_offsets: bool = False,
        with_decoded_window_offset: bool = False,
        relative_offset_dtype: object = None,
    ) -> _Invocable:
        """Build a generated fused decode callable."""

class BlockAdjacentDifferenceType(IntEnum):
    """Direction used by block adjacent difference."""

    SubtractLeft = 1
    SubtractRight = 2

class BlockDiscontinuityType(IntEnum):
    """Output layout used by block discontinuity."""

    HEADS = 1
    TAILS = 2
    HEADS_AND_TAILS = 3

class BlockExchangeType(IntEnum):
    """CUB block-exchange data movement pattern."""

    StripedToBlocked = 1
    BlockedToStriped = 2
    WarpStripedToBlocked = 3
    BlockedToWarpStriped = 4
    ScatterToBlocked = 5
    ScatterToStriped = 6
    ScatterToStripedGuarded = 7
    ScatterToStripedFlagged = 8

class BlockShuffleType(IntEnum):
    """CUB block-shuffle movement pattern."""

    Offset = 1
    Rotate = 2
    Up = 3
    Down = 4

@overload
def load(
    dtype: type[_T],
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    algorithm: LoadStoreAlgorithm | BlockLoadAlgorithm = "direct",
    num_valid_items: object = None,
    oob_default: _T | None = None,
    dim: _Dim | None = None,
) -> _LoadInvocable[_T]:
    """Build a dtype-preserving block-load callable outside compilation."""

@overload
def load(
    source: object,
    output: ThreadDataLike[_T],
    /,
    *,
    algorithm: LoadStoreAlgorithm | BlockLoadAlgorithm = "direct",
    num_valid_items: object = None,
    oob_default: object = None,
    offset: object = None,
    items_per_thread: int | None = None,
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    dim: _Dim | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Load a block tile into an existing per-thread payload."""

@overload
def load(
    source: object,
    output: ThreadDataLike[_T],
    offset_or_valid_items: object,
    /,
    *,
    algorithm: LoadStoreAlgorithm | BlockLoadAlgorithm = "direct",
    items_per_thread: int | None = None,
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    dim: _Dim | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Load an offset or partial block tile."""

@overload
def load(
    source: object,
    output: ThreadDataLike[_T],
    valid_items: object,
    oob_default: object,
    /,
    *,
    algorithm: LoadStoreAlgorithm | BlockLoadAlgorithm = "direct",
    items_per_thread: int | None = None,
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    dim: _Dim | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Load a partial block tile and fill invalid items."""

@overload
def load(
    dtype: object,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    algorithm: LoadStoreAlgorithm | BlockLoadAlgorithm = "direct",
    num_valid_items: object = None,
    oob_default: object = None,
    dim: _Dim | None = None,
) -> _LoadInvocable[Any]:
    """Build a block-load callable from an external compiler dtype token."""

@overload
def store(
    dtype: type[_T],
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    algorithm: LoadStoreAlgorithm | BlockStoreAlgorithm = "direct",
    num_valid_items: object = None,
    oob_default: object = None,
    dim: _Dim | None = None,
) -> _StoreInvocable[_T]:
    """Build a dtype-preserving block-store callable outside compilation."""

@overload
def store(
    destination: object,
    value: ThreadDataLike[_T],
    /,
    *,
    algorithm: LoadStoreAlgorithm | BlockStoreAlgorithm = "direct",
    num_valid_items: object = None,
    offset: object = None,
    items_per_thread: int | None = None,
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    dim: _Dim | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Store an existing per-thread payload to a block tile."""

@overload
def store(
    destination: object,
    value: ThreadDataLike[_T],
    offset_or_valid_items: object,
    /,
    *,
    algorithm: LoadStoreAlgorithm | BlockStoreAlgorithm = "direct",
    items_per_thread: int | None = None,
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    dim: _Dim | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Store an offset or partial block tile."""

@overload
def store(
    dtype: object,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    algorithm: LoadStoreAlgorithm | BlockStoreAlgorithm = "direct",
    num_valid_items: object = None,
    oob_default: object = None,
    dim: _Dim | None = None,
) -> _StoreInvocable[Any]:
    """Build a block-store callable from an external compiler dtype token."""

@overload
def make_load(
    dtype: type[_T],
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    algorithm: LoadStoreAlgorithm | BlockLoadAlgorithm = "direct",
    num_valid_items: object = None,
    oob_default: _T | None = None,
    dim: _Dim | None = None,
) -> _LoadInvocable[_T]:
    """Build a dtype-preserving block-load callable."""

@overload
def make_load(
    dtype: object,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    algorithm: LoadStoreAlgorithm | BlockLoadAlgorithm = "direct",
    num_valid_items: object = None,
    oob_default: object = None,
    dim: _Dim | None = None,
) -> _LoadInvocable[Any]:
    """Build a block-load callable from a compiler dtype token."""

@overload
def make_store(
    dtype: type[_T],
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    algorithm: LoadStoreAlgorithm | BlockStoreAlgorithm = "direct",
    num_valid_items: object = None,
    oob_default: object = None,
    dim: _Dim | None = None,
) -> _StoreInvocable[_T]:
    """Build a dtype-preserving block-store callable."""

@overload
def make_store(
    dtype: object,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    algorithm: LoadStoreAlgorithm | BlockStoreAlgorithm = "direct",
    num_valid_items: object = None,
    oob_default: object = None,
    dim: _Dim | None = None,
) -> _StoreInvocable[Any]:
    """Build a block-store callable from a compiler dtype token."""

@overload
def reduce(
    dtype: type[_T],
    threads_per_block: _Dim | None = None,
    binary_op: Callable[[_T, _T], _T] | None = None,
    items_per_thread: int = 1,
    algorithm: ReduceAlgorithm = "warp_reductions",
    num_valid: object = None,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _ReduceInvocable[_T]:
    """Build a dtype-preserving block-reduction callable outside compilation."""

@overload
def reduce(
    value: _T | ThreadDataLike[_T],
    /,
    *,
    binary_op: Callable[[_T, _T], _T] | None = None,
    items_per_thread: int = 1,
    algorithm: ReduceAlgorithm = "warp_reductions",
    num_valid: object = None,
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
    temp_storage: TempStorage | None = None,
) -> _T:
    """Reduce block values with a qualified device callback."""

@overload
def reduce(
    value: _T,
    valid_items: object,
    /,
    *,
    binary_op: Callable[[_T, _T], _T] | None = None,
    algorithm: ReduceAlgorithm = "warp_reductions",
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
    temp_storage: TempStorage | None = None,
) -> _T:
    """Reduce the valid prefix of scalar block inputs."""

@overload
def reduce(
    dtype: object,
    threads_per_block: _Dim | None = None,
    binary_op: object = None,
    items_per_thread: int = 1,
    algorithm: ReduceAlgorithm = "warp_reductions",
    num_valid: object = None,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _ReduceInvocable[Any]:
    """Build a block-reduction callable from an external compiler dtype token."""

@overload
def sum(
    dtype: type[_T],
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    algorithm: ReduceAlgorithm = "warp_reductions",
    num_valid: object = None,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _ReduceInvocable[_T]:
    """Build a dtype-preserving block-sum callable outside compilation."""

@overload
def sum(
    value: _T | ThreadDataLike[_T],
    /,
    *,
    items_per_thread: int = 1,
    algorithm: ReduceAlgorithm = "warp_reductions",
    num_valid: object = None,
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
    temp_storage: TempStorage | None = None,
) -> _T:
    """Sum block values and preserve their item type."""

@overload
def sum(
    value: _T,
    valid_items: object,
    /,
    *,
    algorithm: ReduceAlgorithm = "warp_reductions",
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
    temp_storage: TempStorage | None = None,
) -> _T:
    """Sum the valid prefix of scalar block inputs."""

@overload
def sum(
    dtype: object,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    algorithm: ReduceAlgorithm = "warp_reductions",
    num_valid: object = None,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _ReduceInvocable[Any]:
    """Build a block-sum callable from an external compiler dtype token."""

@overload
def make_reduce(
    dtype: type[_T],
    threads_per_block: _Dim | None = None,
    binary_op: Callable[[_T, _T], _T] | None = None,
    items_per_thread: int = 1,
    algorithm: ReduceAlgorithm = "warp_reductions",
    num_valid: object = None,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _ReduceInvocable[_T]:
    """Build a dtype-preserving block-reduction callable."""

@overload
def make_reduce(
    dtype: object,
    threads_per_block: _Dim | None = None,
    binary_op: object = None,
    items_per_thread: int = 1,
    algorithm: ReduceAlgorithm = "warp_reductions",
    num_valid: object = None,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _ReduceInvocable[Any]:
    """Build a block-reduction callable from a compiler dtype token."""

@overload
def make_sum(
    dtype: type[_T],
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    algorithm: ReduceAlgorithm = "warp_reductions",
    num_valid: object = None,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _ReduceInvocable[_T]:
    """Build a dtype-preserving block-sum callable."""

@overload
def make_sum(
    dtype: object,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    algorithm: ReduceAlgorithm = "warp_reductions",
    num_valid: object = None,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _ReduceInvocable[Any]:
    """Build a block-sum callable from a compiler dtype token."""

@overload
def scan(
    dtype: type[_T],
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    mode: str = "exclusive",
    scan_op: Callable[[_T, _T], _T] | str = "+",
    initial_value: object = None,
    block_prefix_callback_op: object = None,
    prefix_op: object = None,
    block_aggregate: object = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm = "raking",
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _ScanInvocable[_T]:
    """Build a dtype-preserving block-scan callable outside compilation."""

@overload
def scan(
    value: ThreadDataLike[_T],
    output: ThreadDataLike[_T],
    /,
    *,
    mode: str = "exclusive",
    scan_op: Callable[[_T, _T], _T] | str = "+",
    initial_value: object = None,
    block_prefix_callback_op: object = None,
    prefix_op: object = None,
    block_aggregate: object = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm = "raking",
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Scan a block payload into an explicit output payload."""

@overload
def scan(
    value: ThreadDataLike[_T],
    output: ThreadDataLike[_T],
    aggregate_or_prefix: object,
    /,
    *,
    mode: str = "exclusive",
    scan_op: Callable[[_T, _T], _T] | str = "+",
    initial_value: object = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm = "raking",
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Scan a block payload and expose one configured runtime result."""

@overload
def scan(
    dtype: object,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    mode: str = "exclusive",
    scan_op: object = "+",
    initial_value: object = None,
    block_prefix_callback_op: object = None,
    prefix_op: object = None,
    block_aggregate: object = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm = "raking",
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _ScanInvocable[Any]:
    """Build a block-scan callable from an external compiler dtype token."""

@overload
def exclusive_sum(
    dtype: type[_T],
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    block_prefix_callback_op: object = None,
    prefix_op: object = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm = "raking",
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _ScanInvocable[_T]:
    """Build a dtype-preserving exclusive block-sum callable."""

@overload
def exclusive_sum(
    value: ThreadDataLike[_T],
    output: ThreadDataLike[_T],
    /,
    *,
    block_prefix_callback_op: object = None,
    prefix_op: object = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm = "raking",
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Write exclusive block-sum prefixes to ``output``."""

@overload
def exclusive_sum(
    value: ThreadDataLike[_T],
    output: ThreadDataLike[_T],
    aggregate_or_prefix: object,
    /,
    *,
    algorithm: ScanAlgorithm | BlockScanAlgorithm = "raking",
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Write exclusive prefixes and one configured runtime result."""

@overload
def exclusive_sum(
    dtype: object,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    block_prefix_callback_op: object = None,
    prefix_op: object = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm = "raking",
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _ScanInvocable[Any]:
    """Build an exclusive block-sum callable from a compiler dtype token."""

@overload
def inclusive_sum(
    dtype: type[_T],
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    block_prefix_callback_op: object = None,
    prefix_op: object = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm = "raking",
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _ScanInvocable[_T]:
    """Build a dtype-preserving inclusive block-sum callable."""

@overload
def inclusive_sum(
    value: ThreadDataLike[_T],
    output: ThreadDataLike[_T],
    /,
    *,
    block_prefix_callback_op: object = None,
    prefix_op: object = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm = "raking",
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Write inclusive block-sum prefixes to ``output``."""

@overload
def inclusive_sum(
    value: ThreadDataLike[_T],
    output: ThreadDataLike[_T],
    aggregate_or_prefix: object,
    /,
    *,
    algorithm: ScanAlgorithm | BlockScanAlgorithm = "raking",
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Write inclusive prefixes and one configured runtime result."""

@overload
def inclusive_sum(
    dtype: object,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    block_prefix_callback_op: object = None,
    prefix_op: object = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm = "raking",
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _ScanInvocable[Any]:
    """Build an inclusive block-sum callable from a compiler dtype token."""

@overload
def exclusive_scan(
    dtype: type[_T],
    threads_per_block: _Dim | None = None,
    scan_op: Callable[[_T, _T], _T] | str = "+",
    items_per_thread: int = 1,
    initial_value: object = None,
    block_prefix_callback_op: object = None,
    prefix_op: object = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm = "raking",
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _ScanInvocable[_T]:
    """Build a dtype-preserving exclusive block-scan callable."""

@overload
def exclusive_scan(
    value: ThreadDataLike[_T],
    output: ThreadDataLike[_T],
    /,
    *,
    scan_op: Callable[[_T, _T], _T] | str = "+",
    initial_value: object = None,
    block_prefix_callback_op: object = None,
    prefix_op: object = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm = "raking",
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Write exclusive callback-based prefixes to ``output``."""

@overload
def exclusive_scan(
    value: ThreadDataLike[_T],
    output: ThreadDataLike[_T],
    aggregate_or_prefix: object,
    /,
    *,
    scan_op: Callable[[_T, _T], _T] | str = "+",
    initial_value: object = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm = "raking",
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Write exclusive prefixes and one configured runtime result."""

@overload
def exclusive_scan(
    dtype: object,
    threads_per_block: _Dim | None = None,
    scan_op: object = "+",
    items_per_thread: int = 1,
    initial_value: object = None,
    block_prefix_callback_op: object = None,
    prefix_op: object = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm = "raking",
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _ScanInvocable[Any]:
    """Build an exclusive block-scan callable from a compiler dtype token."""

@overload
def inclusive_scan(
    dtype: type[_T],
    threads_per_block: _Dim | None = None,
    scan_op: Callable[[_T, _T], _T] | str = "+",
    items_per_thread: int = 1,
    initial_value: object = None,
    block_prefix_callback_op: object = None,
    prefix_op: object = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm = "raking",
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _ScanInvocable[_T]:
    """Build a dtype-preserving inclusive block-scan callable."""

@overload
def inclusive_scan(
    value: ThreadDataLike[_T],
    output: ThreadDataLike[_T],
    /,
    *,
    scan_op: Callable[[_T, _T], _T] | str = "+",
    block_prefix_callback_op: object = None,
    prefix_op: object = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm = "raking",
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Write inclusive callback-based prefixes to ``output``."""

@overload
def inclusive_scan(
    value: ThreadDataLike[_T],
    output: ThreadDataLike[_T],
    aggregate_or_prefix: object,
    /,
    *,
    scan_op: Callable[[_T, _T], _T] | str = "+",
    initial_value: object = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm = "raking",
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Write inclusive prefixes and one configured runtime result."""

@overload
def inclusive_scan(
    dtype: object,
    threads_per_block: _Dim | None = None,
    scan_op: object = "+",
    items_per_thread: int = 1,
    initial_value: object = None,
    block_prefix_callback_op: object = None,
    prefix_op: object = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm = "raking",
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _ScanInvocable[Any]:
    """Build an inclusive block-scan callable from a compiler dtype token."""

@overload
def make_scan(
    dtype: type[_T],
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    mode: str = "exclusive",
    scan_op: Callable[[_T, _T], _T] | str = "+",
    initial_value: object = None,
    block_prefix_callback_op: object = None,
    prefix_op: object = None,
    block_aggregate: object = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm = "raking",
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _ScanInvocable[_T]:
    """Build a dtype-preserving generated block-scan callable."""

@overload
def make_scan(
    dtype: object,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    mode: str = "exclusive",
    scan_op: object = "+",
    initial_value: object = None,
    block_prefix_callback_op: object = None,
    prefix_op: object = None,
    block_aggregate: object = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm = "raking",
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _ScanInvocable[Any]:
    """Build a generated block-scan callable."""

@overload
def make_exclusive_sum(
    dtype: type[_T],
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    block_prefix_callback_op: object = None,
    prefix_op: object = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm = "raking",
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _ScanInvocable[_T]:
    """Build a dtype-preserving generated exclusive block-sum callable."""

@overload
def make_exclusive_sum(
    dtype: object,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    block_prefix_callback_op: object = None,
    prefix_op: object = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm = "raking",
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _ScanInvocable[Any]:
    """Build a generated exclusive block-sum callable."""

@overload
def make_inclusive_sum(
    dtype: type[_T],
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    block_prefix_callback_op: object = None,
    prefix_op: object = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm = "raking",
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _ScanInvocable[_T]:
    """Build a dtype-preserving generated inclusive block-sum callable."""

@overload
def make_inclusive_sum(
    dtype: object,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    block_prefix_callback_op: object = None,
    prefix_op: object = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm = "raking",
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _ScanInvocable[Any]:
    """Build a generated inclusive block-sum callable."""

@overload
def make_exclusive_scan(
    dtype: type[_T],
    threads_per_block: _Dim | None = None,
    scan_op: Callable[[_T, _T], _T] | str = "+",
    items_per_thread: int = 1,
    initial_value: object = None,
    block_prefix_callback_op: object = None,
    prefix_op: object = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm = "raking",
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _ScanInvocable[_T]:
    """Build a dtype-preserving generated exclusive block-scan callable."""

@overload
def make_exclusive_scan(
    dtype: object,
    threads_per_block: _Dim | None = None,
    scan_op: object = "+",
    items_per_thread: int = 1,
    initial_value: object = None,
    block_prefix_callback_op: object = None,
    prefix_op: object = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm = "raking",
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _ScanInvocable[Any]:
    """Build a generated exclusive block-scan callable."""

@overload
def make_inclusive_scan(
    dtype: type[_T],
    threads_per_block: _Dim | None = None,
    scan_op: Callable[[_T, _T], _T] | str = "+",
    items_per_thread: int = 1,
    initial_value: object = None,
    block_prefix_callback_op: object = None,
    prefix_op: object = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm = "raking",
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _ScanInvocable[_T]:
    """Build a dtype-preserving generated inclusive block-scan callable."""

@overload
def make_inclusive_scan(
    dtype: object,
    threads_per_block: _Dim | None = None,
    scan_op: object = "+",
    items_per_thread: int = 1,
    initial_value: object = None,
    block_prefix_callback_op: object = None,
    prefix_op: object = None,
    algorithm: ScanAlgorithm | BlockScanAlgorithm = "raking",
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _ScanInvocable[Any]:
    """Build a generated inclusive block-scan callable."""

@overload
def exchange(
    value: ThreadDataLike[_T],
    /,
    *,
    block_exchange_type: BlockExchangeType = BlockExchangeType.StripedToBlocked,
    items_per_thread: int = 1,
    warp_time_slicing: bool = False,
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Exchange a block payload in place."""

@overload
def exchange(
    value: ThreadDataLike[_T],
    output: ThreadDataLike[_T],
    /,
    *,
    block_exchange_type: BlockExchangeType = BlockExchangeType.StripedToBlocked,
    items_per_thread: int = 1,
    warp_time_slicing: bool = False,
    ranks: object = None,
    valid_flags: object = None,
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Exchange a block payload into an explicit output payload."""

@overload
def exchange(
    value: ThreadDataLike[_T],
    output_or_ranks: object,
    ranks_or_flags: object,
    /,
    *,
    block_exchange_type: BlockExchangeType = BlockExchangeType.ScatterToStriped,
    items_per_thread: int = 1,
    warp_time_slicing: bool = False,
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Scatter a block payload with two configured runtime operands."""

@overload
def exchange(
    value: ThreadDataLike[_T],
    output: ThreadDataLike[_T],
    ranks: object,
    valid_flags: object,
    /,
    *,
    block_exchange_type: BlockExchangeType = BlockExchangeType.ScatterToStripedFlagged,
    items_per_thread: int = 1,
    warp_time_slicing: bool = False,
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Flagged-scatter a block payload into ``output``."""

@overload
def exchange(
    block_exchange_type: BlockExchangeType
    | object = BlockExchangeType.StripedToBlocked,
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    warp_time_slicing: bool = False,
    use_output_items: bool | None = None,
    offset_dtype: object = None,
    valid_flag_dtype: object = None,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _ExchangeInvocable[Any]:
    """Build a block-exchange callable outside compilation."""

def make_exchange(
    block_exchange_type: BlockExchangeType
    | object = BlockExchangeType.StripedToBlocked,
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    warp_time_slicing: bool = False,
    use_output_items: bool | None = None,
    offset_dtype: object = None,
    valid_flag_dtype: object = None,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _ExchangeInvocable[Any]:
    """Build a generated block-exchange callable."""

@overload
def adjacent_difference(
    value: ThreadDataLike[_T],
    output: ThreadDataLike[_T],
    /,
    *,
    block_adjacent_difference_type: BlockAdjacentDifferenceType = BlockAdjacentDifferenceType.SubtractLeft,
    difference_op: Callable[[_T, _T], _T] | None = None,
    valid_items: object = None,
    tile_predecessor_item: _T | None = None,
    tile_successor_item: _T | None = None,
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Write qualified adjacent differences to ``output``."""

@overload
def adjacent_difference(
    value: ThreadDataLike[_T],
    output: ThreadDataLike[_T],
    first_runtime_operand: object,
    /,
    *,
    block_adjacent_difference_type: BlockAdjacentDifferenceType = BlockAdjacentDifferenceType.SubtractLeft,
    difference_op: Callable[[_T, _T], _T] | None = None,
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Write adjacent differences with one configured runtime operand."""

@overload
def adjacent_difference(
    value: ThreadDataLike[_T],
    output: ThreadDataLike[_T],
    first_runtime_operand: object,
    second_runtime_operand: object,
    /,
    *,
    block_adjacent_difference_type: BlockAdjacentDifferenceType = BlockAdjacentDifferenceType.SubtractLeft,
    difference_op: Callable[[_T, _T], _T] | None = None,
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Write adjacent differences with two configured runtime operands."""

@overload
def adjacent_difference(
    block_adjacent_difference_type: BlockAdjacentDifferenceType
    | object = BlockAdjacentDifferenceType.SubtractLeft,
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    difference_op: object = None,
    methods: _Methods | None = None,
    valid_items: object = None,
    tile_predecessor_item: object = None,
    tile_successor_item: object = None,
    dim: _Dim | None = None,
) -> _AdjacentDifferenceInvocable[Any]:
    """Build an adjacent-difference callable outside compilation."""

def make_adjacent_difference(
    block_adjacent_difference_type: BlockAdjacentDifferenceType
    | object = BlockAdjacentDifferenceType.SubtractLeft,
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    difference_op: object = None,
    methods: _Methods | None = None,
    valid_items: object = None,
    tile_predecessor_item: object = None,
    tile_successor_item: object = None,
    dim: _Dim | None = None,
) -> _AdjacentDifferenceInvocable[Any]:
    """Build a generated adjacent-difference callable."""

@overload
def discontinuity(
    value: ThreadDataLike[_T],
    flags: object,
    tail_flags: object = None,
    /,
    *,
    flag_op: Callable[[_T, _T], object] | None = None,
    block_discontinuity_type: BlockDiscontinuityType = BlockDiscontinuityType.HEADS,
    tile_predecessor_item: _T | None = None,
    tile_successor_item: _T | None = None,
    dtype: object = None,
    flag_dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Write qualified discontinuity flags to one or two outputs."""

@overload
def discontinuity(
    value: ThreadDataLike[_T],
    flags: object,
    third: object,
    fourth: object,
    /,
    *,
    flag_op: Callable[[_T, _T], object] | None = None,
    block_discontinuity_type: BlockDiscontinuityType = BlockDiscontinuityType.HEADS,
    dtype: object = None,
    flag_dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Write discontinuity flags with four runtime operands."""

@overload
def discontinuity(
    value: ThreadDataLike[_T],
    head_flags: object,
    tail_flags: object,
    tile_predecessor_item: object,
    tile_successor_item: object,
    /,
    *,
    flag_op: Callable[[_T, _T], object] | None = None,
    block_discontinuity_type: BlockDiscontinuityType = BlockDiscontinuityType.HEADS_AND_TAILS,
    dtype: object = None,
    flag_dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Write head and tail flags with both tile boundary items."""

@overload
def discontinuity(
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    flag_op: object = None,
    flag_dtype: object = None,
    block_discontinuity_type: BlockDiscontinuityType = BlockDiscontinuityType.HEADS,
    methods: _Methods | None = None,
    tile_predecessor_item: object = None,
    tile_successor_item: object = None,
    dim: _Dim | None = None,
) -> _DiscontinuityInvocable[Any]:
    """Build a discontinuity callable outside compilation."""

def make_discontinuity(
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    flag_op: object = None,
    flag_dtype: object = None,
    block_discontinuity_type: BlockDiscontinuityType = BlockDiscontinuityType.HEADS,
    methods: _Methods | None = None,
    tile_predecessor_item: object = None,
    tile_successor_item: object = None,
    dim: _Dim | None = None,
) -> _DiscontinuityInvocable[Any]:
    """Build a generated block-discontinuity callable."""

@overload
def shuffle(
    value: object,
    output: object = None,
    /,
    *,
    block_shuffle_type: BlockShuffleType = BlockShuffleType.Up,
    distance: object = None,
    block_prefix: object = None,
    block_suffix: object = None,
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int | None = None,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
    temp_storage: TempStorage | None = None,
) -> object:
    """Apply a qualified scalar or array block shuffle."""

@overload
def shuffle(
    value: object,
    output: object,
    boundary: object,
    /,
    *,
    block_shuffle_type: BlockShuffleType = BlockShuffleType.Up,
    distance: object = None,
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int | None = None,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
    temp_storage: TempStorage | None = None,
) -> object:
    """Shuffle an array payload and expose its boundary item."""

@overload
def shuffle(
    block_shuffle_type: BlockShuffleType | object = BlockShuffleType.Up,
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int | None = None,
    distance: object = None,
    block_prefix: object = None,
    block_suffix: object = None,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _ShuffleInvocable[Any]:
    """Build a block-shuffle callable outside compilation."""

def make_shuffle(
    block_shuffle_type: BlockShuffleType | object = BlockShuffleType.Up,
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int | None = None,
    distance: object = None,
    block_prefix: object = None,
    block_suffix: object = None,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _ShuffleInvocable[Any]:
    """Build a generated block-shuffle callable."""

@overload
def merge_sort_keys(
    keys: ThreadDataLike[_K],
    /,
    *,
    compare_op: Callable[[_K, _K], object] | None = None,
    descending: bool | None = None,
    valid_items: None = None,
    oob_default: None = None,
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Sort a block key payload in place."""

@overload
def merge_sort_keys(
    keys: ThreadDataLike[_K],
    valid_items: _ValidItems,
    oob_default: _K,
    /,
    *,
    compare_op: Callable[[_K, _K], object] | None = None,
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Sort a partial block key payload in place.

    ``oob_default`` must sort after every valid key under ``compare_op``.
    """

@overload
def merge_sort_keys(
    dtype: type[_K],
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    compare_op: Callable[[_K, _K], object] | None = None,
    value_dtype: object = None,
    valid_items: object = None,
    oob_default: object = None,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _TransformInvocable[_K]:
    """Build a dtype-preserving block key-sort callable."""

@overload
def merge_sort_keys(
    dtype: _DTypeToken,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    compare_op: object = None,
    value_dtype: object = None,
    valid_items: object = None,
    oob_default: object = None,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _TransformInvocable[Any]:
    """Build a block key-sort callable from a compiler dtype token."""

@overload
def merge_sort_pairs(
    keys: ThreadDataLike[_K],
    values: ThreadDataLike[_V],
    /,
    *,
    compare_op: Callable[[_K, _K], object] | None = None,
    descending: bool | None = None,
    valid_items: None = None,
    oob_default: None = None,
    key_dtype: object = None,
    value_dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Sort block key-value payloads in place."""

@overload
def merge_sort_pairs(
    keys: ThreadDataLike[_K],
    values: ThreadDataLike[_V],
    valid_items: _ValidItems,
    oob_default: _K,
    /,
    *,
    compare_op: Callable[[_K, _K], object] | None = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Sort partial block key-value payloads in place.

    ``oob_default`` must sort after every valid key under ``compare_op``.
    """

@overload
def merge_sort_pairs(
    keys: type[_K],
    values: type[_V],
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    compare_op: Callable[[_K, _K], object] | None = None,
    valid_items: object = None,
    oob_default: object = None,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _PairTransformInvocable[_K, _V]:
    """Build a dtype-preserving block key-value sort callable."""

@overload
def merge_sort_pairs(
    keys: _DTypeToken,
    values: _DTypeToken,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    compare_op: object = None,
    valid_items: object = None,
    oob_default: object = None,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _PairTransformInvocable[Any, Any]:
    """Build a block key-value sort callable from compiler dtype tokens."""

@overload
def make_merge_sort_keys(
    dtype: type[_K],
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    compare_op: Callable[[_K, _K], object] | None = None,
    value_dtype: object = None,
    valid_items: object = None,
    oob_default: object = None,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _TransformInvocable[_K]:
    """Build a dtype-preserving generated block key-sort callable."""

@overload
def make_merge_sort_keys(
    dtype: _DTypeToken,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    compare_op: object = None,
    value_dtype: object = None,
    valid_items: object = None,
    oob_default: object = None,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _TransformInvocable[Any]:
    """Build a generated block key-sort callable."""

@overload
def make_merge_sort_pairs(
    keys: type[_K],
    values: type[_V],
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    compare_op: Callable[[_K, _K], object] | None = None,
    valid_items: object = None,
    oob_default: object = None,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _PairTransformInvocable[_K, _V]:
    """Build a dtype-preserving generated block key-value sort callable."""

@overload
def make_merge_sort_pairs(
    keys: _DTypeToken,
    values: _DTypeToken,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    compare_op: object = None,
    valid_items: object = None,
    oob_default: object = None,
    methods: _Methods | None = None,
    dim: _Dim | None = None,
) -> _PairTransformInvocable[Any, Any]:
    """Build a generated block key-value sort callable."""

@overload
def radix_sort_keys(
    keys: ThreadDataLike[_K],
    /,
    *,
    begin_bit: int | None = None,
    end_bit: int | None = None,
    blocked_to_striped: bool = False,
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    temp_storage: TempStorage | None = None,
) -> None:
    """Radix-sort block keys in ascending order in place."""

@overload
def radix_sort_keys(
    keys: ThreadDataLike[_K],
    begin_bit: object,
    end_bit: object,
    /,
    *,
    blocked_to_striped: bool = False,
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    temp_storage: TempStorage | None = None,
) -> None:
    """Radix-sort block keys using a runtime bit range."""

@overload
def radix_sort_keys(
    dtype: object,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    value_dtype: object = None,
    begin_bit: int | None = None,
    end_bit: int | None = None,
    blocked_to_striped: bool = False,
    dim: _Dim | None = None,
) -> _TransformInvocable[Any]:
    """Build an ascending block radix-key-sort callable."""

@overload
def radix_sort_keys_descending(
    keys: ThreadDataLike[_K],
    /,
    *,
    begin_bit: int | None = None,
    end_bit: int | None = None,
    blocked_to_striped: bool = False,
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    temp_storage: TempStorage | None = None,
) -> None:
    """Radix-sort block keys in descending order in place."""

@overload
def radix_sort_keys_descending(
    keys: ThreadDataLike[_K],
    begin_bit: object,
    end_bit: object,
    /,
    *,
    blocked_to_striped: bool = False,
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    temp_storage: TempStorage | None = None,
) -> None:
    """Descending-radix-sort block keys using a runtime bit range."""

@overload
def radix_sort_keys_descending(
    dtype: object,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    value_dtype: object = None,
    begin_bit: int | None = None,
    end_bit: int | None = None,
    blocked_to_striped: bool = False,
    dim: _Dim | None = None,
) -> _TransformInvocable[Any]:
    """Build a descending block radix-key-sort callable."""

@overload
def radix_sort_pairs(
    keys: ThreadDataLike[_K],
    values: ThreadDataLike[_V],
    /,
    *,
    begin_bit: int | None = None,
    end_bit: int | None = None,
    blocked_to_striped: bool = False,
    key_dtype: object = None,
    value_dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    temp_storage: TempStorage | None = None,
) -> None:
    """Radix-sort block key-value payloads in ascending order."""

@overload
def radix_sort_pairs(
    keys: ThreadDataLike[_K],
    values: ThreadDataLike[_V],
    begin_bit: object,
    end_bit: object,
    /,
    *,
    blocked_to_striped: bool = False,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    temp_storage: TempStorage | None = None,
) -> None:
    """Radix-sort pairs using a runtime bit range."""

@overload
def radix_sort_pairs(
    keys: object = None,
    values: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    begin_bit: int | None = None,
    end_bit: int | None = None,
    blocked_to_striped: bool = False,
    key_dtype: object = None,
    value_dtype: object = None,
    dim: _Dim | None = None,
) -> _PairTransformInvocable[Any, Any]:
    """Build an ascending block radix-pair-sort callable."""

@overload
def radix_sort_pairs_descending(
    keys: ThreadDataLike[_K],
    values: ThreadDataLike[_V],
    /,
    *,
    begin_bit: int | None = None,
    end_bit: int | None = None,
    blocked_to_striped: bool = False,
    key_dtype: object = None,
    value_dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    temp_storage: TempStorage | None = None,
) -> None:
    """Radix-sort block key-value payloads in descending order."""

@overload
def radix_sort_pairs_descending(
    keys: ThreadDataLike[_K],
    values: ThreadDataLike[_V],
    begin_bit: object,
    end_bit: object,
    /,
    *,
    blocked_to_striped: bool = False,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    temp_storage: TempStorage | None = None,
) -> None:
    """Descending-radix-sort pairs using a runtime bit range."""

@overload
def radix_sort_pairs_descending(
    keys: object = None,
    values: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    begin_bit: int | None = None,
    end_bit: int | None = None,
    blocked_to_striped: bool = False,
    key_dtype: object = None,
    value_dtype: object = None,
    dim: _Dim | None = None,
) -> _PairTransformInvocable[Any, Any]:
    """Build a descending block radix-pair-sort callable."""

def make_radix_sort_keys(
    dtype: object,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    value_dtype: object = None,
    begin_bit: int | None = None,
    end_bit: int | None = None,
    blocked_to_striped: bool = False,
    dim: _Dim | None = None,
) -> _TransformInvocable[Any]:
    """Build an ascending block radix-key-sort callable."""

def make_radix_sort_keys_descending(
    dtype: object,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    value_dtype: object = None,
    begin_bit: int | None = None,
    end_bit: int | None = None,
    blocked_to_striped: bool = False,
    dim: _Dim | None = None,
) -> _TransformInvocable[Any]:
    """Build a descending block radix-key-sort callable."""

def make_radix_sort_pairs(
    keys: object = None,
    values: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    begin_bit: int | None = None,
    end_bit: int | None = None,
    blocked_to_striped: bool = False,
    key_dtype: object = None,
    value_dtype: object = None,
    dim: _Dim | None = None,
) -> _PairTransformInvocable[Any, Any]:
    """Build an ascending block radix-pair-sort callable."""

def make_radix_sort_pairs_descending(
    keys: object = None,
    values: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    begin_bit: int | None = None,
    end_bit: int | None = None,
    blocked_to_striped: bool = False,
    key_dtype: object = None,
    value_dtype: object = None,
    dim: _Dim | None = None,
) -> _PairTransformInvocable[Any, Any]:
    """Build a descending block radix-pair-sort callable."""

@overload
def radix_rank(
    keys: ThreadDataLike[_K],
    ranks: object,
    /,
    *,
    begin_bit: int = 0,
    end_bit: int | None = None,
    radix_bits: int | None = None,
    descending: bool = False,
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    exclusive_digit_prefix: object = None,
    dim: _Dim | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Write stable radix ranks for a block key payload."""

@overload
def radix_rank(
    keys: ThreadDataLike[_K],
    ranks: object,
    exclusive_digit_prefix: object,
    /,
    *,
    begin_bit: int = 0,
    end_bit: int | None = None,
    radix_bits: int | None = None,
    descending: bool = False,
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    dim: _Dim | None = None,
    temp_storage: TempStorage | None = None,
) -> None:
    """Write radix ranks and the exclusive digit prefix."""

@overload
def radix_rank(
    dtype: object,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    begin_bit: int = 0,
    end_bit: int | None = None,
    descending: bool = False,
    exclusive_digit_prefix: object = None,
    dim: _Dim | None = None,
) -> _RadixRankInvocable[Any]:
    """Build a block radix-rank callable outside compilation."""

def make_radix_rank(
    dtype: object,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    begin_bit: int = 0,
    end_bit: int | None = None,
    descending: bool = False,
    exclusive_digit_prefix: object = None,
    dim: _Dim | None = None,
) -> _RadixRankInvocable[Any]:
    """Build a generated block radix-rank callable."""

@overload
def histogram(
    samples: object,
    histogram: object,
    /,
    *,
    algorithm: BlockHistogramAlgorithm = BlockHistogramAlgorithm.ATOMIC,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    bins: int = 256,
    dim: _Dim | None = None,
    temp_storage: TempStorage | None = None,
    item_dtype: object = None,
    counter_dtype: object = None,
) -> _Histogram:
    """Bind block samples and a shared histogram parent object."""

@overload
def histogram(
    items: object = None,
    histogram: object = None,
    algorithm: BlockHistogramAlgorithm = BlockHistogramAlgorithm.ATOMIC,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    bins: int = 256,
    dim: _Dim | None = None,
    temp_storage: TempStorage | None = None,
    item_dtype: object = None,
    counter_dtype: object = None,
) -> _Histogram:
    """Build or bind a block-histogram parent outside compilation."""

def make_histogram(
    items: object = None,
    histogram: object = None,
    algorithm: BlockHistogramAlgorithm = BlockHistogramAlgorithm.ATOMIC,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    bins: int = 256,
    dim: _Dim | None = None,
    temp_storage: TempStorage | None = None,
    item_dtype: object = None,
    counter_dtype: object = None,
) -> _Histogram:
    """Build a generated block-histogram parent constructor."""

@overload
def run_length(
    run_values: object,
    run_lengths: object,
    runs_per_thread: int = 1,
    decoded_items_per_thread: int = 1,
    total_decoded_size: object = None,
    /,
    *,
    decoded_offset_dtype: object = None,
    temp_storage: TempStorage | None = None,
    threads_per_block: _Dim | None = None,
    dim: _Dim | None = None,
    item_dtype: object = None,
) -> _RunLengthDecoder:
    """Bind run-length encoded payloads and return a decoder."""

@overload
def run_length(
    run_values: object = None,
    run_lengths: object = None,
    runs_per_thread: int = 1,
    decoded_items_per_thread: int = 1,
    total_decoded_size: object = None,
    decoded_offset_dtype: object = None,
    temp_storage: TempStorage | None = None,
    threads_per_block: _Dim | None = None,
    dim: _Dim | None = None,
    item_dtype: object = None,
) -> _RunLengthInvocable:
    """Build a run-length parent factory outside compilation."""

def make_run_length(
    run_values: object = None,
    run_lengths: object = None,
    runs_per_thread: int = 1,
    decoded_items_per_thread: int = 1,
    total_decoded_size: object = None,
    decoded_offset_dtype: object = None,
    temp_storage: TempStorage | None = None,
    threads_per_block: _Dim | None = None,
    dim: _Dim | None = None,
    item_dtype: object = None,
) -> _RunLengthInvocable:
    """Build a generated run-length decoder parent constructor."""

@overload
def topk_max_keys(
    keys: ThreadDataLike[_K],
    k: object,
    /,
    *,
    num_valid: object = None,
    begin_bit: int | None = None,
    end_bit: int | None = None,
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    temp_storage: TempStorage | None = None,
) -> None:
    """Select maximum block keys into the first ``k`` positions."""

@overload
def topk_max_keys(
    keys: ThreadDataLike[_K],
    k: object,
    num_valid: object,
    /,
    *,
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    temp_storage: TempStorage | None = None,
) -> None:
    """Select maximum keys from a valid prefix."""

@overload
def topk_max_keys(
    keys: ThreadDataLike[_K],
    k: object,
    begin_bit: object,
    end_bit: object,
    /,
    *,
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    temp_storage: TempStorage | None = None,
) -> None:
    """Select maximum keys using a runtime bit range."""

@overload
def topk_max_keys(
    keys: ThreadDataLike[_K],
    k: object,
    num_valid: object,
    begin_bit: object,
    end_bit: object,
    /,
    *,
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    temp_storage: TempStorage | None = None,
) -> None:
    """Select maximum keys from a valid prefix and runtime bit range."""

@overload
def topk_max_keys(
    dtype: object,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    num_valid: object = None,
    begin_bit: int | None = None,
    end_bit: int | None = None,
    dim: _Dim | None = None,
) -> _TopKKeysInvocable[Any]:
    """Build a maximum-key TopK callable outside compilation."""

@overload
def topk_min_keys(
    keys: ThreadDataLike[_K],
    k: object,
    /,
    *,
    num_valid: object = None,
    begin_bit: int | None = None,
    end_bit: int | None = None,
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    temp_storage: TempStorage | None = None,
) -> None:
    """Select minimum block keys into the first ``k`` positions."""

@overload
def topk_min_keys(
    keys: ThreadDataLike[_K],
    k: object,
    num_valid: object,
    /,
    *,
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    temp_storage: TempStorage | None = None,
) -> None:
    """Select minimum keys from a valid prefix."""

@overload
def topk_min_keys(
    keys: ThreadDataLike[_K],
    k: object,
    begin_bit: object,
    end_bit: object,
    /,
    *,
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    temp_storage: TempStorage | None = None,
) -> None:
    """Select minimum keys using a runtime bit range."""

@overload
def topk_min_keys(
    keys: ThreadDataLike[_K],
    k: object,
    num_valid: object,
    begin_bit: object,
    end_bit: object,
    /,
    *,
    dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    temp_storage: TempStorage | None = None,
) -> None:
    """Select minimum keys from a valid prefix and runtime bit range."""

@overload
def topk_min_keys(
    dtype: object,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    num_valid: object = None,
    begin_bit: int | None = None,
    end_bit: int | None = None,
    dim: _Dim | None = None,
) -> _TopKKeysInvocable[Any]:
    """Build a minimum-key TopK callable outside compilation."""

@overload
def topk_max_pairs(
    keys: ThreadDataLike[_K],
    values: ThreadDataLike[_V],
    k: object,
    /,
    *,
    num_valid: object = None,
    begin_bit: int | None = None,
    end_bit: int | None = None,
    key_dtype: object = None,
    value_dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    temp_storage: TempStorage | None = None,
) -> None:
    """Select maximum block key-value pairs in place."""

@overload
def topk_max_pairs(
    keys: ThreadDataLike[_K],
    values: ThreadDataLike[_V],
    k: object,
    num_valid: object,
    /,
    *,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    temp_storage: TempStorage | None = None,
) -> None:
    """Select maximum pairs from a valid prefix."""

@overload
def topk_max_pairs(
    keys: ThreadDataLike[_K],
    values: ThreadDataLike[_V],
    k: object,
    begin_bit: object,
    end_bit: object,
    /,
    *,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    temp_storage: TempStorage | None = None,
) -> None:
    """Select maximum pairs using a runtime bit range."""

@overload
def topk_max_pairs(
    keys: ThreadDataLike[_K],
    values: ThreadDataLike[_V],
    k: object,
    num_valid: object,
    begin_bit: object,
    end_bit: object,
    /,
    *,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    temp_storage: TempStorage | None = None,
) -> None:
    """Select maximum pairs from a valid prefix and runtime bit range."""

@overload
def topk_max_pairs(
    keys: object = None,
    values: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    num_valid: object = None,
    begin_bit: int | None = None,
    end_bit: int | None = None,
    key_dtype: object = None,
    value_dtype: object = None,
    dim: _Dim | None = None,
) -> _TopKPairsInvocable[Any, Any]:
    """Build a maximum-pair TopK callable outside compilation."""

@overload
def topk_min_pairs(
    keys: ThreadDataLike[_K],
    values: ThreadDataLike[_V],
    k: object,
    /,
    *,
    num_valid: object = None,
    begin_bit: int | None = None,
    end_bit: int | None = None,
    key_dtype: object = None,
    value_dtype: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    temp_storage: TempStorage | None = None,
) -> None:
    """Select minimum block key-value pairs in place."""

@overload
def topk_min_pairs(
    keys: ThreadDataLike[_K],
    values: ThreadDataLike[_V],
    k: object,
    num_valid: object,
    /,
    *,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    temp_storage: TempStorage | None = None,
) -> None:
    """Select minimum pairs from a valid prefix."""

@overload
def topk_min_pairs(
    keys: ThreadDataLike[_K],
    values: ThreadDataLike[_V],
    k: object,
    begin_bit: object,
    end_bit: object,
    /,
    *,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    temp_storage: TempStorage | None = None,
) -> None:
    """Select minimum pairs using a runtime bit range."""

@overload
def topk_min_pairs(
    keys: ThreadDataLike[_K],
    values: ThreadDataLike[_V],
    k: object,
    num_valid: object,
    begin_bit: object,
    end_bit: object,
    /,
    *,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    temp_storage: TempStorage | None = None,
) -> None:
    """Select minimum pairs from a valid prefix and runtime bit range."""

@overload
def topk_min_pairs(
    keys: object = None,
    values: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    num_valid: object = None,
    begin_bit: int | None = None,
    end_bit: int | None = None,
    key_dtype: object = None,
    value_dtype: object = None,
    dim: _Dim | None = None,
) -> _TopKPairsInvocable[Any, Any]:
    """Build a minimum-pair TopK callable outside compilation."""

def make_topk_max_keys(
    dtype: object,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    num_valid: object = None,
    begin_bit: int | None = None,
    end_bit: int | None = None,
    dim: _Dim | None = None,
) -> _TopKKeysInvocable[Any]:
    """Build a generated maximum-key TopK callable."""

def make_topk_min_keys(
    dtype: object,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    num_valid: object = None,
    begin_bit: int | None = None,
    end_bit: int | None = None,
    dim: _Dim | None = None,
) -> _TopKKeysInvocable[Any]:
    """Build a generated minimum-key TopK callable."""

def make_topk_max_pairs(
    keys: object = None,
    values: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    num_valid: object = None,
    begin_bit: int | None = None,
    end_bit: int | None = None,
    key_dtype: object = None,
    value_dtype: object = None,
    dim: _Dim | None = None,
) -> _TopKPairsInvocable[Any, Any]:
    """Build a generated maximum-pair TopK callable."""

def make_topk_min_pairs(
    keys: object = None,
    values: object = None,
    threads_per_block: _Dim | None = None,
    items_per_thread: int = 1,
    num_valid: object = None,
    begin_bit: int | None = None,
    end_bit: int | None = None,
    key_dtype: object = None,
    value_dtype: object = None,
    dim: _Dim | None = None,
) -> _TopKPairsInvocable[Any, Any]:
    """Build a generated minimum-pair TopK callable."""

__all__ = [
    "BlockAdjacentDifferenceType",
    "BlockDiscontinuityType",
    "BlockExchangeType",
    "BlockHistogramAlgorithm",
    "BlockLoadAlgorithm",
    "BlockScanAlgorithm",
    "BlockShuffleType",
    "BlockStoreAlgorithm",
    "adjacent_difference",
    "discontinuity",
    "exchange",
    "exclusive_scan",
    "exclusive_sum",
    "histogram",
    "inclusive_scan",
    "inclusive_sum",
    "load",
    "make_adjacent_difference",
    "make_discontinuity",
    "make_exchange",
    "make_exclusive_scan",
    "make_exclusive_sum",
    "make_histogram",
    "make_inclusive_scan",
    "make_inclusive_sum",
    "make_load",
    "make_merge_sort_keys",
    "make_merge_sort_pairs",
    "make_radix_rank",
    "make_radix_sort_keys",
    "make_radix_sort_keys_descending",
    "make_radix_sort_pairs",
    "make_radix_sort_pairs_descending",
    "make_reduce",
    "make_run_length",
    "make_scan",
    "make_shuffle",
    "make_store",
    "make_sum",
    "make_topk_max_keys",
    "make_topk_max_pairs",
    "make_topk_min_keys",
    "make_topk_min_pairs",
    "merge_sort_keys",
    "merge_sort_pairs",
    "radix_rank",
    "radix_sort_keys",
    "radix_sort_keys_descending",
    "radix_sort_pairs",
    "radix_sort_pairs_descending",
    "reduce",
    "run_length",
    "scan",
    "shuffle",
    "store",
    "sum",
    "topk_max_keys",
    "topk_max_pairs",
    "topk_min_keys",
    "topk_min_pairs",
]
