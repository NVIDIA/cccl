# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typed block-scoped Numba-CUDA-MLIR cooperative primitives."""

from collections.abc import Callable, Mapping
from typing import Any, Protocol, TypeVar, overload

from ..._typing import (
    LoadStoreAlgorithm,
    ReduceAlgorithm,
    ThreadDataLike,
)
from ..._typing import _ValidItems as _ValidItems
from .. import (
    BlockLoadAlgorithm,
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

__all__ = [
    "BlockLoadAlgorithm",
    "BlockStoreAlgorithm",
    "load",
    "make_load",
    "make_reduce",
    "make_store",
    "make_sum",
    "reduce",
    "store",
    "sum",
]
