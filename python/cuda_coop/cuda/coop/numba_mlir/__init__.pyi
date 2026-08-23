# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Numba-CUDA-MLIR-qualified cooperative group building blocks."""

from typing import Any, Generic, Literal, Protocol, TypeAlias, overload

from numpy import int32 as _NumpyInt32
from numpy import uint8 as _NumpyUint8
from typing_extensions import TypeVar

from .. import ThreadGroup as _CommonThreadGroup
from .. import ThreadHierarchy as ThreadHierarchy
from .._typing import TempStorageSharing as _TempStorageSharing
from .._typing import ThreadDataLike as _ThreadDataLike
from .._typing import ThreadGroupKind as _ThreadGroupKind
from .._typing import ThreadLevel as _ThreadLevel
from .._typing import _SynchronizableGroupKind as _SynchronizableGroupKind

_ItemT = TypeVar("_ItemT")
_OpT = TypeVar("_OpT")
_DataclassT = TypeVar("_DataclassT")
_GroupKindT_co = TypeVar(
    "_GroupKindT_co",
    bound=_ThreadGroupKind,
    covariant=True,
    default=_ThreadGroupKind,
)
_ArrayShape: TypeAlias = int | tuple[int, ...]

Hierarchy = ThreadHierarchy

class StatefulFunction(Generic[_OpT]):
    """Device callable paired with explicit state for generated wrappers."""

    op: _OpT
    dtype: object
    name: str | None

    def __init__(
        self,
        op: _OpT,
        dtype: object,
        name: str | None = None,
    ) -> None:
        """Pair ``op`` with its state ``dtype`` and optional generated name."""

class ThreadGroup(
    _CommonThreadGroup[_GroupKindT_co],
    Generic[_GroupKindT_co],
):
    """Compile-time CUDA group descriptor for Numba-CUDA-MLIR."""

    def rank(self, level: _ThreadLevel = "thread") -> _NumpyInt32:
        """Return this group's rank as a NumPy-compatible ``int32`` scalar."""

    def count(self, level: _ThreadLevel = "thread") -> _NumpyInt32:
        """Return this group's count as a NumPy-compatible ``int32`` scalar."""

    @overload
    def rank_as(self, dtype: type[_ItemT], level: _ThreadLevel = "thread") -> _ItemT:
        """Return the group rank converted to an ordinary scalar dtype."""

    @overload
    def rank_as(self, dtype: object = None, level: _ThreadLevel = "thread") -> Any:
        """Return the group rank converted to a compiler dtype token."""

    @overload
    def count_as(
        self,
        dtype: type[_ItemT],
        level: _ThreadLevel = "thread",
    ) -> _ItemT:
        """Return the group count converted to an ordinary scalar dtype."""

    @overload
    def count_as(self, dtype: object = None, level: _ThreadLevel = "thread") -> Any:
        """Return the group count converted to a compiler dtype token."""

    def sync(self: ThreadGroup[_SynchronizableGroupKind]) -> None:
        """Synchronize participating members; grid groups are unsupported."""

    def sync_aligned(self: ThreadGroup[_SynchronizableGroupKind]) -> None:
        """Synchronize a converged non-grid group."""

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

    def is_member(self) -> _NumpyUint8:
        """Return a NumPy-compatible ``uint8`` membership flag."""

class TempStorage:
    """Explicit opaque byte scratch for planned shared-memory operations."""

    size_in_bytes: int | None
    alignment: int | None
    auto_sync: bool
    sharing: _TempStorageSharing

    def __init__(
        self,
        size_in_bytes: int | None = None,
        alignment: int | None = None,
        auto_sync: bool | None = None,
        sharing: _TempStorageSharing = "shared",
    ) -> None:
        """Configure scratch size, alignment, synchronization, and sharing."""

class _LocalMemory(Protocol):
    """Numba-CUDA-MLIR thread-local memory namespace."""

    def array(
        self,
        shape: _ArrayShape,
        dtype: object,
        alignas: int | None = 8,
        *,
        alignment: int | None = None,
    ) -> Any:
        """Allocate thread-local compiler storage."""

class _SharedMemory(Protocol):
    """Numba-CUDA-MLIR shared-memory namespace."""

    def array(
        self,
        shape: _ArrayShape,
        dtype: object,
        alignas: int | None = 8,
        *,
        alignment: int | None = None,
    ) -> Any:
        """Allocate shared compiler storage."""

local: _LocalMemory
shared: _SharedMemory

class _GpuDataclassArgumentHandler(Protocol):
    """Launch-time marshalling extension for registered GPU dataclasses."""

    def prepare_args(
        self,
        ty: Any,
        val: Any,
        stream: Any = None,
        retr: list[Any] | None = None,
    ) -> tuple[Any, Any]:
        """Flatten a registered dataclass and preserve its compiler type."""

gpu_dataclass_argument_handler: _GpuDataclassArgumentHandler

@overload
def ThreadData(
    items_per_thread: int,
    dtype: type[_ItemT],
    *,
    alignas: int = 8,
) -> _ThreadDataLike[_ItemT]:
    """Construct typed thread-local storage."""

@overload
def ThreadData(
    items_per_thread: int,
    dtype: object = None,
    *,
    alignas: int = 8,
) -> _ThreadDataLike[Any]:
    """Construct storage using a compiler dtype token or inferred dtype."""

def gpu_dataclass(
    dc: _DataclassT,
    *,
    compute_temp_storage: bool = True,
) -> _DataclassT:
    """Register a dataclass instance for Numba-CUDA-MLIR device use."""

def this_thread() -> ThreadGroup[Literal["thread"]]:
    """Describe the current thread."""

def this_warp() -> ThreadGroup[Literal["warp"]]:
    """Describe the current complete physical warp."""

def this_block() -> ThreadGroup[Literal["block"]]:
    """Describe the current CUDA thread block."""

def this_cluster() -> ThreadGroup[Literal["cluster"]]:
    """Describe the current cluster where the launch can represent it."""

def this_grid() -> ThreadGroup[Literal["grid"]]:
    """Describe the current grid."""

__all__ = [
    "Hierarchy",
    "StatefulFunction",
    "TempStorage",
    "ThreadData",
    "ThreadGroup",
    "ThreadHierarchy",
    "gpu_dataclass",
    "gpu_dataclass_argument_handler",
    "local",
    "shared",
    "this_block",
    "this_cluster",
    "this_grid",
    "this_thread",
    "this_warp",
]
