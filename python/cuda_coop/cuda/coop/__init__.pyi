# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from typing import Any, Literal, Protocol, TypeVar

_ItemT = TypeVar("_ItemT")

class ThreadGroup:
    """Descriptor for the current CUDA thread block.

    Creating this descriptor does not require an active compiler backend. When
    a cooperative primitive is compiled, the backend resolves the descriptor
    against the kernel's exact launch dimensions.

    Raises:
        TypeError: If user code attempts to construct the opaque descriptor
            directly instead of calling ``this_block``.

    Example:
        This tested CUTLASS kernel uses the current CUDA thread block:

        .. literalinclude:: ../../python/cuda_coop/examples/cutlass/block_load_store.py
           :language: python
           :start-after: example-begin block-load-store
           :end-before: example-end block-load-store
           :dedent: 4
    """

    @property
    def kind(self) -> Literal["block"]: ...

class _ThreadDataLike(Protocol[_ItemT]):
    dtype: Any | None

    @property
    def items_per_thread(self) -> int: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> _ItemT: ...
    def __setitem__(self, index: int, value: _ItemT) -> None: ...

def ThreadData(
    items_per_thread: int,
    dtype: Any = None,
) -> _ThreadDataLike[Any]:
    """Create an uninitialized per-thread register payload.

    The active backend owns the concrete payload type. When ``dtype`` is
    omitted, a primitive may infer it from its inputs for use by later
    primitives.

    Args:
        items_per_thread: Number of consecutive values owned by each thread.
        dtype: Optional portable numeric dtype. A primitive may infer it from
            its inputs.

    Returns:
        The active compiler backend's fixed-size payload object.

    Raises:
        ValueError: If ``items_per_thread`` is not positive.
        CoopCompilerContextRequiredError: If no compatible backend is active.

    Example:
        This tested CUTLASS kernel creates and uses a per-thread payload:

        .. literalinclude:: ../../python/cuda_coop/examples/cutlass/block_load_store.py
           :language: python
           :start-after: example-begin block-load-store
           :end-before: example-end block-load-store
           :dedent: 4
    """

def this_block() -> ThreadGroup:
    """Return a descriptor for the current CUDA thread block.

    The returned group has no user-supplied dimensions. The active backend
    supplies exact launch dimensions when it lowers a cooperative primitive.

    Returns:
        An opaque block descriptor accepted by cooperative primitives.

    Raises:
        RuntimeError: If a compiler backend later cannot resolve exact block
            dimensions for an operation using this descriptor.

    Example:
        This tested CUTLASS kernel uses the current CUDA thread block:

        .. literalinclude:: ../../python/cuda_coop/examples/cutlass/block_load_store.py
           :language: python
           :start-after: example-begin block-load-store
           :end-before: example-end block-load-store
           :dedent: 4
    """

def load(
    group: ThreadGroup,
    source: object,
    items: _ThreadDataLike[_ItemT],
    /,
    *,
    valid_items: Any = None,
    oob_default: Any = None,
    offset: Any = None,
) -> _ThreadDataLike[_ItemT]:
    """Collectively load one block tile into a per-thread payload.

    Every thread in ``group`` must participate in converged control flow. The
    payload size determines the number of consecutive values loaded per thread.
    Contiguous operands are traversed in linear storage order; multidimensional
    logical indexing is not applied.

    Args:
        group: The current CUDA thread block.
        source: Contiguous pointer-backed input memory.
        items: Payload whose size determines the values owned by each thread.
        valid_items: Optional valid element count for a partial block tile.
        oob_default: Optional value assigned to invalid Load positions.
        offset: Optional element offset from the input pointer.

    Returns:
        ``items`` after the active compiler backend populates it.

    Raises:
        TypeError: If ``group`` is invalid or ``oob_default`` is supplied
            without ``valid_items``.
        CoopCompilerContextRequiredError: If no compatible backend is active.

    Example:
        This tested CUTLASS kernel loads a partial block tile:

        .. literalinclude:: ../../python/cuda_coop/examples/cutlass/block_load_store.py
           :language: python
           :start-after: example-begin block-load-store
           :end-before: example-end block-load-store
           :dedent: 4
    """

def store(
    group: ThreadGroup,
    destination: object,
    items: _ThreadDataLike[Any],
    /,
    *,
    valid_items: Any = None,
    offset: Any = None,
) -> None:
    """Collectively store one per-thread payload as one block tile.

    Every thread in ``group`` must participate in converged control flow. The
    payload size determines the number of consecutive values stored per thread.
    Contiguous operands are traversed in linear storage order; multidimensional
    logical indexing is not applied.

    Args:
        group: The current CUDA thread block.
        destination: Contiguous pointer-backed output memory.
        items: Fixed-size payload stored by each thread.
        valid_items: Optional valid element count for a partial block tile.
        offset: Optional element offset from the output pointer.

    Returns:
        ``None``.

    Raises:
        TypeError: If ``group`` is not a ``ThreadGroup``.
        CoopCompilerContextRequiredError: If no compatible backend is active.

    Example:
        This tested CUTLASS kernel stores a partial block tile:

        .. literalinclude:: ../../python/cuda_coop/examples/cutlass/block_load_store.py
           :language: python
           :start-after: example-begin block-load-store
           :end-before: example-end block-load-store
           :dedent: 4
    """

__version__: str
