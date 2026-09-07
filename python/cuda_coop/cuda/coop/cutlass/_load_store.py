# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Qualified CUTLASS block-wide Load and Store entry points."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any

import numpy as np

from cuda.coop._core import (
    ArgumentBinding,
    GroupLoadStoreAlgorithm,
    GroupLoadStoreKind,
    GroupLoadStoreSemantics,
    ThreadGroup,
    make_group_primitive_call,
    plan_group_primitive,
)

from ._launch import current_launch_facts
from ._provider import (
    _canonical_type,
    _memory_type,
    materialize_load,
    materialize_store,
)
from ._thread_data import ThreadData


def _is_boolean(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    from cutlass.base_dsl.typing import Boolean

    return isinstance(value, Boolean)


def _validate_group(group: ThreadGroup, *, operation: str) -> None:
    if not isinstance(group, ThreadGroup):
        raise TypeError(f"cuda.coop.cutlass.{operation} group must be a ThreadGroup")
    if group.kind != "block":
        raise NotImplementedError(
            f"cuda.coop.cutlass.{operation} currently supports block groups only"
        )


def _integer_binding(value: Any, *, name: str) -> ArgumentBinding:
    if value is None:
        return ArgumentBinding.omitted()
    if _is_boolean(value):
        raise TypeError(f"cuda.coop.cutlass {name} must be an integer")
    if isinstance(value, Integral):
        return ArgumentBinding.static(int(value))
    from cutlass.base_dsl.typing import Integer

    if isinstance(value, Integer):
        return ArgumentBinding.runtime()
    raise TypeError(
        f"cuda.coop.cutlass {name} must be an integer, not {type(value).__name__}"
    )


def _oob_binding(value: Any) -> ArgumentBinding:
    if value is None:
        return ArgumentBinding.omitted()
    if _is_boolean(value):
        raise TypeError("cuda.coop.cutlass oob_default must be numeric, not boolean")
    if isinstance(value, Integral):
        return ArgumentBinding.static(value)
    if isinstance(value, Real):
        if not math.isfinite(float(value)):
            raise ValueError("cuda.coop.cutlass oob_default must be finite")
        return ArgumentBinding.static(value)
    from cutlass.base_dsl.typing import Numeric

    if isinstance(value, Numeric):
        return ArgumentBinding.runtime()
    raise TypeError(
        "cuda.coop.cutlass oob_default must be a numeric scalar, not "
        f"{type(value).__name__}"
    )


def _plan(
    *,
    group: ThreadGroup,
    kind: GroupLoadStoreKind,
    dtype: type,
    items_per_thread: int,
    valid_items: Any,
    oob_default: Any,
    offset: Any,
):
    operation = GroupLoadStoreSemantics(
        kind=kind,
        dtype=dtype,
        items_per_thread=items_per_thread,
        algorithm=GroupLoadStoreAlgorithm.DIRECT,
        valid_items=_integer_binding(valid_items, name="valid_items"),
        oob_default=_oob_binding(oob_default),
        offset=_integer_binding(offset, name="offset"),
    )
    call = make_group_primitive_call(
        group,
        operation,
        source="cuda.coop.cutlass",
    )
    return plan_group_primitive(
        call,
        current_launch_facts(feature=kind.value),
    ).require_supported()


def load(
    group: ThreadGroup,
    source: Any,
    items: ThreadData,
    /,
    *,
    valid_items: Any = None,
    oob_default: Any = None,
    offset: Any = None,
) -> ThreadData:
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

    _validate_group(group, operation="load")
    if not isinstance(items, ThreadData):
        raise TypeError("cuda.coop.cutlass.load items must be ThreadData")
    if oob_default is not None and valid_items is None:
        raise TypeError("cuda.coop.cutlass.load oob_default requires valid_items")
    value_type = _memory_type(source, feature="load")
    plan = _plan(
        group=group,
        kind=GroupLoadStoreKind.LOAD,
        dtype=value_type,
        items_per_thread=items.items_per_thread,
        valid_items=valid_items,
        oob_default=oob_default,
        offset=offset,
    )
    return materialize_load(
        plan=plan,
        source=source,
        output=items,
        valid_items=valid_items,
        oob_default=oob_default,
        offset=offset,
    )


def store(
    group: ThreadGroup,
    destination: Any,
    items: ThreadData,
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

    _validate_group(group, operation="store")
    if not isinstance(items, ThreadData):
        raise TypeError("cuda.coop.cutlass.store items must be ThreadData")
    if items.dtype is None:
        raise TypeError("cuda.coop.cutlass.store items must have a dtype")
    value_type = _canonical_type(items.dtype, feature="store")
    plan = _plan(
        group=group,
        kind=GroupLoadStoreKind.STORE,
        dtype=value_type,
        items_per_thread=items.items_per_thread,
        valid_items=valid_items,
        oob_default=None,
        offset=offset,
    )
    materialize_store(
        plan=plan,
        destination=destination,
        value=items,
        valid_items=valid_items,
        offset=offset,
    )


__all__ = ["load", "store"]
