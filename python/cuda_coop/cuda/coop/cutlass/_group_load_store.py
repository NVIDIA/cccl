# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUTLASS primitive Load/Store entry points."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any

from cuda.coop._core import ArgumentBinding, GroupLoadStoreAlgorithm
from cuda.coop._core.api._payload import _validate_common_temp_storage
from cuda.coop._core.thread_group import ThreadGroup as CommonThreadGroup

from ._thread_data import ThreadData
from ._thread_group import (
    _require_complete_warp_partition,
    _resolve_primitive_group_from_launch,
)

_SCOPE = "cuda.coop.cutlass"
_MAX_STATIC_OFFSET = (1 << 63) - 1


def _resolve_group(group, algorithm, temp_storage, operation):
    if not isinstance(group, CommonThreadGroup):
        raise TypeError(f"{_SCOPE}.{operation} group must be a ThreadGroup")
    if group.kind not in {"block", "warp"}:
        raise NotImplementedError(
            f"{_SCOPE}.{operation} requires a block or physical warp group"
        )
    if group.kind == "warp" and temp_storage is not None:
        raise NotImplementedError(
            f"{_SCOPE}.{operation} explicit TempStorage is supported only for block groups"
        )
    algorithm = _normalize_algorithm(algorithm)
    if temp_storage is not None:
        _validate_common_temp_storage(operation, temp_storage)
    from ._compiler._launch import current_kernel_launch_facts

    launch = current_kernel_launch_facts()
    resolved = _resolve_primitive_group_from_launch(group, launch, feature=operation)
    _require_complete_warp_partition(
        resolved, feature=operation, exact_block_dim=launch.exact_block_dim
    )
    return resolved, launch, algorithm


def load(
    group: CommonThreadGroup,
    source: Any,
    output: ThreadData,
    /,
    *,
    algorithm: Any = "direct",
    valid_items: Any = None,
    oob_default: Any = None,
    offset: Any = None,
    temp_storage: Any = None,
) -> None:
    """Load a contiguous group tile into a writable per-thread payload.

    The payload is populated in place. Beyond ``valid_items``, initialized
    slots keep their values unless ``oob_default`` is supplied. DIRECT, STRIPED,
    and VECTORIZE require no shared scratch or synchronization. ``offset`` is measured in elements.
    """

    if not isinstance(output, ThreadData):
        raise TypeError(f"{_SCOPE}.load output must be ThreadData")
    if oob_default is not None and valid_items is None:
        raise ValueError(f"{_SCOPE}.load oob_default requires valid_items")
    group, launch, algorithm = _resolve_group(group, algorithm, temp_storage, "load")
    from ._lowering._load_store import provider_load

    provider_load(
        group=group,
        launch=launch,
        source=source,
        output=output,
        algorithm=algorithm,
        valid_items=valid_items,
        valid_items_binding=_classify_integer_binding(valid_items, name="valid_items"),
        oob_default=oob_default,
        oob_default_binding=_classify_oob_default(oob_default),
        offset=offset,
        offset_binding=_classify_integer_binding(offset, name="offset"),
        temp_storage=temp_storage,
    )


def store(
    group: CommonThreadGroup,
    destination: Any,
    value: Any,
    /,
    *,
    algorithm: Any = "direct",
    valid_items: Any = None,
    offset: Any = None,
    temp_storage: Any = None,
) -> None:
    """Store per-thread values into a contiguous group tile.

    ``valid_items`` limits the written prefix; ``offset`` is in elements.
    The value dtype must match the destination. Transpose algorithms use shared
    scratch; an optional TempStorage descriptor controls allocation and reuse.
    """

    group, launch, algorithm = _resolve_group(group, algorithm, temp_storage, "store")
    from ._lowering._load_store import provider_store

    provider_store(
        group=group,
        launch=launch,
        destination=destination,
        value=value,
        algorithm=algorithm,
        valid_items=valid_items,
        valid_items_binding=_classify_integer_binding(valid_items, name="valid_items"),
        offset=offset,
        offset_binding=_classify_integer_binding(offset, name="offset"),
        temp_storage=temp_storage,
    )


def _normalize_algorithm(algorithm: Any) -> GroupLoadStoreAlgorithm:
    token = getattr(algorithm, "value", algorithm)
    if isinstance(token, str):
        token = token.lower().replace("-", "_")
    try:
        return GroupLoadStoreAlgorithm(token)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(item.value for item in GroupLoadStoreAlgorithm)
        raise ValueError(
            f"{_SCOPE}.load/store algorithm must be one of {choices}"
        ) from exc


def _is_boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        if isinstance(value, np.bool_):
            return True
    from cutlass.base_dsl.typing import Boolean

    return isinstance(value, Boolean)


def _classify_integer_binding(value: Any, *, name: str) -> ArgumentBinding:
    if value is None:
        return ArgumentBinding.omitted()
    if _is_boolean(value):
        raise TypeError(f"{_SCOPE}.load/store {name} must be an integer")
    if isinstance(value, Integral):
        normalized = int(value)
        if name == "offset" and normalized < 0:
            raise ValueError(f"{_SCOPE}.load/store offset must be non-negative")
        if name == "offset" and normalized > _MAX_STATIC_OFFSET:
            raise ValueError(
                f"{_SCOPE}.load/store offset must fit a signed 64-bit integer"
            )
        return ArgumentBinding.static(normalized)
    from cutlass.base_dsl.typing import Integer

    if isinstance(value, Integer):
        return ArgumentBinding.runtime()
    raise TypeError(
        f"{_SCOPE}.load/store {name} must be an integer, not {type(value).__name__}"
    )


def _classify_oob_default(value: Any) -> ArgumentBinding:
    if value is None:
        return ArgumentBinding.omitted()
    if _is_boolean(value):
        raise TypeError(f"{_SCOPE}.load oob_default must be numeric, not boolean")
    if isinstance(value, Integral):
        return ArgumentBinding.static(int(value))
    if isinstance(value, Real):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(f"{_SCOPE}.load oob_default must be finite")
        return ArgumentBinding.static(normalized)
    from cutlass.base_dsl.typing import Numeric

    if isinstance(value, Numeric):
        return ArgumentBinding.runtime()
    raise TypeError(
        f"{_SCOPE}.load oob_default must be a numeric scalar, not "
        f"{type(value).__name__}"
    )
