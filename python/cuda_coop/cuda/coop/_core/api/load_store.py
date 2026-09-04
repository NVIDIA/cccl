# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable cooperative load and store entry points.

These frontends validate the shared algorithm subset and delegate one call to
the active compiler backend. ThreadData allocation and backend-specific CUB
selection remain outside this module.
"""

from __future__ import annotations

from typing import Any

from ..thread_group import ThreadGroup
from ._dispatch import (
    _backend_module_name,
    _group_primitive_marker,
    _portable_group_operation,
    _portable_selector,
    _validate_portable_operation_group,
)
from ._payload import (
    ThreadDataLike,
    _common_thread_data_extent,
    _ReadableThreadDataLike,
    _validate_common_integer_value,
    _validate_common_numeric_scalar,
    _validate_common_numeric_value,
    _validate_common_temp_storage,
)

_I32_MAX = (1 << 31) - 1
_I64_MAX = (1 << 63) - 1
_PORTABLE_LOAD_STORE_ALGORITHMS = frozenset(
    {
        "direct",
        "striped",
        "vectorize",
        "transpose",
        "warp_transpose",
        "warp_transpose_timesliced",
    }
)
_WARP_LOAD_STORE_ALGORITHMS = frozenset(
    {
        "direct",
        "striped",
        "vectorize",
        "transpose",
    }
)


def _validate_portable_load_store_options(
    operation: str,
    group: ThreadGroup,
    *,
    algorithm: Any,
    payload: Any,
    valid_items: Any,
    oob_default: Any = None,
    offset: Any,
    temp_storage: Any,
) -> None:
    """Enforce the group-dependent portable overload matrix."""

    if _backend_module_name() is None:
        return
    _validate_portable_operation_group(operation, group)
    if operation == "load" and oob_default is not None and valid_items is None:
        raise ValueError("cuda.coop.load oob_default requires valid_items")
    if valid_items is not None:
        static_valid_items = _validate_common_integer_value(
            operation,
            "valid_items",
            valid_items,
        )
        if static_valid_items is not None:
            if not 0 <= static_valid_items <= _I32_MAX:
                raise ValueError(
                    f"cuda.coop.{operation} valid_items must be between 0 "
                    "and 2147483647"
                )
            if group.static_size is not None:
                items_per_thread = (
                    _common_thread_data_extent(
                        operation,
                        "output" if operation == "load" else "value",
                        payload,
                    )
                    if isinstance(payload, _ReadableThreadDataLike)
                    else 1
                )
                tile_items = group.static_size * items_per_thread
                if static_valid_items > tile_items:
                    raise ValueError(
                        f"cuda.coop.{operation} valid_items "
                        f"{static_valid_items} exceeds group tile size "
                        f"{tile_items}"
                    )
    if oob_default is not None:
        _validate_common_numeric_scalar(operation, "oob_default", oob_default)
    if offset is not None:
        static_offset = _validate_common_integer_value(
            operation,
            "offset",
            offset,
        )
        if static_offset is not None and not 0 <= static_offset <= _I64_MAX:
            raise ValueError(
                f"cuda.coop.{operation} offset must be between 0 and "
                "9223372036854775807"
            )
    if group.kind in {"warp", "threads_within_warp"}:
        if algorithm not in _WARP_LOAD_STORE_ALGORITHMS:
            raise ValueError(
                f"cuda.coop.{operation} algorithm {algorithm!r} is supported "
                "only for block groups"
            )
        if temp_storage is not None:
            raise ValueError(
                f"cuda.coop.{operation} temp_storage is not supported for "
                "Warp groups; omit it so the implementation can provide "
                "per-group storage"
            )
    elif temp_storage is not None:
        _validate_common_temp_storage(operation, temp_storage)


@_portable_group_operation(
    "load",
    group_kinds=("block", "warp", "threads_within_warp"),
)
def load(
    group: ThreadGroup,
    source: Any,
    output: ThreadDataLike[Any],
    /,
    *,
    algorithm: Any = "direct",
    valid_items: Any = None,
    oob_default: Any = None,
    offset: Any = None,
    temp_storage: Any = None,
) -> ThreadDataLike[Any]:
    """Load values cooperatively through the compiler-selected backend.

    Use the qualified ``cuda.coop.<backend>`` API for backend-specific behavior.
    """

    algorithm = _portable_selector(
        "load", "algorithm", algorithm, _PORTABLE_LOAD_STORE_ALGORITHMS
    )
    if _backend_module_name() is not None:
        _validate_common_numeric_value(
            "load",
            "output",
            output,
            allow_untyped_thread_data=True,
            require_thread_data=True,
        )
    _validate_portable_load_store_options(
        "load",
        group,
        algorithm=algorithm,
        payload=output,
        valid_items=valid_items,
        oob_default=oob_default,
        offset=offset,
        temp_storage=temp_storage,
    )

    return _group_primitive_marker(
        "load",
        group,
        source,
        output,
        algorithm=algorithm,
        valid_items=valid_items,
        oob_default=oob_default,
        offset=offset,
        temp_storage=temp_storage,
    )


@_portable_group_operation(
    "store",
    group_kinds=("block", "warp", "threads_within_warp"),
)
def store(
    group: ThreadGroup,
    destination: Any,
    value: Any,
    /,
    *,
    algorithm: Any = "direct",
    valid_items: Any = None,
    offset: Any = None,
    temp_storage: Any = None,
) -> None:
    """Store values cooperatively through the compiler-selected backend.

    Use the qualified ``cuda.coop.<backend>`` API for backend-specific behavior.
    """

    algorithm = _portable_selector(
        "store", "algorithm", algorithm, _PORTABLE_LOAD_STORE_ALGORITHMS
    )
    if _backend_module_name() is not None:
        _validate_common_numeric_value(
            "store",
            "value",
            value,
            allow_readonly_thread_data=True,
        )
    _validate_portable_load_store_options(
        "store",
        group,
        algorithm=algorithm,
        payload=value,
        valid_items=valid_items,
        offset=offset,
        temp_storage=temp_storage,
    )

    _group_primitive_marker(
        "store",
        group,
        destination,
        value,
        algorithm=algorithm,
        valid_items=valid_items,
        offset=offset,
        temp_storage=temp_storage,
    )


__all__ = ["load", "store"]
