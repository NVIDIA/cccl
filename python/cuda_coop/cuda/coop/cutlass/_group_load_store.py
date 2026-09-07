# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUTLASS group-first CUB load and store entrypoints."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any

from cuda.coop._core import (
    ArgumentBinding,
    GroupLoadStoreAlgorithm,
    LaunchFacts,
    ResultVisibility,
)

from ._thread_data import ThreadData, _coerce_thread_payload
from ._thread_group import (
    ThreadGroup,
    _require_complete_warp_partition,
    _resolve_collective_group_from_launch,
)
from ._value_metadata import (
    attach_thread_data_metadata,
    metadata_for_group,
    validate_operand_domains,
)

_SCOPE = __name__.rsplit(".", 1)[0]
_MAX_STATIC_OFFSET = (1 << 63) - 1


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


def _validate_group(group: ThreadGroup, *, primitive_name: str) -> None:
    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_SCOPE}.{primitive_name} group must be a ThreadGroup")
    if group.kind not in {"block", "warp", "threads_within_warp"}:
        raise NotImplementedError(
            f"{_SCOPE}.{primitive_name} supports complete physical block, "
            "physical-warp, and logical-warp groups only"
        )


def _launch_and_resolve_group(
    group: ThreadGroup,
    *,
    primitive_name: str,
) -> tuple[LaunchFacts, ThreadGroup]:
    from ._compiler._launch import infer_launch_facts

    launch = infer_launch_facts(
        {},
        scope=_SCOPE,
        primitive_name=primitive_name,
    )
    if not launch.is_verified("exact_block_dim"):
        raise NotImplementedError(
            f"{_SCOPE}.{primitive_name} requires exact block dimensions from "
            "verified compiler launch facts"
        )
    resolved = _resolve_collective_group_from_launch(
        group,
        launch,
        feature=primitive_name,
    )
    assert resolved.hierarchy is not None
    _require_complete_warp_partition(
        resolved,
        feature=primitive_name,
        exact_block_dim=resolved.hierarchy.block_dim,
    )
    return launch, resolved


def load(
    group: ThreadGroup,
    source: Any,
    output: ThreadData,
    /,
    *,
    algorithm: Any = "direct",
    valid_items: Any = None,
    oob_default: Any = None,
    offset: Any = None,
    temp_storage: Any = None,
) -> ThreadData:
    """Collectively load one block, physical-warp, or logical-warp tile.

    ``group`` always names the participating threads explicitly.
    ``output`` is returned after being populated. Its ``items_per_thread`` is
    the sole source of the CUB item count. ``source`` must expose a contiguous
    pointer-backed CUTLASS tensor or ``cutlass.Array`` view whose origin is the
    start of the current block tile. Physical warps automatically advance that
    origin by one warp tile per warp in the CTA.
    """

    _validate_group(group, primitive_name="load")
    if not isinstance(output, ThreadData):
        raise TypeError(f"{_SCOPE}.load output must be ThreadData")
    valid_items_binding = _classify_integer_binding(
        valid_items,
        name="valid_items",
    )
    oob_default_binding = _classify_oob_default(oob_default)
    if oob_default is not None and valid_items is None:
        raise TypeError(f"{_SCOPE}.load oob_default requires valid_items")
    offset_binding = _classify_integer_binding(offset, name="offset")
    launch, resolved_group = _launch_and_resolve_group(
        group,
        primitive_name="load",
    )

    from ._lowering import _load_store as _provider

    result = _provider.provider_load(
        group=resolved_group,
        launch=launch,
        source=source,
        output=output,
        algorithm=_normalize_algorithm(algorithm),
        valid_items=valid_items,
        valid_items_binding=valid_items_binding,
        oob_default=oob_default,
        oob_default_binding=oob_default_binding,
        offset=offset,
        offset_binding=offset_binding,
        temp_storage=temp_storage,
    )
    return attach_thread_data_metadata(
        result,
        metadata_for_group(
            resolved_group,
            visibility=ResultVisibility.PER_MEMBER,
        ),
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
    """Collectively store one value or register tile through CUB.

    A scalar is one item per group member. ``ThreadData``, an rmem tensor, or
    ``TensorSSA`` supplies its own item count. ``group`` always names the
    participating threads explicitly.
    """

    _validate_group(group, primitive_name="store")
    value = _coerce_thread_payload(
        value,
        scope=_SCOPE,
        primitive_name="store",
        arg_name="value",
        common_root_payload_kind="scalar_or_thread_data",
    )
    valid_items_binding = _classify_integer_binding(
        valid_items,
        name="valid_items",
    )
    offset_binding = _classify_integer_binding(offset, name="offset")
    launch, resolved_group = _launch_and_resolve_group(
        group,
        primitive_name="store",
    )
    validate_operand_domains(
        resolved_group,
        {"value": value},
        scope=_SCOPE,
        primitive_name="store",
    )

    from ._lowering import _load_store as _provider

    _provider.provider_store(
        group=resolved_group,
        launch=launch,
        destination=destination,
        value=value,
        algorithm=_normalize_algorithm(algorithm),
        valid_items=valid_items,
        valid_items_binding=valid_items_binding,
        offset=offset,
        offset_binding=offset_binding,
        temp_storage=temp_storage,
    )


__all__ = ["load", "store"]
