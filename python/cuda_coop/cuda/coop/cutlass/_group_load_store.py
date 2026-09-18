# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Validate block and physical-warp Load/Store calls for CuTe lowering.

Exact launch facts and shared group planning determine participation and
algorithm support. Binding records separate embedded constants from runtime
arguments. Block calls can use explicit scratch descriptors; physical warps
use compiler-managed scratch when their selected algorithm needs it.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any

from cuda.coop._core import ArgumentBinding, GroupLoadStoreAlgorithm
from cuda.coop._core.api._payload import _validate_common_temp_storage
from cuda.coop._core.thread_group import ThreadGroup as CommonThreadGroup
from cuda.coop._typing import (
    BlockLoadStoreAlgorithm,
    CommonThreadDataLike,
    IntegerValue,
    TempStorageLike,
    ThreadDataLike,
    ValidItems,
    WarpLoadStoreAlgorithm,
    _CommonNumericT,
)

from .._core.api.thread_group import BlockGroup, WarpGroup
from ._thread_data import ThreadData
from ._thread_group import (
    _require_complete_warp_partition,
    _resolve_primitive_group_from_launch,
)

_SCOPE = "cuda.coop.cutlass"
_MAX_STATIC_OFFSET = (1 << 63) - 1


def _resolve_group(group, algorithm, temp_storage, operation):
    """Resolve launch dimensions and reject unsupported group/storage pairs.

    Physical warps need complete 32-thread membership. Their scratch is
    compiler-owned; only block calls can use a caller's descriptor. Return the
    resolved group, exact launch facts and normalized algorithm for planning.
    """

    if not isinstance(group, CommonThreadGroup):
        raise TypeError(f"{_SCOPE}.{operation} group must be a ThreadGroup")
    if group.kind not in {"block", "warp", "threads_within_warp"}:
        raise NotImplementedError(
            f"{_SCOPE}.{operation} requires a block, physical warp, or logical warp group"
        )
    if group.kind != "block" and temp_storage is not None:
        raise NotImplementedError(
            f"{_SCOPE}.{operation} explicit TempStorage is supported only "
            "for block groups"
        )
    algorithm = _normalize_algorithm(algorithm)
    if temp_storage is not None:
        _validate_common_temp_storage(operation, temp_storage)
    from ._compiler._launch import current_kernel_launch_facts

    launch = current_kernel_launch_facts()
    resolved = _resolve_primitive_group_from_launch(
        group, launch, feature=operation
    )
    _require_complete_warp_partition(
        resolved, feature=operation, exact_block_dim=launch.exact_block_dim
    )
    return resolved, launch, algorithm


def load(
    group: BlockGroup | WarpGroup,
    source: object,
    output: ThreadDataLike[_CommonNumericT],
    /,
    *,
    algorithm: BlockLoadStoreAlgorithm | WarpLoadStoreAlgorithm = "direct",
    valid_items: ValidItems | None = None,
    oob_default: _CommonNumericT | float | None = None,
    offset: IntegerValue | None = None,
    temp_storage: TempStorageLike | None = None,
) -> None:
    """Load a contiguous group tile into a writable per-thread payload.

    Shared parameters and participation follow :func:`cuda.coop.load`. This
    implementation accepts block groups and physical warps. The output must be
    CUTLASS ThreadData; its dtype is inferred from the source, or must agree
    with it when already declared.

    Load populates the payload in the selected algorithm's layout. Beyond
    ``valid_items``, slots have unspecified values unless ``oob_default`` is
    supplied, even if initialized before Load. A default requires
    ``valid_items``. A runtime default must have the memory dtype.

    The count ranges from zero through the group's full tile size. ``offset``
    is a nonnegative element offset. Counts, offsets, and supplied defaults
    must agree within the group. For a physical warp, the compiler also adds
    that warp's tile origin within the block. Different warps can use
    different controls. The caller must provide enough accessible memory for
    the selected prefix at the resulting offset.

    The source must expose a raw pointer and a provably compact layout. A bare
    pointer object with no shape or stride metadata is also accepted, but its
    capacity is not checked. Register or local-memory tensors are rejected.
    Load reads addressable memory, such as global or shared memory. DIRECT,
    STRIPED, and VECTORIZE need no shared scratch or reuse barrier; an
    accepted block descriptor does not change that.

    Transpose algorithms use shared scratch. With no ``temp_storage``, the
    compiler allocates it and inserts a trailing reuse barrier. An explicit
    :class:`cuda.coop.cutlass.TempStorage` is supported only for block calls.
    It sets sharing and synchronization policy; its default
    ``auto_sync=False`` requires a barrier before reuse. Physical warps use
    independent scratch slices and a warp barrier.

    Examples
    --------
    Copy a partial tile between different source and destination offsets.
    The launcher accepts device pointers and a compile-time
    ``items_per_thread`` value.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/cutlass/runtime/test_qualified_load_store_examples.py
        :language: python
        :start-after: # qualified-load-store-example-begin
        :end-before: # qualified-load-store-example-end
        :dedent: 4
    """

    if not isinstance(output, ThreadData):
        raise TypeError(f"{_SCOPE}.load output must be ThreadData")
    if oob_default is not None and valid_items is None:
        raise ValueError(f"{_SCOPE}.load oob_default requires valid_items")
    group, launch, algorithm = _resolve_group(
        group, algorithm, temp_storage, "load"
    )
    from ._lowering._load_store import provider_load

    provider_load(
        group=group,
        launch=launch,
        source=source,
        output=output,
        algorithm=algorithm,
        valid_items=valid_items,
        valid_items_binding=_classify_integer_binding(
            valid_items, name="valid_items"
        ),
        oob_default=oob_default,
        oob_default_binding=_classify_oob_default(oob_default),
        offset=offset,
        offset_binding=_classify_integer_binding(offset, name="offset"),
        temp_storage=temp_storage,
    )


def store(
    group: BlockGroup | WarpGroup,
    destination: object,
    value: _CommonNumericT | CommonThreadDataLike[_CommonNumericT],
    /,
    *,
    algorithm: BlockLoadStoreAlgorithm | WarpLoadStoreAlgorithm = "direct",
    valid_items: ValidItems | None = None,
    offset: IntegerValue | None = None,
    temp_storage: TempStorageLike | None = None,
) -> None:
    """Store per-thread values into a contiguous group tile.

    Shared parameters and participation follow :func:`cuda.coop.store`. This
    implementation accepts block groups and physical warps. Each thread
    supplies a scalar or an initialized CUTLASS ThreadData payload whose dtype
    matches the destination. As with :func:`cuda.coop.store`, do not rely on
    the payload's contents after a transpose Store; copy values needed later.

    ``valid_items`` selects a prefix from zero through the group's full tile
    size. ``offset`` is a nonnegative element offset. Both must agree within
    the group. For a physical warp, the compiler also adds that warp's tile
    origin within the block. Different warps can use different controls. The
    caller must provide enough accessible destination memory for the prefix
    at the resulting offset. Items outside it are not written. The destination
    has the same raw-pointer and compact-layout requirements as :func:`load`.

    Transpose algorithms use shared scratch. With no ``temp_storage``, the
    compiler allocates it and inserts a trailing reuse barrier. An explicit
    descriptor is supported only for block calls and controls allocation and
    reuse. The caller must synchronize before reuse unless ``auto_sync=True``.
    Physical warps use independent scratch slices and a warp barrier. DIRECT,
    STRIPED, and VECTORIZE need no scratch or reuse barrier.

    Examples
    --------
    Copy a partial tile between different source and destination offsets.
    The launcher accepts device pointers and a compile-time
    ``items_per_thread`` value.

    .. literalinclude::
        ../../python/cuda_coop/tests/backends/cutlass/runtime/test_qualified_load_store_examples.py
        :language: python
        :start-after: # qualified-load-store-example-begin
        :end-before: # qualified-load-store-example-end
        :dedent: 4
    """

    group, launch, algorithm = _resolve_group(
        group, algorithm, temp_storage, "store"
    )
    from ._lowering._load_store import provider_store

    provider_store(
        group=group,
        launch=launch,
        destination=destination,
        value=value,
        algorithm=algorithm,
        valid_items=valid_items,
        valid_items_binding=_classify_integer_binding(
            valid_items, name="valid_items"
        ),
        offset=offset,
        offset_binding=_classify_integer_binding(offset, name="offset"),
        temp_storage=temp_storage,
    )


def _normalize_algorithm(algorithm: Any) -> GroupLoadStoreAlgorithm:
    """Normalize enum values and strings before variant validation."""

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
    """Recognize Python, NumPy, and DSL booleans before integer checks."""

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
    """Separate omitted, embedded, and runtime integer controls.

    Reject booleans. Validate static offset bounds here; shared planning
    checks static valid counts against the tile size. DSL integer values
    remain runtime bindings for the provider ABI.
    """

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
        f"{_SCOPE}.load/store {name} must be an integer, "
        f"not {type(value).__name__}"
    )


def _classify_oob_default(value: Any) -> ArgumentBinding:
    """Embed host numeric defaults and retain DSL inputs for runtime.

    This selects the binding form. The provider later checks compatibility
    with the memory dtype.
    """

    if value is None:
        return ArgumentBinding.omitted()
    if _is_boolean(value):
        raise TypeError(
            f"{_SCOPE}.load oob_default must be numeric, not boolean"
        )
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


__all__ = [
    "_is_boolean",
    "load",
    "store",
]
