# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TempStorage sizing and slice planning for CuTe cooperative primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from cuda.coop._core.block import (
    BlockRowReduceGeometry,
    normalize_block_row_reduce_geometry,
)

from ._launch import (
    LAUNCH_METADATA_KEYS,
    block_dim_from_launch_metadata,
    block_dim_from_nvvm_thread_attr,
    current_kernel_block_dim,
    infer_block_dim,
)
from ._thread_data import _UNSET, ThreadData

_DEFAULT_SCOPE = "cuda.coop.cutlass"
_TEMP_STORAGE_SHARING_VALUES = frozenset(("shared", "exclusive"))
_BLOCK_ROW_REDUCE_SCALAR_BYTES = frozenset((1, 4, 8))
_SCAN_REDUCE_FAMILY = frozenset(
    (
        "exclusive_sum",
        "exclusive_scan",
        "inclusive_sum",
        "inclusive_scan",
        "scan",
        "reduce",
        "sum",
        "row_sum",
        "adjacent_difference_subtract_left",
        "adjacent_difference_subtract_right",
        "discontinuity_flag_heads",
        "discontinuity_flag_tails",
        "discontinuity_flag_heads_and_tails",
        "histogram",
        "run_length_decode",
        "shuffle",
    )
)


def _validate_block_row_reduce_launch(
    geometry: BlockRowReduceGeometry,
    kwargs: Mapping[str, Any],
    *,
    scope: str,
) -> None:
    """Match CuTe launch metadata to CUB's static row partition."""

    block_dim = infer_block_dim(
        kwargs,
        scope=scope,
        primitive_name="row_sum",
    )
    launch_threads = block_dim[0] * block_dim[1] * block_dim[2]
    try:
        geometry.validate_block_threads(launch_threads)
    except ValueError as exc:
        raise ValueError(f"{scope}.row_sum launch {exc}") from exc


def _block_row_reduce_temp_storage_bytes(
    geometry: BlockRowReduceGeometry,
    bytes_per_item: int,
) -> int:
    """Mirror CUB 3.5's warp-broadcast row-reduce storage layout."""

    if bytes_per_item not in _BLOCK_ROW_REDUCE_SCALAR_BYTES:
        supported = ", ".join(
            str(width) for width in sorted(_BLOCK_ROW_REDUCE_SCALAR_BYTES)
        )
        raise ValueError(
            f"row_sum TempStorage sizing supports scalar widths of {supported} bytes"
        )

    if geometry.warps_per_row == 1:
        # CUB selects an empty storage struct for single-warp rows. Empty C++
        # objects still occupy one byte.
        return 1

    # For a full architectural warp, WarpReduce::TempStorage is NullType, so
    # CUB first lays out one byte per logical warp and then aligns the array of
    # Uninitialized<T> row partials. All row_sum provider scalar types have
    # alignof(T) == sizeof(T).
    warp_storage_bytes = geometry.logical_warps
    partials_offset = (
        (warp_storage_bytes + bytes_per_item - 1) // bytes_per_item
    ) * bytes_per_item
    return partials_offset + geometry.logical_warps * bytes_per_item


def _block_row_reduce_temp_storage_alignment(bytes_per_item: int) -> int:
    """Return the shared-buffer alignment for a supported row scalar."""

    if bytes_per_item not in _BLOCK_ROW_REDUCE_SCALAR_BYTES:
        supported = ", ".join(
            str(width) for width in sorted(_BLOCK_ROW_REDUCE_SCALAR_BYTES)
        )
        raise ValueError(
            f"row_sum TempStorage sizing supports scalar widths of {supported} bytes"
        )
    return min(16, max(4, bytes_per_item))


def _block_row_reduce_geometry(
    kwargs: Mapping[str, Any],
    *,
    scope: str,
) -> BlockRowReduceGeometry:
    try:
        return normalize_block_row_reduce_geometry(
            rows_per_block=kwargs.get("rows_per_block"),
            warps_per_row=kwargs.get("warps_per_row"),
        )
    except ValueError as exc:
        raise ValueError(f"{scope}.row_sum TempStorage sizing: {exc}") from exc


def _block_row_reduce_temp_storage_requirement(
    geometry: BlockRowReduceGeometry,
    value: Any,
    context_thread_data: "ThreadData | None",
) -> tuple[int, int]:
    bytes_per_item = _effective_byte_width(value, context_thread_data)
    required_size = _block_row_reduce_temp_storage_bytes(
        geometry,
        bytes_per_item,
    )
    required_alignment = _block_row_reduce_temp_storage_alignment(bytes_per_item)
    return required_size, required_alignment


_SORT_KEYS_FAMILY = frozenset(
    (
        "radix_rank",
        "radix_sort_keys",
        "merge_sort_keys",
        "topk_max_keys",
        "topk_min_keys",
    )
)
_SORT_PAIRS_FAMILY = frozenset(
    (
        "radix_sort_pairs",
        "merge_sort_pairs",
        "topk_max_pairs",
        "topk_min_pairs",
    )
)
_TOPK_KEYS_FAMILY = frozenset(("topk_max_keys", "topk_min_keys"))
_TOPK_PAIRS_FAMILY = frozenset(("topk_max_pairs", "topk_min_pairs"))


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    remainder = value % alignment
    return value if remainder == 0 else (value + alignment - remainder)


def _width_from_type_token(candidate: Any) -> int | None:
    token = str(candidate)
    if token in {"i1", "ui1", "u1"}:
        return 1
    if token in {"i8", "ui8", "u8"}:
        return 8
    if token in {"i16", "ui16", "u16", "f16"}:
        return 16
    if token in {"i32", "ui32", "u32", "f32"}:
        return 32
    if token in {"i64", "ui64", "u64", "f64"}:
        return 64
    return None


def _canonical_item_width_bits(value: Any) -> int:
    if isinstance(value, ThreadData):
        if value.dtype is not None:
            width = getattr(value.dtype, "width", None)
            if isinstance(width, int) and width > 0:
                return width
            token_width = _width_from_type_token(value.dtype)
            if token_width is not None:
                return token_width
        for candidate in value._values:
            if candidate is not _UNSET:
                return _canonical_item_width_bits(candidate)
        raise ValueError(
            "ThreadData used with TempStorage requires a dtype with positive width "
            "or initialized values"
        )

    value_type = value if isinstance(value, type) else type(value)
    width = getattr(value_type, "width", None)
    if isinstance(width, int) and width > 0:
        return width

    for attr_name in ("dtype", "type", "element_type"):
        candidate = getattr(value, attr_name, None)
        width = getattr(candidate, "width", None)
        if isinstance(width, int) and width > 0:
            return width
        token_width = _width_from_type_token(candidate)
        if token_width is not None:
            return token_width

    token_width = _width_from_type_token(value_type)
    if token_width is not None:
        return token_width

    raise ValueError(
        "TempStorage item width could not be inferred; pass ThreadData with an "
        "explicit dtype width"
    )


def _items_per_thread(value: Any) -> int:
    return value.items_per_thread if isinstance(value, ThreadData) else 1


def _byte_width(value: Any) -> int:
    return max(1, (_canonical_item_width_bits(value) + 7) // 8)


def _has_explicit_width(value: Any) -> bool:
    value_type = value if isinstance(value, type) else type(value)
    width = getattr(value_type, "width", None)
    if isinstance(width, int) and width > 0:
        return True
    for attr_name in ("dtype", "type", "element_type"):
        candidate = getattr(value, attr_name, None)
        width = getattr(candidate, "width", None)
        if isinstance(width, int) and width > 0:
            return True
        if _width_from_type_token(candidate) is not None:
            return True
    return _width_from_type_token(value_type) is not None


def _effective_items_per_thread(
    value: Any,
    context_thread_data: "ThreadData | None",
) -> int:
    if isinstance(value, ThreadData):
        return value.items_per_thread
    if context_thread_data is not None:
        return context_thread_data.items_per_thread
    return 1


def _effective_byte_width(
    value: Any,
    context_thread_data: "ThreadData | None",
) -> int:
    if isinstance(value, ThreadData):
        return _byte_width(value)
    if _has_explicit_width(value):
        return _byte_width(value)
    if context_thread_data is not None:
        return _byte_width(context_thread_data)
    raise ValueError(
        "TempStorage item width could not be inferred; pass ThreadData with an "
        "explicit dtype width"
    )


def _histogram_counter_byte_width(kwargs: dict[str, Any]) -> int:
    counter_dtype = kwargs.get("counter_dtype")
    if counter_dtype is None:
        return 4
    return max(4, _byte_width(counter_dtype))


def _threads_from_launch_metadata(metadata: Any) -> int | None:
    block_dim = block_dim_from_launch_metadata(metadata)
    if block_dim is None:
        return None
    x, y, z = block_dim
    return x * y * z


def _threads_from_nvvm_thread_attr(attr: Any) -> int | None:
    block_dim = block_dim_from_nvvm_thread_attr(attr)
    if block_dim is None:
        return None
    x, y, z = block_dim
    return x * y * z


def _threads_from_current_kernel_attrs() -> int | None:
    # An upper bound is sufficient (and conservative) for storage capacity.
    block_dim = current_kernel_block_dim(allow_maxntid=True)
    if block_dim is None:
        return None
    x, y, z = block_dim
    return x * y * z


def infer_group_width(
    kwargs: dict[str, Any],
    *,
    default: int | None = None,
    scope: str = _DEFAULT_SCOPE,
) -> int:
    saw_launch_metadata = False
    for key in LAUNCH_METADATA_KEYS:
        if key not in kwargs:
            continue
        saw_launch_metadata = True
        inferred = _threads_from_launch_metadata(kwargs.get(key))
        if inferred is not None:
            return inferred

    inferred_from_kernel_attrs = _threads_from_current_kernel_attrs()
    if inferred_from_kernel_attrs is not None:
        return inferred_from_kernel_attrs

    if default is not None:
        return default
    if saw_launch_metadata:
        raise ValueError(
            f"{scope}.TempStorage sizing requires launch_metadata with a "
            "positive integer thread count; group-shape values such as "
            "threads_per_block, block, and block_dim must be compile-time "
            "constants"
        )
    raise ValueError(
        f"{scope}.TempStorage sizing requires launch_metadata or "
        "kernel reqntid/maxntid attributes"
    )


def _infer_temp_storage_threads(kwargs: dict[str, Any], *, scope: str) -> int:
    # Provider shims currently stage values in warp-indexed lanes.
    threads = max(32, infer_group_width(kwargs, default=None, scope=scope))
    return ((threads + 31) // 32) * 32


def _topk_cub_temp_storage_requirement(
    *,
    block_threads: int,
    items_per_thread: int,
    key_bytes: int,
    value_bytes: int = 0,
) -> tuple[int, int]:
    tile_items = max(1, block_threads) * max(1, items_per_thread)
    item_payload_bytes = max(1, key_bytes)
    if value_bytes > 0:
        item_payload_bytes += max(1, value_bytes)
    exchange_bytes = 8 + tile_items * item_payload_bytes
    staged_output_bytes = tile_items * max(1, key_bytes)
    if value_bytes > 0:
        staged_output_bytes += tile_items * max(1, value_bytes)
    # CUB TopK uses a 256-bin histogram plus BlockScan scratch before the
    # exchange phase. This is intentionally conservative because CuTe must
    # allocate before NVRTC can tell us the exact C++ TempStorage size.
    pass_bytes = 4096 + max(1, block_threads) * 16
    return _align_up(max(exchange_bytes, pass_bytes) + staged_output_bytes, 16), 16


def topk_temp_storage_requirement(
    primitive_name: str,
    *,
    keys: Any,
    block_threads: int,
    values: Any | None = None,
    context_thread_data: "ThreadData | None" = None,
) -> tuple[int, int]:
    """Return the canonical scratch plan for one TopK operand shape."""

    if primitive_name in _TOPK_KEYS_FAMILY:
        key_bytes = max(1, _effective_byte_width(keys, context_thread_data))
        items = _effective_items_per_thread(keys, context_thread_data)
        value_bytes = 0
    elif primitive_name in _TOPK_PAIRS_FAMILY:
        if values is None:
            raise ValueError(f"{primitive_name!r} requires values")
        key_bytes = max(1, _effective_byte_width(keys, context_thread_data))
        value_bytes = max(1, _effective_byte_width(values, context_thread_data))
        items = max(
            _effective_items_per_thread(keys, context_thread_data),
            _effective_items_per_thread(values, context_thread_data),
        )
    else:
        raise ValueError(f"{primitive_name!r} is not a TopK primitive")

    return _topk_cub_temp_storage_requirement(
        block_threads=block_threads,
        items_per_thread=items,
        key_bytes=key_bytes,
        value_bytes=value_bytes,
    )


def infer_temp_storage_requirement(
    primitive_name: str,
    kwargs: dict[str, Any],
    *,
    context_thread_data: "ThreadData | None" = None,
    scope: str = _DEFAULT_SCOPE,
) -> tuple[int, int]:
    """
    Infer per-primitive temp-storage requirement from available call metadata.

    Behavior when inference is incomplete/invalid:
    - unknown/invalid launch metadata -> error
    - unknown/unsupported value widths -> default item width = 32 bits
    - unrecognized primitive -> error instead of silently planning 0 bytes
    """
    if primitive_name == "row_sum":
        geometry = _block_row_reduce_geometry(kwargs, scope=scope)
        return _block_row_reduce_temp_storage_requirement(
            geometry,
            kwargs.get("value"),
            context_thread_data,
        )

    if primitive_name in _SCAN_REDUCE_FAMILY:
        value = kwargs.get("value")
        if primitive_name == "histogram":
            value = kwargs.get("samples")
        elif primitive_name == "run_length_decode":
            value = kwargs.get("run_values")
        bytes_per_item = _effective_byte_width(value, context_thread_data)
        if primitive_name == "run_length_decode":
            value_bytes = max(4, bytes_per_item)
            length_bytes = max(
                4,
                _effective_byte_width(
                    kwargs.get("run_lengths"),
                    context_thread_data,
                ),
            )
            alignment = min(16, max(4, value_bytes, length_bytes))
            temp_storage_threads = _infer_temp_storage_threads(kwargs, scope=scope)
            items = _effective_items_per_thread(value, context_thread_data)
            value_region_size = temp_storage_threads * items * value_bytes
            length_region_offset = _align_up(
                value_region_size,
                min(16, max(1, length_bytes)),
            )
            return (
                length_region_offset + temp_storage_threads * items * length_bytes,
                alignment,
            )
        elif primitive_name == "histogram":
            sample_bytes = max(4, bytes_per_item)
            counter_bytes = _histogram_counter_byte_width(kwargs)
            if kwargs.get("algorithm", "atomic") == "atomic":
                temp_storage_threads = _infer_temp_storage_threads(kwargs, scope=scope)
                return (
                    temp_storage_threads * counter_bytes,
                    min(16, max(4, counter_bytes)),
                )
            bytes_per_item = sample_bytes
            alignment = min(16, max(4, sample_bytes))
        else:
            bytes_per_item = max(4, bytes_per_item)
            alignment = min(16, max(4, bytes_per_item))
        temp_storage_threads = _infer_temp_storage_threads(kwargs, scope=scope)
        required_size = (
            temp_storage_threads
            * _effective_items_per_thread(value, context_thread_data)
            * bytes_per_item
        )
        return required_size, alignment

    if primitive_name in _TOPK_KEYS_FAMILY:
        return topk_temp_storage_requirement(
            primitive_name,
            keys=kwargs.get("keys"),
            block_threads=infer_group_width(kwargs, default=None, scope=scope),
            context_thread_data=context_thread_data,
        )

    if primitive_name in _TOPK_PAIRS_FAMILY:
        return topk_temp_storage_requirement(
            primitive_name,
            keys=kwargs.get("keys"),
            values=kwargs.get("values"),
            block_threads=infer_group_width(kwargs, default=None, scope=scope),
            context_thread_data=context_thread_data,
        )

    if primitive_name in _SORT_KEYS_FAMILY:
        keys = kwargs.get("keys")
        bytes_per_item = max(4, _effective_byte_width(keys, context_thread_data))
        temp_storage_threads = _infer_temp_storage_threads(kwargs, scope=scope)
        required_size = (
            temp_storage_threads
            * _effective_items_per_thread(keys, context_thread_data)
            * bytes_per_item
        )
        return required_size, min(16, max(4, bytes_per_item))

    if primitive_name in _SORT_PAIRS_FAMILY:
        keys = kwargs.get("keys")
        values = kwargs.get("values")
        key_bytes = max(4, _effective_byte_width(keys, context_thread_data))
        value_bytes = max(4, _effective_byte_width(values, context_thread_data))
        items = max(
            _effective_items_per_thread(keys, context_thread_data),
            _effective_items_per_thread(values, context_thread_data),
        )
        temp_storage_threads = _infer_temp_storage_threads(kwargs, scope=scope)
        key_region_size = temp_storage_threads * items * key_bytes
        value_region_offset = _align_up(key_region_size, min(16, max(1, value_bytes)))
        required_size = value_region_offset + temp_storage_threads * items * value_bytes
        return required_size, min(16, max(4, max(key_bytes, value_bytes)))

    raise NotImplementedError(
        f"{scope}.TempStorage sizing is not known for primitive {primitive_name!r}"
    )


def _validate_block_without_temp_storage(
    primitive_name: str,
    kwargs: dict[str, Any],
    *,
    scope: str,
) -> None:
    if primitive_name == "row_sum":
        raise ValueError(
            f"{scope}.row_sum requires TempStorage because the "
            "CUB row reduction uses shared memory"
        )

    try:
        group_width = infer_group_width(kwargs, default=None, scope=scope)
    except ValueError as exc:
        raise ValueError(
            f"{scope}.{primitive_name} requires launch_metadata or "
            "kernel reqntid/maxntid attributes when TempStorage is omitted"
        ) from exc

    if group_width > 32:
        raise ValueError(
            f"{scope}.{primitive_name} requires TempStorage for "
            f"group widths larger than one warp ({group_width} threads)"
        )


def register_block_temp_storage_use(
    primitive_name: str,
    context: Any,
    kwargs: dict[str, Any],
    *,
    scope: str,
) -> None:
    temp_storage = context.temp_storage
    if temp_storage is None:
        _validate_block_without_temp_storage(
            primitive_name,
            kwargs,
            scope=scope,
        )
        return

    if primitive_name == "row_sum":
        geometry = _block_row_reduce_geometry(kwargs, scope=scope)
        _validate_block_row_reduce_launch(geometry, kwargs, scope=scope)
        required_size, required_alignment = _block_row_reduce_temp_storage_requirement(
            geometry,
            kwargs.get("value"),
            context.thread_data,
        )
    else:
        required_size, required_alignment = infer_temp_storage_requirement(
            primitive_name,
            kwargs,
            context_thread_data=context.thread_data,
            scope=scope,
        )
    temp_storage.record_use(
        primitive_name,
        required_size_in_bytes=required_size,
        required_alignment=required_alignment,
    )


@dataclass(frozen=True)
class TempStorageSlice:
    byte_offset_in_bytes: int
    size_in_bytes: int
    alignment: int


def _merge_slice_requirements(
    existing: TempStorageSlice,
    *,
    required_size_in_bytes: int,
    required_alignment: int,
) -> TempStorageSlice:
    next_size = max(existing.size_in_bytes, required_size_in_bytes)
    next_alignment = max(existing.alignment, required_alignment)
    if existing.byte_offset_in_bytes % next_alignment != 0:
        raise RuntimeError(
            "TempStorage planner cannot strengthen alignment for an existing primitive "
            f"slice (offset={existing.byte_offset_in_bytes}, "
            f"required_alignment={next_alignment})"
        )
    return TempStorageSlice(
        byte_offset_in_bytes=existing.byte_offset_in_bytes,
        size_in_bytes=next_size,
        alignment=next_alignment,
    )


def _plan_next_slice(
    planned_size_in_bytes: int,
    *,
    required_size_in_bytes: int,
    required_alignment: int,
) -> TempStorageSlice:
    offset = _align_up(planned_size_in_bytes, required_alignment)
    return TempStorageSlice(
        byte_offset_in_bytes=offset,
        size_in_bytes=required_size_in_bytes,
        alignment=required_alignment,
    )


def _plan_primitive_slice(
    primitive_slices: dict[str, TempStorageSlice],
    primitive_name: str,
    *,
    sharing: str,
    planned_size_in_bytes: int,
    required_size_in_bytes: int,
    required_alignment: int,
) -> tuple[TempStorageSlice, int]:
    if sharing == "exclusive":
        planned = _plan_next_slice(
            planned_size_in_bytes,
            required_size_in_bytes=required_size_in_bytes,
            required_alignment=required_alignment,
        )
        return planned, planned.byte_offset_in_bytes + planned.size_in_bytes

    current = primitive_slices.get(primitive_name)
    if current is not None:
        merged = _merge_slice_requirements(
            current,
            required_size_in_bytes=required_size_in_bytes,
            required_alignment=required_alignment,
        )
        return merged, max(
            planned_size_in_bytes,
            merged.size_in_bytes,
        )

    planned = TempStorageSlice(
        byte_offset_in_bytes=0,
        size_in_bytes=required_size_in_bytes,
        alignment=required_alignment,
    )
    return planned, max(planned_size_in_bytes, planned.size_in_bytes)


@dataclass(frozen=True)
class TempStorageUse:
    primitive_name: str
    required_size_in_bytes: int
    required_alignment: int
    byte_offset_in_bytes: int


class TempStorageBase:
    """Explicit shared-memory scratch planner for CuTe collectives."""

    scope = _DEFAULT_SCOPE

    def __init__(
        self,
        size_in_bytes: int | None = None,
        alignment: int | None = None,
        auto_sync: bool | None = None,
        sharing: Literal["shared", "exclusive"] = "shared",
    ):
        deferred = size_in_bytes is None
        if size_in_bytes is not None:
            if not isinstance(size_in_bytes, int) or isinstance(size_in_bytes, bool):
                raise TypeError("TempStorage size_in_bytes must be an integer or None.")
            if size_in_bytes <= 0:
                raise ValueError(
                    "TempStorage size_in_bytes must be a positive integer."
                )

        if alignment is not None:
            if not isinstance(alignment, int) or isinstance(alignment, bool):
                raise TypeError("TempStorage alignment must be an integer or None.")
            if alignment <= 0:
                raise ValueError("TempStorage alignment must be a positive integer.")
            if alignment & (alignment - 1):
                raise ValueError("TempStorage alignment must be a power of 2.")

        if not isinstance(sharing, str):
            raise TypeError(
                "TempStorage sharing must be a string: 'shared' or 'exclusive'."
            )
        sharing_value = sharing.strip().lower()
        if sharing_value not in _TEMP_STORAGE_SHARING_VALUES:
            raise ValueError("TempStorage sharing must be 'shared' or 'exclusive'.")

        if auto_sync is not None and not isinstance(auto_sync, bool):
            raise TypeError("TempStorage auto_sync must be None/True/False.")
        if sharing_value == "exclusive" and auto_sync is True:
            raise ValueError(
                "TempStorage with sharing='exclusive' does not support auto_sync=True."
            )
        self.size_in_bytes = size_in_bytes
        self.alignment = alignment
        self.sharing = sharing_value
        self.auto_sync = (
            False
            if sharing_value == "exclusive"
            else (True if auto_sync is None else auto_sync)
        )
        self._deferred = deferred
        self._required_size_in_bytes = 0
        self._required_alignment = 1
        self._uses: list[TempStorageUse] = []
        self._primitive_slices: dict[str, TempStorageSlice] = {}
        self._planned_size_in_bytes = 0

    def record_use(
        self,
        primitive_name: str,
        *,
        required_size_in_bytes: int = 0,
        required_alignment: int = 1,
    ) -> TempStorageUse:
        if not isinstance(required_size_in_bytes, int) or isinstance(
            required_size_in_bytes, bool
        ):
            raise TypeError("required_size_in_bytes must be an integer")
        if required_size_in_bytes < 0:
            raise ValueError("required_size_in_bytes must be >= 0")
        if not isinstance(required_alignment, int) or isinstance(
            required_alignment, bool
        ):
            raise TypeError("required_alignment must be an integer")
        if required_alignment <= 0:
            raise ValueError("required_alignment must be > 0")

        if (
            self.size_in_bytes is not None
            and self.size_in_bytes < required_size_in_bytes
        ):
            raise ValueError(
                "TempStorage size_in_bytes is smaller than required by primitive "
                f"use ({self.size_in_bytes} < {required_size_in_bytes})"
            )
        if self.alignment is not None and self.alignment < required_alignment:
            raise ValueError(
                "TempStorage alignment is smaller than required by primitive use "
                f"({self.alignment} < {required_alignment})"
            )

        slice_plan, next_planned_size = _plan_primitive_slice(
            self._primitive_slices,
            primitive_name,
            sharing=self.sharing,
            planned_size_in_bytes=self._planned_size_in_bytes,
            required_size_in_bytes=required_size_in_bytes,
            required_alignment=required_alignment,
        )
        if self.size_in_bytes is not None and self.size_in_bytes < next_planned_size:
            raise ValueError(
                "TempStorage size_in_bytes is smaller than the cumulative planned "
                "primitive uses "
                f"({self.size_in_bytes} < {next_planned_size})"
            )
        self._primitive_slices[primitive_name] = slice_plan
        self._planned_size_in_bytes = next_planned_size

        self._required_size_in_bytes = max(
            self._required_size_in_bytes, required_size_in_bytes
        )
        self._required_alignment = max(self._required_alignment, required_alignment)
        use = TempStorageUse(
            primitive_name=primitive_name,
            required_size_in_bytes=required_size_in_bytes,
            required_alignment=required_alignment,
            byte_offset_in_bytes=slice_plan.byte_offset_in_bytes,
        )
        self._uses.append(use)
        return use

    @property
    def uses(self) -> tuple[TempStorageUse, ...]:
        return tuple(self._uses)

    @property
    def is_deferred(self) -> bool:
        """Whether this storage is finalized after whole-kernel tracing."""

        return self._deferred

    @property
    def required_size_in_bytes(self) -> int:
        return self._planned_size_in_bytes

    @property
    def capacity_size_in_bytes(self) -> int | None:
        return self.size_in_bytes

    @property
    def required_alignment(self) -> int:
        return (
            self.alignment if self.alignment is not None else self._required_alignment
        )

    def reset_uses(self) -> None:
        self._required_size_in_bytes = 0
        self._required_alignment = 1
        self._uses.clear()
        self._primitive_slices.clear()
        self._planned_size_in_bytes = 0

    def _snapshot_uses(self):
        return (
            self._required_size_in_bytes,
            self._required_alignment,
            list(self._uses),
            dict(self._primitive_slices),
            self._planned_size_in_bytes,
        )

    def _restore_uses(self, snapshot) -> None:
        (
            self._required_size_in_bytes,
            self._required_alignment,
            uses,
            primitive_slices,
            self._planned_size_in_bytes,
        ) = snapshot
        self._uses = list(uses)
        self._primitive_slices = dict(primitive_slices)

    def slice_for_primitive(self, primitive_name: str) -> TempStorageSlice | None:
        return self._primitive_slices.get(primitive_name)

    def slice_for_latest_use(self, primitive_name: str) -> TempStorageSlice | None:
        if not self._uses or self._uses[-1].primitive_name != primitive_name:
            return None
        use = self._uses[-1]
        return TempStorageSlice(
            byte_offset_in_bytes=use.byte_offset_in_bytes,
            size_in_bytes=use.required_size_in_bytes,
            alignment=use.required_alignment,
        )
