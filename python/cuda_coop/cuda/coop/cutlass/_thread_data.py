# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Register-resident per-thread payloads for CUTLASS collectives.

``ThreadData`` is a public semantic type. Compiler-specific validation remains
lazy so constructing or inspecting the type does not activate provider code.
"""

from __future__ import annotations

import inspect as _inspect
import operator as _operator
from collections.abc import Callable, Iterator
from copy import deepcopy as _deepcopy
from typing import Any, Literal, Protocol

from ._prims import is_cutlass_array_operand as _is_cutlass_array_operand
from ._value_metadata import merge_value_metadata as _merge_group_metadata
from ._value_metadata import thread_data_metadata as _thread_data_metadata
from ._value_metadata import value_group_metadata as _value_group_metadata

_ROOT_SCOPE = "cuda.coop.cutlass"
_UNSET = object()
_THREAD_DATA_LOAD_ATTR = "__cuda_coop_thread_data_load__"
_MEMORY_PROTOCOL_ATTRS = (
    "__array_interface__",
    "__cuda_array_interface__",
    "__dlpack__",
    "__dlpack_device__",
)
_COMMON_ROOT_OPERATION_FAMILIES = {
    "reduce": frozenset({"reduce", "sum"}),
    "scan": frozenset(
        {
            "scan",
            "exclusive_sum",
            "inclusive_sum",
            "exclusive_scan",
            "inclusive_scan",
        }
    ),
}


class ThreadDataLoadSource(Protocol):
    """Producer capability that loads one payload into per-thread registers.

    Implementations own the storage-specific copy policy and expose it through
    this trace-time hook.
    """

    def __cuda_coop_thread_data_load__(self) -> Any:
        """Return a register payload that :meth:`ThreadData.load` can adapt."""
        ...


class ThreadDataTensorMetadata(Protocol):
    """Static shape and dtype used to reconstruct a CuTe ``TensorSSA``."""

    @property
    def shape(self) -> Any:
        """Static shape of the target register payload."""
        ...

    @property
    def dtype(self) -> Any:
        """Element type of the target register payload."""
        ...


class ThreadDataSource(
    ThreadDataLoadSource,
    ThreadDataTensorMetadata,
    Protocol,
):
    """Producer source that supports loading and TensorSSA reconstruction."""


def _normalize_index_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        normalized = _operator.index(value)
    except Exception:
        return None
    if isinstance(normalized, bool):
        return None
    return int(normalized)


def _normalize_group_width(value: Any) -> int | None:
    try:
        normalized = _normalize_index_int(value)
    except Exception:
        return None
    if normalized is not None and normalized > 0:
        return normalized
    return None


def _get_optional_metadata_attr(value: Any, attr_name: str) -> Any:
    try:
        return getattr(value, attr_name, None)
    except Exception:
        return None


def _first_optional_metadata_attr(value: Any, attr_names: tuple[str, ...]) -> Any:
    for attr_name in attr_names:
        candidate = _get_optional_metadata_attr(value, attr_name)
        if candidate is not None:
            return candidate
    return None


def _require_like_metadata_attr(value: Any, attr_name: str, *, source: str) -> Any:
    try:
        metadata = getattr(value, attr_name)
    except Exception as exc:
        raise TypeError(
            f"{source} like= source must expose accessible {attr_name} metadata"
        ) from exc
    if metadata is None:
        raise TypeError(
            f"{source} like= source must expose non-None {attr_name} metadata"
        )
    return metadata


def _infer_1d_static_extent(shape: Any) -> int | None:
    inferred = _normalize_group_width(shape)
    if inferred is not None:
        return inferred
    if isinstance(shape, (tuple, list)) and len(shape) == 1:
        return _normalize_group_width(shape[0])
    return None


def _infer_static_extent(shape: Any) -> int | None:
    inferred = _normalize_group_width(shape)
    if inferred is not None:
        return inferred
    if not isinstance(shape, (tuple, list)) or len(shape) == 0:
        return None

    extent = 1
    for dimension in shape:
        inferred = _infer_static_extent(dimension)
        if inferred is None:
            return None
        extent *= inferred
    return extent


def _infer_fragment_items_per_thread(
    fragment: Any,
    *,
    allow_nested: bool = True,
) -> int | None:
    infer_extent = _infer_static_extent if allow_nested else _infer_1d_static_extent

    # Prefer explicit tensor-like shape metadata when available.
    for attr_name in ("shape",):
        candidate = _get_optional_metadata_attr(fragment, attr_name)
        inferred = infer_extent(candidate)
        if inferred is not None:
            return inferred

    # Fall back to layout/type metadata for tensor-like fragments.
    layout = _get_optional_metadata_attr(fragment, "layout")
    if layout is not None:
        inferred = infer_extent(_get_optional_metadata_attr(layout, "shape"))
        if inferred is not None:
            return inferred

    fragment_type = _get_optional_metadata_attr(fragment, "type")
    if fragment_type is not None:
        inferred = infer_extent(_get_optional_metadata_attr(fragment_type, "shape"))
        if inferred is not None:
            return inferred

    return None


def _infer_vector_items_per_thread(vector: Any) -> int | None:
    numel = _get_optional_metadata_attr(vector, "numel")
    if callable(numel):
        try:
            inferred = _normalize_group_width(numel())
        except Exception:
            inferred = None
        if inferred is not None:
            return inferred

    inferred = _infer_fragment_items_per_thread(vector, allow_nested=False)
    if inferred is not None:
        return inferred

    candidate = _get_optional_metadata_attr(vector, "_shape")
    return _infer_1d_static_extent(candidate)


def _is_register_fragment(value: Any) -> bool:
    try:
        memspace = getattr(value, "memspace", None)
    except Exception:
        return False
    return _is_register_memory_space(memspace)


def _is_register_memory_space(memspace: Any) -> bool:
    register_spaces = []
    try:
        from cutlass import AddressSpace as _AddressSpace

        register_spaces.append(_AddressSpace.rmem)
    except Exception:
        pass
    try:
        from cutlass._mlir.dialects.cute import AddressSpace as _CuteAddressSpace

        register_spaces.append(_CuteAddressSpace.rmem)
    except Exception:
        pass

    for register_space in register_spaces:
        try:
            if memspace == register_space:
                return True
        except Exception:
            continue
    return False


def _has_memory_space(value: Any) -> bool:
    for attr_name in ("memspace", "space"):
        try:
            attr = getattr(value, attr_name, None)
        except Exception:
            try:
                _inspect.getattr_static(value, attr_name)
            except AttributeError:
                continue
            return True
        if attr is not None:
            return True
    return False


def _has_memory_protocol(value: Any) -> bool:
    for attr_name in _MEMORY_PROTOCOL_ATTRS:
        try:
            _inspect.getattr_static(value, attr_name)
        except AttributeError:
            pass
        else:
            return True

        try:
            attr = getattr(value, attr_name, None)
        except Exception:
            return True
        if attr is not None:
            return True
    return False


def _is_memory_backed_payload(value: Any) -> bool:
    return (
        _is_cutlass_array_payload(value)
        or _has_memory_space(value)
        or _has_memory_protocol(value)
    )


def _is_cutlass_array_payload(value: Any) -> bool:
    return _is_cutlass_array_operand(value)


def _is_cutlass_dsl_dtype(dtype: Any) -> bool:
    return isinstance(dtype, type) and dtype.__module__ == "cutlass.base_dsl.typing"


def _is_ordinary_scalar_dtype(dtype: Any) -> bool:
    if any(dtype is candidate for candidate in (bool, int, float, complex)):
        return True
    if not isinstance(dtype, type) or dtype.__module__.split(".", 1)[0] != "numpy":
        return False

    # NumPy is not a base dependency of cuda-coop.  Import it only when the
    # caller has already supplied a NumPy-owned scalar type so importing the
    # qualified CUTLASS namespace remains dependency-light.
    try:
        import numpy as np
    except ImportError:
        return False
    return issubclass(dtype, np.generic)


def _coerce_payload_values_to_dtype(
    values: tuple[Any, ...],
    dtype: Any,
    *,
    source: str,
) -> tuple[Any, ...]:
    if not (_is_cutlass_dsl_dtype(dtype) or _is_ordinary_scalar_dtype(dtype)):
        return values

    coerced = []
    for idx, value in enumerate(values):
        if value is _UNSET:
            coerced.append(value)
            continue
        try:
            if isinstance(value, dtype):
                coerced.append(value)
            else:
                coerced.append(dtype(value))
        except Exception as exc:
            raise TypeError(
                f"{source} dtype cannot be applied to payload item {idx}"
            ) from exc
    return tuple(coerced)


def _validate_items_per_thread(value: Any) -> int:
    normalized = _normalize_index_int(value)
    if normalized is None:
        raise TypeError("items_per_thread must be an integer")
    if normalized <= 0:
        raise ValueError("items_per_thread must be a positive integer")
    return normalized


def _resolve_items_per_thread(
    *,
    explicit: Any,
    infer: Callable[[], int | None],
    source: str,
    missing_message: str,
) -> int:
    explicit = None if explicit is None else _validate_items_per_thread(explicit)
    inferred = infer()
    if explicit is None:
        if inferred is not None:
            return inferred
        raise ValueError(missing_message)
    if inferred is not None and explicit != inferred:
        raise ValueError(
            f"{source} items_per_thread does not match payload item count "
            f"({explicit} != {inferred})"
        )
    return explicit


def _resolve_export_shape(
    shape: Any,
    *,
    items_per_thread: int,
    source: str,
) -> Any:
    if shape is None:
        return (items_per_thread,)
    inferred = _infer_static_extent(shape)
    if inferred is None:
        raise ValueError(f"{source} shape must be positive and fully static")
    if inferred != items_per_thread:
        raise ValueError(
            f"{source} shape must contain exactly items_per_thread elements "
            f"({inferred} != {items_per_thread})"
        )
    return shape


def _resolve_export_dtype(dtype: Any, *, fallback: Any, source: str) -> Any:
    dtype = fallback if dtype is None else dtype
    if dtype is None:
        raise TypeError(f"{source} requires dtype when ThreadData.dtype is not set")
    try:
        from cutlass.base_dsl.typing import Numeric as _Numeric
    except Exception as exc:
        raise TypeError(f"{source} requires a CUTLASS Numeric dtype") from exc
    if not isinstance(dtype, type) or not issubclass(dtype, _Numeric):
        raise TypeError(f"{source} dtype must be a CUTLASS Numeric type")
    return dtype


def _validate_export_domain(value: ThreadData, *, source: str) -> None:
    metadata = _thread_data_metadata(value)
    if metadata is not None and metadata.defined_domain.constraints:
        raise ValueError(
            f"{source} requires ThreadData values defined for all calling threads; "
            "root-only and incomplete-group results must remain ThreadData"
        )


class ThreadData:
    """Per-thread register payload used by CUTLASS cooperative primitives.

    ``ThreadData`` carries the number of logical items owned by each thread,
    optional dtype metadata, and the per-item register values traced by a
    CUTLASS DSL kernel. Block- and warp-group primitives infer
    ``items_per_thread`` from this object, so users specify the item count once
    when constructing the payload.
    """

    def __init__(
        self,
        items_per_thread: int,
        dtype: Any = None,
        *,
        values: tuple[Any, ...] | list[Any] | None = None,
    ):
        items_per_thread = _validate_items_per_thread(items_per_thread)

        from cuda.coop._core.api import _common_root_operation_name

        common_root = _common_root_operation_name() == "ThreadData"
        if dtype is not None and common_root:
            from ._compiler._state import _validate_common_root_numeric_dtype

            _validate_common_root_numeric_dtype(dtype, operation="ThreadData")

        if values is not None:
            if not isinstance(values, (tuple, list)):
                raise TypeError("values must be a tuple/list when provided")
            if len(values) != items_per_thread:
                raise ValueError(
                    "values length must match items_per_thread "
                    f"({len(values)} != {items_per_thread})"
                )

        self.items_per_thread = items_per_thread
        self.dtype = dtype
        self._values = (
            list(values)
            if values is not None
            else [_UNSET for _ in range(self.items_per_thread)]
        )
        self._common_root = common_root
        self._item_group_metadata = [
            _value_group_metadata(value) for value in self._values
        ]
        self._refresh_group_metadata()

    @classmethod
    def from_values(cls, *values: Any, dtype: Any = None) -> ThreadData:
        if len(values) == 0:
            raise ValueError("ThreadData.from_values requires at least one value")
        return cls(len(values), dtype=dtype, values=list(values))

    @classmethod
    def from_fn(
        cls,
        items_per_thread: int,
        fn: Callable[[int], Any],
        *,
        dtype: Any = None,
    ) -> ThreadData:
        """Build ThreadData by calling ``fn(item_idx)`` for each item."""
        items_per_thread = _validate_items_per_thread(items_per_thread)
        if not callable(fn):
            raise TypeError("ThreadData.from_fn requires a callable")

        values = []
        for item_idx in range(items_per_thread):
            try:
                values.append(fn(item_idx))
            except Exception as exc:
                raise TypeError(
                    f"ThreadData.from_fn callable failed for item {item_idx}"
                ) from exc
        values = _coerce_payload_values_to_dtype(
            tuple(values),
            dtype,
            source="ThreadData.from_fn",
        )
        return cls.from_values(*values, dtype=dtype)

    @classmethod
    def from_register_tensor(
        cls,
        fragment: Any,
        *,
        items_per_thread: int | None = None,
        dtype: Any = None,
    ) -> ThreadData:
        """
        Build ThreadData from a per-thread register-backed CuTe tensor fragment.

        This bridge is intentionally strict: it only accepts rmem fragments so users
        cannot accidentally pass global/shared tensors as cooperative thread payload.
        Passing an explicit CUTLASS DSL dtype casts the extracted register values
        to that dtype.
        """
        try:
            memspace = getattr(fragment, "memspace", None)
        except Exception as exc:
            raise TypeError(
                "ThreadData.from_register_tensor requires a register-memory "
                "(rmem) CuTe tensor fragment"
            ) from exc
        if not _is_register_memory_space(memspace):
            raise TypeError(
                "ThreadData.from_register_tensor requires a register-memory "
                "(rmem) CuTe tensor fragment"
            )

        items_per_thread = _resolve_items_per_thread(
            explicit=items_per_thread,
            infer=lambda: _infer_fragment_items_per_thread(fragment),
            source="ThreadData.from_register_tensor",
            missing_message=(
                "ThreadData.from_register_tensor could not infer items_per_thread "
                "from fragment shape; pass items_per_thread explicitly"
            ),
        )

        if dtype is None:
            dtype = _get_optional_metadata_attr(fragment, "element_type")

        values = tuple(fragment[idx] for idx in range(items_per_thread))
        values = _coerce_payload_values_to_dtype(
            values,
            dtype,
            source="ThreadData.from_register_tensor",
        )
        return cls.from_values(*values, dtype=dtype)

    @classmethod
    def from_vector(
        cls,
        vector: Any,
        *,
        items_per_thread: int | None = None,
        dtype: Any = None,
    ) -> ThreadData:
        """
        Build ThreadData from a register vector-like object.

        This bridge accepts CuTe ``TensorSSA`` values and CUTLASS register
        vectors that expose a static item count through ``numel()`` or 1-D shape
        metadata and support integer indexing for each per-thread item.
        """
        if _is_memory_backed_payload(vector):
            raise TypeError(
                "ThreadData.from_vector requires a CUTLASS vector-like "
                "per-thread payload; use ThreadData.from_register_tensor for "
                "CuTe register fragments, or a group-first load for cutlass.Array "
                "values and memory tensors"
            )

        items_per_thread = _resolve_items_per_thread(
            explicit=items_per_thread,
            infer=lambda: _infer_vector_items_per_thread(vector),
            source="ThreadData.from_vector",
            missing_message=(
                "ThreadData.from_vector could not infer items_per_thread "
                "from vector shape; pass items_per_thread explicitly"
            ),
        )

        if dtype is None:
            dtype = _first_optional_metadata_attr(
                vector,
                ("dtype", "_dtype", "element_type"),
            )

        try:
            values = tuple(vector[idx] for idx in range(items_per_thread))
        except Exception as exc:
            raise TypeError(
                "ThreadData.from_vector requires integer-indexable vector items"
            ) from exc
        values = _coerce_payload_values_to_dtype(
            values,
            dtype,
            source="ThreadData.from_vector",
        )
        return cls.from_values(*values, dtype=dtype)

    @classmethod
    def from_payload(
        cls,
        payload: Any,
        *,
        items_per_thread: int | None = None,
        dtype: Any = None,
    ) -> ThreadData:
        """
        Build ThreadData from a backend-specific per-thread register payload.

        This is the common CUTLASS-root bridge for payloads that are already
        thread-local: CuTe register-memory fragments are adapted through
        :meth:`from_register_tensor`, while CuTe ``TensorSSA`` and CUTLASS
        vector-like values are adapted through :meth:`from_vector`.
        Memory-backed tensors and arrays remain outside this boundary; use
        group-first load/store helpers to move them into per-thread registers
        first.
        """
        if isinstance(payload, cls):
            if items_per_thread is not None:
                items_per_thread = _validate_items_per_thread(items_per_thread)
                if payload.items_per_thread != items_per_thread:
                    raise ValueError(
                        "ThreadData.from_payload items_per_thread does not match "
                        "payload.items_per_thread"
                    )
            if dtype is None or payload.dtype == dtype:
                return payload
            if payload.dtype is not None:
                raise TypeError("ThreadData.from_payload dtype does not match payload")
            if payload._common_root:
                from ._compiler._state import _validate_common_root_numeric_dtype

                _validate_common_root_numeric_dtype(
                    dtype,
                    operation="ThreadData",
                )
            values = _coerce_payload_values_to_dtype(
                tuple(payload._values),
                dtype,
                source="ThreadData.from_payload",
            )
            result = cls(payload.items_per_thread, dtype=dtype, values=values)
            result = payload._preserve_common_root(result)
            return payload._preserve_group_metadata(result)

        if _is_register_fragment(payload):
            return cls.from_register_tensor(
                payload,
                items_per_thread=items_per_thread,
                dtype=dtype,
            )
        if _is_memory_backed_payload(payload):
            raise TypeError(
                "ThreadData.from_payload requires a per-thread register payload; "
                "use ThreadData.from_register_tensor for CuTe register fragments, "
                "or a group-first load for cutlass.Array values and memory tensors"
            )
        return cls.from_vector(
            payload,
            items_per_thread=items_per_thread,
            dtype=dtype,
        )

    @classmethod
    def load(
        cls,
        source: ThreadDataLoadSource,
        *,
        items_per_thread: int | None = None,
        dtype: Any = None,
    ) -> ThreadData:
        """Load a producer-owned source into per-thread registers.

        The source must explicitly implement the private
        ``__cuda_coop_thread_data_load__`` tracing hook. The producer owns any
        required tensor-memory-to-register copy, including its copy atom,
        thread partition, readiness, and lifetime. The hook must return either
        ``ThreadData`` or a statically sized per-thread register payload
        accepted by :meth:`from_payload`. The hook is invoked exactly once by
        each call during DSL tracing; a single-use producer source is
        responsible for rejecting a second call.

        Bare memory-backed tensors are intentionally not interpreted here:
        their address space and shape do not provide enough information to
        safely select or schedule a register load.

        ``items_per_thread`` and ``dtype`` are optional static constraints on
        the returned register payload, matching :meth:`from_payload`.
        """

        if items_per_thread is not None:
            items_per_thread = _validate_items_per_thread(items_per_thread)
        source_type = type(source).__name__
        try:
            _inspect.getattr_static(source, _THREAD_DATA_LOAD_ATTR)
        except AttributeError:
            if _is_memory_backed_payload(source):
                raise TypeError(
                    f"ThreadData.load source {source_type} requires a producer-provided "
                    f"{_THREAD_DATA_LOAD_ATTR} hook; bare memory-backed tensors "
                    "including TMEM do not carry the copy atom, thread partition, "
                    "readiness, and lifetime needed for a safe register load"
                ) from None
            raise TypeError(
                "ThreadData.load requires source to define the producer-provided "
                f"{_THREAD_DATA_LOAD_ATTR} hook; got {source_type}"
            ) from None

        try:
            load_payload = getattr(source, _THREAD_DATA_LOAD_ATTR)
        except Exception as exc:
            raise TypeError(
                f"ThreadData.load could not access {source_type}'s producer-provided "
                f"{_THREAD_DATA_LOAD_ATTR} hook"
            ) from exc
        if not callable(load_payload):
            raise TypeError(
                f"ThreadData.load source {source_type} requires the producer-provided "
                f"{_THREAD_DATA_LOAD_ATTR} hook to be callable"
            )

        payload = load_payload()
        payload_type = type(payload).__name__

        try:
            return cls.from_payload(
                payload,
                items_per_thread=items_per_thread,
                dtype=dtype,
            )
        except TypeError as exc:
            raise TypeError(
                f"ThreadData.load hook on {source_type} returned {payload_type}; "
                "expected ThreadData or a statically sized per-thread register "
                f"payload: {exc}"
            ) from exc
        except ValueError as exc:
            raise ValueError(
                f"ThreadData.load hook on {source_type} returned {payload_type}; "
                "expected ThreadData or a statically sized per-thread register "
                f"payload: {exc}"
            ) from exc

    def to_tensor_ssa(
        self,
        *,
        dtype: Any = None,
        shape: Any = None,
        like: ThreadDataTensorMetadata | None = None,
    ) -> Any:
        """Materialize this payload as a register-only CuTe ``TensorSSA``.

        The default shape is ``(items_per_thread,)``. An explicit shape may be
        nested, but it must be positive, fully static, and contain the same
        number of logical elements. The result preserves ``ThreadData`` flat
        item order; it does not recover an input fragment's original shape.

        ``like`` supplies missing ``shape`` and ``dtype`` metadata from another
        static payload or producer capability. Explicit arguments take
        precedence.
        """

        source = "ThreadData.to_tensor_ssa"
        _validate_export_domain(self, source=source)
        if like is not None:
            if dtype is None:
                dtype = _require_like_metadata_attr(
                    like,
                    "dtype",
                    source=source,
                )
            if shape is None:
                shape = _require_like_metadata_attr(
                    like,
                    "shape",
                    source=source,
                )
        dtype = _resolve_export_dtype(dtype, fallback=self.dtype, source=source)
        shape = _resolve_export_shape(
            shape,
            items_per_thread=self.items_per_thread,
            source=source,
        )
        values = self.values(source)

        from cutlass import Vector as _Vector
        from cutlass import cute as _cute

        vector = _Vector.from_elements(values, dtype)
        return _cute.TensorSSA(vector, shape, dtype)

    def to_register_tensor(
        self,
        *,
        dtype: Any = None,
        shape: Any = None,
    ) -> Any:
        """Materialize this payload as a fresh mutable CuTe rmem tensor.

        This creates addressable register-memory storage and stores a newly
        assembled ``TensorSSA`` into it. It does not alias this ``ThreadData``
        object, and code generation may still spill register storage locally.
        """

        ssa = self.to_tensor_ssa(dtype=dtype, shape=shape)

        from cutlass import cute as _cute

        result = _cute.make_rmem_tensor_like(ssa)
        result.store(ssa)
        return result

    def __len__(self) -> int:
        return self.items_per_thread

    def __getitem__(self, idx: int) -> Any:
        return self._values[idx]

    def __setitem__(self, idx: int, value: Any) -> None:
        if self._common_root:
            from ._compiler._state import _validate_common_root_numeric_dtype

            _validate_common_root_numeric_dtype(value, operation="ThreadData")
        item_metadata = _value_group_metadata(value)
        self._values[idx] = value
        self._item_group_metadata[idx] = item_metadata
        self._refresh_group_metadata()

    def _refresh_group_metadata(self) -> None:
        self._group_metadata = _merge_group_metadata(self._item_group_metadata)

    def _set_group_metadata(self, metadata: Any) -> None:
        self._item_group_metadata = [metadata] * self.items_per_thread
        self._group_metadata = metadata

    def _preserve_group_metadata(self, result: ThreadData) -> ThreadData:
        result._item_group_metadata = list(self._item_group_metadata)
        result._refresh_group_metadata()
        return result

    def _preserve_common_root(self, result: ThreadData) -> ThreadData:
        result._common_root = self._common_root
        return result

    def __copy__(self) -> ThreadData:
        result = ThreadData(
            self.items_per_thread,
            dtype=self.dtype,
            values=list(self._values),
        )
        result = self._preserve_common_root(result)
        return self._preserve_group_metadata(result)

    def __deepcopy__(self, memo: dict[int, Any]) -> ThreadData:
        result = ThreadData(
            self.items_per_thread,
            dtype=_deepcopy(self.dtype, memo),
            values=_deepcopy(self._values, memo),
        )
        result = self._preserve_common_root(result)
        memo[id(self)] = result
        return self._preserve_group_metadata(result)

    def _require_values(self, primitive_name: str | None) -> list[Any]:
        missing = [idx for idx, value in enumerate(self._values) if value is _UNSET]
        if missing:
            context = (
                "ThreadData iteration"
                if primitive_name is None
                else f"{_ROOT_SCOPE}.{primitive_name}"
            )
            raise ValueError(
                f"{context} requires ThreadData values to be initialized before use; "
                "missing index(es): " + ", ".join(str(i) for i in missing)
            )
        return self._values

    def values(self, primitive_name: str) -> tuple[Any, ...]:
        return tuple(self._require_values(primitive_name))

    def __iter__(self) -> Iterator[Any]:
        return iter(self._require_values(None))

    def _new_uninitialized(self, *, dtype: Any = None) -> ThreadData:
        resolved_dtype = self.dtype if dtype is None else dtype
        if self._common_root and resolved_dtype is not None:
            from ._compiler._state import _validate_common_root_numeric_dtype

            _validate_common_root_numeric_dtype(
                resolved_dtype,
                operation="ThreadData",
            )
        result = ThreadData(
            self.items_per_thread,
            dtype=resolved_dtype,
        )
        return self._preserve_common_root(result)


def _is_thread_payload_candidate(value: Any) -> bool:
    if _is_register_fragment(value) or _is_memory_backed_payload(value):
        return True
    # A callable numel() is the stable TensorSSA/vector recognition hook even
    # when it reports an invalid or non-static extent. Keep recognition
    # separate from successful extent inference so malformed register vectors
    # produce the same targeted conversion error as malformed rmem fragments
    # instead of falling through as scalar operands.
    if callable(_get_optional_metadata_attr(value, "numel")):
        return True
    return _infer_vector_items_per_thread(value) is not None


def _coerce_thread_payload(
    value: Any,
    *,
    scope: str,
    primitive_name: str,
    arg_name: str,
    common_root_payload_kind: Literal[
        "thread_data",
        "scalar_or_thread_data",
    ]
    | None = None,
) -> Any:
    """Adapt a backend register payload without widening the common contract."""

    if common_root_payload_kind is not None:
        from cuda.coop._core.api import _common_root_operation_name

        common_operation = _common_root_operation_name()
        family = _COMMON_ROOT_OPERATION_FAMILIES.get(
            primitive_name,
            frozenset({primitive_name}),
        )
        if common_operation in family:
            assert common_operation is not None
            if common_root_payload_kind == "thread_data":
                if not isinstance(value, ThreadData):
                    raise TypeError(
                        f"cuda.coop.{common_operation} requires a fixed-size "
                        f"ThreadData {arg_name} payload in the portable API; use "
                        "cuda.coop.cutlass for backend-qualified scalar or "
                        "register payload support"
                    )
            elif common_root_payload_kind == "scalar_or_thread_data":
                if not isinstance(value, ThreadData) and _is_thread_payload_candidate(
                    value
                ):
                    raise TypeError(
                        f"cuda.coop.{common_operation} accepts only a scalar or "
                        f"fixed-size ThreadData {arg_name} payload in the portable API; "
                        "use cuda.coop.cutlass for backend-qualified register "
                        "payload support"
                    )
            else:  # pragma: no cover - the annotation defines the private contract.
                raise ValueError(
                    "common_root_payload_kind must be 'thread_data' or "
                    "'scalar_or_thread_data'"
                )

    if isinstance(value, ThreadData) or not _is_thread_payload_candidate(value):
        return value
    try:
        return ThreadData.from_payload(value)
    except Exception as exc:
        raise TypeError(
            f"{scope}.{primitive_name} could not auto-convert "
            f"'{arg_name}' payload to ThreadData: {exc}"
        ) from exc
