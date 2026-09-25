# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Per-thread payloads and qualified CuTe register conversions."""

from __future__ import annotations

import inspect as _inspect
import operator as _operator
from collections.abc import Callable, Iterator
from copy import deepcopy as _deepcopy
from typing import Any, Literal

from cuda.coop._core.api._payload import _normalize_alignment

_ROOT_SCOPE = "cuda.coop.cutlass"
_UNSET = object()
_MEMORY_PROTOCOL_ATTRS = (
    "__array_interface__",
    "__cuda_array_interface__",
    "__dlpack__",
    "__dlpack_device__",
)
_COMMON_ROOT_OPERATION_FAMILIES = {
    "load": frozenset({"load"}),
    "store": frozenset({"store"}),
}


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
        from cutlass import AddressSpace as CutlassAddressSpace

        register_spaces.append(CutlassAddressSpace.rmem)
    except Exception:
        pass
    try:
        from cutlass._mlir.dialects.cute import AddressSpace as CuteAddressSpace

        register_spaces.append(CuteAddressSpace.rmem)
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
    return _has_memory_space(value) or _has_memory_protocol(value)


def _is_cutlass_dsl_dtype(dtype: Any) -> bool:
    return isinstance(dtype, type) and dtype.__module__ == "cutlass.base_dsl.typing"


def _is_ordinary_scalar_dtype(dtype: Any) -> bool:
    if any(dtype is candidate for candidate in (bool, int, float, complex)):
        return True
    if not isinstance(dtype, type) or dtype.__module__.split(".", 1)[0] != "numpy":
        return False

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
    from ._compiler._types import ALL_PROVIDER_TYPES, canonical_dsl_type

    resolved = canonical_dsl_type(dtype)
    if resolved not in ALL_PROVIDER_TYPES:
        raise TypeError(f"{source} dtype must be a supported numeric type")
    return resolved


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
        alignment: int | None = None,
    ) -> None:
        items_per_thread = _validate_items_per_thread(items_per_thread)

        from cuda.coop._core.api._dispatch import _common_root_operation_name

        common_root = _common_root_operation_name() == "ThreadData"
        if dtype is not None and common_root:
            from ._compiler._types import _validate_common_root_numeric_dtype

            _validate_common_root_numeric_dtype(dtype, operation="ThreadData")

        if values is not None:
            if not isinstance(values, (tuple, list)):
                raise TypeError("values must be a tuple/list when provided")
            if len(values) != items_per_thread:
                raise ValueError(
                    "values length must match items_per_thread "
                    f"({len(values)} != {items_per_thread})"
                )

        self.alignment = _normalize_alignment(alignment)
        self.items_per_thread = items_per_thread
        self.dtype = dtype
        self._values = (
            list(values)
            if values is not None
            else [_UNSET for _ in range(self.items_per_thread)]
        )
        self._common_root = common_root

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
                "CuTe register fragments, or a group-first load for memory tensors"
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

        This qualified bridge accepts payloads that are already
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
                from ._compiler._types import _validate_common_root_numeric_dtype

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
            return result

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
                "or a group-first load for memory tensors"
            )
        return cls.from_vector(
            payload,
            items_per_thread=items_per_thread,
            dtype=dtype,
        )

    def to_tensor_ssa(
        self,
        *,
        dtype: Any = None,
        shape: Any = None,
    ) -> Any:
        """Materialize this payload as a register-only CuTe ``TensorSSA``.

        The default shape is ``(items_per_thread,)``. An explicit shape may be
        nested, but it must be positive, fully static, and contain the same
        number of logical elements. The result preserves ``ThreadData`` flat
        item order; it does not recover an input fragment's original shape.

        """

        source = "ThreadData.to_tensor_ssa"
        dtype = _resolve_export_dtype(dtype, fallback=self.dtype, source=source)
        shape = _resolve_export_shape(
            shape,
            items_per_thread=self.items_per_thread,
            source=source,
        )
        values = self.values(source)

        import cutlass.cute as _cute
        from cutlass import Vector

        vector = Vector.from_elements(values, dtype)
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

        result = _make_rmem_tensor(ssa.shape, ssa.dtype, self.alignment)
        result.store(ssa)
        return result

    def __len__(self) -> int:
        return self.items_per_thread

    def __getitem__(self, idx: int) -> Any:
        idx = self._index(idx)
        value = self._values[idx]
        if value is _UNSET:
            raise ValueError(f"ThreadData item {idx} is uninitialized")
        return value

    def _index(self, index: Any) -> int:
        normalized = _normalize_index_int(index)
        if normalized is None:
            raise TypeError("ThreadData index must be a compile-time integer")
        if not -self.items_per_thread <= normalized < self.items_per_thread:
            raise IndexError("ThreadData index out of range")
        return normalized

    def __setitem__(self, idx: int, value: Any) -> None:
        idx = self._index(idx)
        if self._common_root:
            from ._compiler._types import _validate_common_root_numeric_dtype

            _validate_common_root_numeric_dtype(value, operation="ThreadData")
        self._values[idx] = value

    def _dynamic_values(self) -> tuple[type, tuple[Any, ...]]:
        from ._compiler import _types

        resolve_type = _types.make_provider_type_resolver(
            scope=_ROOT_SCOPE, root_scope=_ROOT_SCOPE, namespace="ThreadData"
        )
        value_type, values = _types.resolve_thread_data_value_type(
            self,
            allowed=_types.ALL_PROVIDER_TYPES,
            feature="ThreadData control flow",
            scope=_ROOT_SCOPE,
            resolve_type=resolve_type,
        )
        converted = []
        for value in values:
            plain = _types.coerce_plain_scalar(
                value,
                value_type,
                name="ThreadData control-flow value",
                scope=_ROOT_SCOPE,
                allow_nonfinite=True,
            )
            converted.append(
                value_type(value) if plain is _types._NOT_PLAIN_SCALAR else plain
            )
        return value_type, tuple(converted)

    def __extract_mlir_values__(self) -> list[Any]:
        """Carry initialized scalar lanes through CuTe control flow."""
        _, values = self._dynamic_values()
        return [value.ir_value() for value in values]

    def __new_from_mlir_values__(self, values: list[Any]) -> ThreadData:
        """Rebuild lanes while preserving static payload and root metadata."""
        if len(values) != self.items_per_thread:
            raise ValueError("ThreadData control flow requires one value per item")
        from ._compiler._types import thread_data_output_dtype

        value_type, _ = self._dynamic_values()
        result = ThreadData(
            self.items_per_thread,
            dtype=thread_data_output_dtype(self, value_type),
            values=[value_type(value) for value in values],
            alignment=self.alignment,
        )
        return self._preserve_common_root(result)

    def _preserve_common_root(self, result: ThreadData) -> ThreadData:
        result._common_root = self._common_root
        result.alignment = self.alignment
        return result

    def __copy__(self) -> ThreadData:
        result = ThreadData(
            self.items_per_thread,
            dtype=self.dtype,
            values=list(self._values),
        )
        result = self._preserve_common_root(result)
        return result

    def __deepcopy__(self, memo: dict[int, Any]) -> ThreadData:
        result = ThreadData(
            self.items_per_thread,
            dtype=_deepcopy(self.dtype, memo),
            values=[
                _UNSET if value is _UNSET else _deepcopy(value, memo)
                for value in self._values
            ],
        )
        result = self._preserve_common_root(result)
        memo[id(self)] = result
        return result

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
            from ._compiler._types import _validate_common_root_numeric_dtype

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
    if _is_ordinary_scalar_dtype(type(value)) or _is_cutlass_dsl_dtype(type(value)):
        return False
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
        from cuda.coop._core.api._dispatch import _common_root_operation_name

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
                        f"ThreadData {arg_name} payload in the common API; use "
                        "cuda.coop.cutlass for backend-qualified scalar or "
                        "register payload support"
                    )
            elif common_root_payload_kind == "scalar_or_thread_data":
                if not isinstance(value, ThreadData) and _is_thread_payload_candidate(
                    value
                ):
                    raise TypeError(
                        f"cuda.coop.{common_operation} accepts only a scalar or "
                        f"fixed-size ThreadData {arg_name} payload in the common API; "
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


def _snapshot_readable_payload(value, *, name, primitive, allow_scalar=False):
    """Read the common payload interface without requiring writable inputs."""

    from cuda.coop._core.api._dispatch import _common_root_operation_name
    from cuda.coop._core.api._payload import (
        _ReadableThreadDataLike,
        _validate_common_numeric_value,
    )

    common = _common_root_operation_name() == primitive
    if common or isinstance(value, _ReadableThreadDataLike):
        _validate_common_numeric_value(
            primitive,
            name,
            value,
            require_thread_data=True,
            allow_readonly_thread_data=True,
        )
        return ThreadData(
            value.items_per_thread,
            dtype=value.dtype,
            values=[value[index] for index in range(value.items_per_thread)],
            alignment=getattr(value, "alignment", None),
        )
    value = _coerce_thread_payload(
        value, scope=_ROOT_SCOPE, primitive_name=primitive, arg_name=name
    )
    if not isinstance(value, ThreadData) and not allow_scalar:
        raise TypeError(
            f"{_ROOT_SCOPE}.{primitive} {name} must be a fixed-size ThreadData"
        )
    return value


def _make_rmem_tensor(shape: Any, dtype: Any, alignment: int | None = None) -> Any:
    """Allocate CuTe register storage honoring a minimum byte alignment."""

    from cutlass import cute

    alignment = _normalize_alignment(alignment)
    if alignment is None or alignment <= 32:
        return cute.make_rmem_tensor(shape, dtype)

    from cutlass._mlir.dialects import cute as cute_ir
    from cutlass.cute.tensor import _Tensor

    layout = cute.make_layout(shape)
    pointer_type = cute_ir.PtrType.get(
        dtype.mlir_type, cute.AddressSpace.rmem, alignment
    )
    tensor_type = cute_ir.MemRefType.get(pointer_type, layout.type)
    allocation = cute_ir.memref_alloca(tensor_type, layout=layout)
    return _Tensor(allocation.value, dtype)
