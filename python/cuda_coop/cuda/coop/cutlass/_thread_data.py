# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Represent a thread's fixed payload as CuTe scalar expressions.

The Python container supports static indexing and in-place item replacement.
Register conversion methods copy values. When a kernel changes a payload in
a runtime ``if`` or loop, CuTe passes its items through the generated compiler
IR region. The control-flow hooks split the payload into one IR scalar per
item and rebuild it from the region results. Every item must be initialized
first because an unset slot has no IR value to pass.

Payloads created through the common ``cuda.coop`` API retain that origin
during reconstruction. This keeps the common API dtype checks active on
later item assignments.

Module helpers decide which arguments are register payloads. Qualified
calls convert CuTe register tensors and vectors to ThreadData; common calls
reject them.

The snapshot helper copies readable payloads into new ThreadData before
lowering, so the operation can leave its inputs unchanged. The copy only
reads items, so these operations also accept read-only inputs. Lowerings
allocate register outputs with the payload's alignment when one is set.
"""

from __future__ import annotations

import inspect as _inspect
import operator as _operator
from collections.abc import Callable, Iterator
from copy import deepcopy as _deepcopy
from typing import Any, Literal, Protocol

from cuda.coop._core.api._payload import _normalize_alignment


class CutlassTensorSample(Protocol):
    """Describe the attributes read from a mutable CuTe tensor.

    Each operation checks whether register or global memory is required.
    """

    @property
    def element_type(self) -> object: ...
    @property
    def shape(self) -> object: ...
    @property
    def memspace(self) -> object: ...
    def __getitem__(self, index: int, /) -> Any: ...
    def load(self) -> object: ...


class CutlassTensorSSASample(Protocol):
    """Describe an immutable register value for payload conversion."""

    @property
    def dtype(self) -> object: ...
    @property
    def shape(self) -> object: ...
    def __getitem__(self, index: int, /) -> Any: ...
    def ir_value(self) -> object: ...


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
    """Return a static integer, or None for booleans and dynamic values."""

    if isinstance(value, bool):
        return None
    try:
        normalized = _operator.index(value)
    except Exception:  # noqa: BLE001
        # Dynamic DSL values may reject integer conversion.
        return None
    if isinstance(normalized, bool):
        return None
    return int(normalized)


def _normalize_group_width(value: Any) -> int | None:
    try:
        normalized = _normalize_index_int(value)
    except Exception:  # noqa: BLE001
        # An uninspectable width is not a static extent.
        return None
    if normalized is not None and normalized > 0:
        return normalized
    return None


def _get_optional_metadata_attr(value: Any, attr_name: str) -> Any:
    """Read foreign metadata when its accessor is available."""

    try:
        return getattr(value, attr_name, None)
    except Exception:  # noqa: BLE001
        # Optional foreign metadata may reject access.
        return None


def _first_optional_metadata_attr(
    value: Any, attr_names: tuple[str, ...]
) -> Any:
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
    """Multiply positive static leaves of a possibly nested shape.

    Any missing, dynamic, or nonpositive leaf makes the total unknown.
    """

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
    """Infer extent from tensor shape, then layout shape, then type shape.

    ``ThreadData.from_register_tensor`` uses this result to size its Python
    item list before reading register values. Here an extent is the number
    of scalar items held by one thread. Different CuTe containers expose
    that count on the value, layout, or type, so the helper checks each.

    Mutable register tensors can expose nested layouts; vector callers can
    require a one-dimensional extent instead.

    Parameters
    ----------
    fragment : object
        Register container whose optional shape metadata is inspected.
    allow_nested : bool
        Multiply all static dimensions of a nested tensor shape when true.
        False accepts only a scalar dimension or a one-dimensional shape.

    Returns
    -------
    int or None
        Positive item count known while tracing, or None when no supported
        shape proves that count. No register values are read or allocated.
    """

    infer_extent = (
        _infer_static_extent if allow_nested else _infer_1d_static_extent
    )

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
        inferred = infer_extent(
            _get_optional_metadata_attr(fragment_type, "shape")
        )
        if inferred is not None:
            return inferred

    return None


def _infer_vector_items_per_thread(vector: Any) -> int | None:
    """Find vector extent through numel() or one-dimensional shape."""

    numel = _get_optional_metadata_attr(vector, "numel")
    if callable(numel):
        try:
            inferred = _normalize_group_width(numel())
        except Exception:  # noqa: BLE001
            # A failing optional numel method cannot prove an extent.
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
    except Exception:  # noqa: BLE001
        # Uninspectable memory space cannot prove register storage.
        return False
    return _is_register_memory_space(memspace)


def _is_register_memory_space(memspace: Any) -> bool:
    """Recognize the runtime's CUTLASS or CuTe register-space enum.

    Optional imports and foreign equality can fail; absent evidence must
    not establish register storage.
    """

    register_spaces = []
    try:
        from cutlass import AddressSpace as CutlassAddressSpace

        register_spaces.append(CutlassAddressSpace.rmem)
    except Exception:  # noqa: BLE001, S110
        # Optional CUTLASS address-space discovery may be unavailable.
        pass
    try:
        from cutlass._mlir.dialects.cute import AddressSpace as CuteAddressSpace

        register_spaces.append(CuteAddressSpace.rmem)
    except Exception:  # noqa: BLE001, S110
        # Optional CuTe address-space discovery may be unavailable.
        pass

    for register_space in register_spaces:
        try:
            if memspace == register_space:
                return True
        except Exception:  # noqa: BLE001, S112
            # Foreign address-space wrappers may reject equality.
            continue
    return False


def _has_memory_space(value: Any) -> bool:
    """Detect memory-space metadata, even when its accessor fails.

    Any declared space, including rmem, marks a tensor rather than an
    immutable register vector.
    """

    for attr_name in ("memspace", "space"):
        try:
            attr = getattr(value, attr_name, None)
        except Exception:  # noqa: BLE001
            # Reject payloads with a declared but unreadable memory space.
            try:
                _inspect.getattr_static(value, attr_name)
            except AttributeError:
                continue
            return True
        if attr is not None:
            return True
    return False


def _has_memory_protocol(value: Any) -> bool:
    """Detect array or DLPack protocols without requesting their data.

    A present or failing protocol identifies memory-backed data, which must
    not be mistaken for per-thread vector contents.
    """

    for attr_name in _MEMORY_PROTOCOL_ATTRS:
        try:
            _inspect.getattr_static(value, attr_name)
        except AttributeError:
            pass
        else:
            return True

        try:
            attr = getattr(value, attr_name, None)
        except Exception:  # noqa: BLE001
            # A failing memory protocol still identifies memory-backed data.
            return True
        if attr is not None:
            return True
    return False


def _is_memory_backed_payload(value: Any) -> bool:
    return _has_memory_space(value) or _has_memory_protocol(value)


def _is_cutlass_dsl_dtype(dtype: Any) -> bool:
    return (
        isinstance(dtype, type)
        and dtype.__module__ == "cutlass.base_dsl.typing"
    )


def _is_ordinary_scalar_dtype(dtype: Any) -> bool:
    """Recognize Python scalar classes and NumPy scalar subclasses."""

    if any(dtype is candidate for candidate in (bool, int, float, complex)):
        return True
    if (
        not isinstance(dtype, type)
        or dtype.__module__.split(".", 1)[0] != "numpy"
    ):
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
    """Cast known scalar dtypes and retain uninitialized slots.

    Leave opaque dtype metadata alone. Wrap conversion failures with the item
    index and conversion source so a caller can locate the bad value.
    """

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
    """Reconcile an explicit positive item count with any inferred extent.

    An explicit count can fill missing metadata but must not contradict
    a known payload size.
    """

    explicit = (
        None if explicit is None else _validate_items_per_thread(explicit)
    )
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
    """Choose a flat export shape or validate a static reshape.

    The new shape must contain exactly the payload's item count; export does
    not pad or truncate values.
    """

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
    """Resolve explicit or retained dtype metadata for a register export.

    Require a supported numeric type rather than inferring a dtype from
    exported items at this stage.
    """

    dtype = fallback if dtype is None else dtype
    if dtype is None:
        raise TypeError(
            f"{source} requires dtype when ThreadData.dtype is not set"
        )
    from ._compiler._types import ALL_PROVIDER_TYPES, canonical_dsl_type

    resolved = canonical_dsl_type(dtype)
    if resolved not in ALL_PROVIDER_TYPES:
        raise TypeError(f"{source} dtype must be a supported numeric type")
    return resolved


class ThreadData:
    """Hold a fixed number of items per thread inside a CuTe kernel.

    This is the CUTLASS implementation of :class:`cuda.coop.ThreadData`.
    Primitives infer the item count from the payload. Indexing uses
    compile-time integers; initialize each item before reading it or passing
    it to an operation that reads the payload.

    The conversion methods below copy values between this payload and CuTe
    registers. Use :func:`cuda.coop.load` and :func:`cuda.coop.store` to move
    data to or from global memory. See :ref:`coop-cutlass-register-payloads`
    for the distinction between mutable register tensors and ``TensorSSA``.

    Parameters
    ----------
    items_per_thread : int
        Positive compile-time item count.
    dtype : type, optional
        Optional numeric element-type metadata. Load supplies the memory
        element type; consuming primitives can infer it from homogeneous
        initialized values. Advanced interop may need explicit metadata
        when a raw integer IR value has no signedness. See
        :ref:`coop-cutlass-dtype-inference`. The constructor does not cast
        entries supplied in ``values``.
    values : tuple or list, optional
        Initial values, with exactly ``items_per_thread`` entries. When
        omitted, the items remain uninitialized until assigned or loaded.
    alignment : int, optional
        Minimum byte alignment for materialized storage. Must be a positive
        power of two. The backend may select a larger alignment.

    Examples
    --------
    Convert a CuTe register tensor to a payload, change one item, and export
    both immutable and mutable register results. The original tensor retains
    its values:

    .. literalinclude::
       ../../python/cuda_coop/tests/backends/cutlass/runtime/test_qualified_payload_examples.py
       :language: python
       :start-after: # example-begin
       :end-before: # example-end
       :dedent: 4
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
        """Construct a payload from one or more per-thread scalar values.

        Parameters
        ----------
        *values : scalar
            Initial items, in payload order. At least one value is required.
        dtype : type, optional
            Dtype metadata. Values are stored as supplied, without casting.

        Returns
        -------
        ThreadData
            A new payload containing ``len(values)`` initialized items.
        """
        if len(values) == 0:
            raise ValueError(
                "ThreadData.from_values requires at least one value"
            )
        return cls(len(values), dtype=dtype, values=list(values))

    @classmethod
    def from_fn(
        cls,
        items_per_thread: int,
        fn: Callable[[int], Any],
        *,
        dtype: Any = None,
    ) -> ThreadData:
        """Initialize each item with a compile-time Python callable.

        Parameters
        ----------
        items_per_thread : int
            Positive compile-time item count.
        fn : callable
            Called once for each integer index from zero to
            ``items_per_thread - 1`` while tracing the kernel. It may build
            CuTe scalar expressions; it is not a device callback.
        dtype : type, optional
            Dtype for the generated values. An explicit CUTLASS numeric type
            casts each item to that type.

        Returns
        -------
        ThreadData
            A new initialized payload in increasing item-index order.
        """
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
        """Copy a CuTe register tensor into a per-thread payload.

        Parameters
        ----------
        fragment : cute.Tensor
            Tensor in CuTe register memory (``rmem``). Global- and
            shared-memory tensors are rejected; load their data first.
        items_per_thread : int, optional
            Positive compile-time item count. Inferred from the static
            tensor shape when available. An explicit count must match the
            inferred count; it is required when inference is unavailable.
        dtype : type, optional
            Defaults to ``fragment.element_type``. An explicit CUTLASS
            numeric type casts each extracted item.

        Returns
        -------
        ThreadData
            A new payload in the tensor's flat integer-index order.
            Subsequent assignments to either object do not update the other.
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
                "ThreadData.from_register_tensor could not infer "
                "items_per_thread from fragment shape; pass "
                "items_per_thread explicitly"
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
        """Copy an immutable register value into a per-thread payload.

        Parameters
        ----------
        vector : cute.TensorSSA or CUTLASS register vector
            Integer-indexable register value. Memory-backed tensors and
            arrays are rejected. For mutable CuTe ``rmem`` tensors, use
            :meth:`from_register_tensor`.
        items_per_thread : int, optional
            Positive compile-time item count, normally inferred from
            ``numel()`` or a static one-dimensional shape. An explicit count
            must match any inferred count.
        dtype : type, optional
            Defaults to the value's dtype metadata. An explicit CUTLASS
            numeric type casts each extracted item.

        Returns
        -------
        ThreadData
            A new payload in flat integer-index order. The input is unchanged.
        """
        if _is_memory_backed_payload(vector):
            raise TypeError(
                "ThreadData.from_vector requires a CUTLASS vector-like "
                "per-thread payload; use ThreadData.from_register_tensor for "
                "CuTe register fragments, or a group-first load "
                "for memory tensors"
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

        from cutlass import cute

        if isinstance(vector, cute.TensorSSA):
            # Indexing may cache a layout operation in the current IR region.
            # A fresh view prevents this conversion from leaving region-local
            # metadata on a caller's value that also lives outside the region.
            vector = vector.reshape(vector.shape)

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
        """Adapt a ``ThreadData`` or CuTe register payload.

        Parameters
        ----------
        payload : ThreadData, cute.Tensor, cute.TensorSSA, or register vector
            Per-thread values. CuTe tensors must use register memory;
            global/shared tensors and arrays require a load first.
        items_per_thread : int, optional
            Positive compile-time item count. If supplied, it must match the
            existing or inferred count.
        dtype : type, optional
            Dtype for conversion. For an existing ``ThreadData`` with dtype
            metadata, this must match that dtype; this method does not recast
            an already typed payload.

        Returns
        -------
        ThreadData
            The same object when ``payload`` is already ``ThreadData`` and
            no dtype change is needed. Supplying a dtype to an untyped
            ``ThreadData`` creates a new payload. CuTe inputs produce a new
            payload through :meth:`from_register_tensor` or :meth:`from_vector`.
        """
        if isinstance(payload, cls):
            if items_per_thread is not None:
                items_per_thread = _validate_items_per_thread(items_per_thread)
                if payload.items_per_thread != items_per_thread:
                    raise ValueError(
                        "ThreadData.from_payload items_per_thread does not "
                        "match payload.items_per_thread"
                    )
            if dtype is None or payload.dtype == dtype:
                return payload
            if payload.dtype is not None:
                raise TypeError(
                    "ThreadData.from_payload dtype does not match payload"
                )
            if payload._common_root:
                from ._compiler._types import (
                    _validate_common_root_numeric_dtype,
                )

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
                "ThreadData.from_payload requires a per-thread "
                "register payload; use ThreadData.from_register_tensor "
                "for CuTe register fragments, "
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
        """Export initialized items as an immutable CuTe register value.

        Parameters
        ----------
        dtype : type, optional
            Supported CUTLASS or NumPy numeric dtype. Defaults to this
            payload's dtype; required if the payload has no dtype metadata.
        shape : int or tuple, optional
            Defaults to ``(items_per_thread,)``. May be nested, but every
            extent must be a positive compile-time integer and the total
            number of elements must equal ``items_per_thread``.

        Returns
        -------
        cute.TensorSSA
            Register values with the requested dtype and shape, in payload
            item order. The original input fragment's shape is not retained
            automatically. Later payload assignments do not change this value.
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
        """Export initialized items to a new mutable CuTe register tensor.

        Parameters
        ----------
        dtype : type, optional
            Output dtype, with the same rules as :meth:`to_tensor_ssa`.
        shape : int or tuple, optional
            Static output shape, with the same rules as :meth:`to_tensor_ssa`.

        Returns
        -------
        cute.Tensor
            A new ``rmem`` tensor containing the payload values. It does not
            alias the payload or an earlier input fragment. Its storage honors
            this payload's minimum alignment. As with other register tensors,
            the compiler may spill values to local memory.
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
        """Check static indices, including Python-style negative indices."""

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
        """Resolve payload items to typed scalars for CuTe IR operations.

        CuTe uses ``__extract_mlir_values__`` and ``__new_from_mlir_values__``
        when a payload crosses a function boundary or runtime branch or loop.
        Those hooks call this helper because IR uses separate typed scalar
        operands, rather than a Python ThreadData object.

        Convert host literals to scalar expressions. A declared dtype supplies
        the signedness for raw integer IR values of matching width.

        Returns
        -------
        tuple
            Common CUTLASS scalar type and the initialized item expressions in
            payload order. Uninitialized or incompatible items raise an error.
        """

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
                value_type(value)
                if plain is _types._NOT_PLAIN_SCALAR
                else plain
            )
        return value_type, tuple(converted)

    def __extract_mlir_values__(self) -> list[Any]:
        """Expose one scalar IR value per payload item to CuTe.

        CuTe calls this hook to flatten a Python payload into scalar IR
        operands, including for function calls and runtime branches or loops.
        The returned order defines the correspondence used by
        ``__new_from_mlir_values__`` to rebuild the payload.

        Returns
        -------
        list
            One typed MLIR scalar per initialized payload item, in index
            order.
        """
        _, values = self._dynamic_values()
        return [value.ir_value() for value in values]

    def __new_from_mlir_values__(self, values: list[Any]) -> ThreadData:
        """Rebuild a payload from replacement scalar IR values.

        CuTe calls this hook when it needs a Python payload for replacement IR
        values, such as function block arguments or runtime control-flow
        results. A block argument names a value supplied on entry to an IR
        block. The hook wraps the supplied scalars in a new Python container.

        Require one value per item. Keep the alignment and the common API
        origin, so later assignments still apply common dtype checks. Keep a
        declared dtype; without one, record the item type inferred from
        the original payload.

        Parameters
        ----------
        values : list
            Replacement scalar IR values in the item order established by
            ``__extract_mlir_values__``; their count must match
            items_per_thread.

        Returns
        -------
        ThreadData
            Reconstructed payload with the same item count and alignment.
        """
        if len(values) != self.items_per_thread:
            raise ValueError(
                "ThreadData control flow requires one value per item"
            )
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
        """Preserve common API dtype checks and alignment in a new payload."""

        result._common_root = self._common_root
        result.alignment = self.alignment
        return result

    def __copy__(self) -> ThreadData:
        """Copy the container while sharing scalar values and metadata."""

        result = ThreadData(
            self.items_per_thread,
            dtype=self.dtype,
            values=list(self._values),
        )
        result = self._preserve_common_root(result)
        return result

    def __deepcopy__(self, memo: dict[int, Any]) -> ThreadData:
        """Copy dtype and items while retaining the uninitialized sentinel."""

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
        """Reject incomplete payloads and report all missing item indices."""

        missing = [
            idx for idx, value in enumerate(self._values) if value is _UNSET
        ]
        if missing:
            context = (
                "ThreadData iteration"
                if primitive_name is None
                else f"{_ROOT_SCOPE}.{primitive_name}"
            )
            raise ValueError(
                f"{context} requires ThreadData values to be initialized "
                "before use; missing index(es): "
                + ", ".join(str(i) for i in missing)
            )
        return self._values

    def values(self, primitive_name: str) -> tuple[Any, ...]:
        """Return initialized items in payload order for a primitive."""

        return tuple(self._require_values(primitive_name))

    def __iter__(self) -> Iterator[Any]:
        return iter(self._require_values(None))


def _is_thread_payload_candidate(value: Any) -> bool:
    """Recognize register containers even when their extent is unknown.

    A malformed vector must reach conversion diagnostics instead of falling
    through as a scalar. Memory-backed values are candidates too, so
    conversion can explain why an explicit load is needed.
    """

    if _is_ordinary_scalar_dtype(type(value)) or _is_cutlass_dsl_dtype(
        type(value)
    ):
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
    """Adapt register payloads within the caller's common API contract.

    When invoked for a delegated common operation, check its permitted payload
    kind before considering automatic conversion. Existing ThreadData and
    scalar values pass through; eligible register containers use from_payload.
    """

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
                if not isinstance(
                    value, ThreadData
                ) and _is_thread_payload_candidate(value):
                    raise TypeError(
                        f"cuda.coop.{common_operation} accepts only "
                        "a scalar or fixed-size ThreadData "
                        f"{arg_name} payload in the common API; "
                        "use cuda.coop.cutlass for backend-qualified register "
                        "payload support"
                    )
            # The annotation defines the private contract.
            else:  # pragma: no cover
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
    """Copy readable items before lowering to preserve the caller's payload.

    Common calls, and any input with the readable ThreadData interface,
    pass the shared payload checks. Copy their items, dtype, extent, and
    optional alignment. Qualified register containers use the usual adapter.
    With ``allow_scalar=True``, other qualified inputs pass through for the
    operation to validate as scalars. Common calls still require a fixed-size
    payload. Read-only inputs work because the copy only reads items and the
    operation writes fresh results.
    """

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


def _make_rmem_tensor(
    shape: Any, dtype: Any, alignment: int | None = None
) -> Any:
    """Emit register-storage allocation with a minimum byte alignment.

    The standard CuTe allocator already aligns register storage to 32 bytes,
    so use it for requests up to 32. Larger requests build an aligned pointer
    type and allocate the memref directly. The compiler may still spill
    register values to local memory.
    """

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


__all__ = [
    "ThreadData",
    "_coerce_thread_payload",
    "_make_rmem_tensor",
    "_snapshot_readable_payload",
]
