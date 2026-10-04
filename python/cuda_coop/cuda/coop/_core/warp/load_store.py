# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Describe CUB WarpLoad and WarpStore wrappers for physical or logical warps.

Builders validate each warp's tile and describe the selected call signatures.
The resulting specialization includes the logical width and pointer-offset
metadata. Group planning and backend lowering use that metadata to give each
warp its own consecutive tile within the block's input or output.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .._algorithm import Algorithm
from .._bindings import (
    ArgumentBinding,
    BindingKind,
    cxx_scalar_literal,
    i32_parameter,
    normalize_i32_binding,
    normalize_i64_binding,
)
from .._symbols import semantic_token
from .._types import (
    INT64,
    Array,
    CxxFunction,
    Dependency,
    Pointer,
    PointerOffset,
    TemplateParameter,
    TempStorageParameter,
    Value,
)

_MAX_WARP_THREADS = 32
_SUPPORTED_LOGICAL_WARP_THREADS = frozenset({1, 2, 4, 8, 16, 32})


class WarpLoadStoreKind(str, Enum):
    """Direction of data movement between memory and per-thread items."""

    LOAD = "load"
    STORE = "store"


class WarpLoadStoreAlgorithm(str, Enum):
    """CUB Warp Load/Store algorithms and their per-thread arrangements.

    ``DIRECT``, ``VECTORIZE``, and ``TRANSPOSE`` use a blocked arrangement:
    lane ``r`` owns tile indices ``r * items_per_thread + i``. ``STRIPED``
    uses indices ``r + i * threads_in_warp`` instead. In both expressions,
    ``i`` indexes the thread's items and ``r`` is its logical-warp lane.

    ``DIRECT`` accesses that blocked arrangement directly. ``VECTORIZE``
    requests vectorized access where CUB's type, size, and alignment
    requirements permit it; guarded calls use direct access. ``TRANSPOSE``
    exchanges items within the logical warp so memory accesses are striped
    while the per-thread arrangement remains blocked. Stores reverse the
    corresponding load's data movement.
    """

    DIRECT = "direct"
    STRIPED = "striped"
    VECTORIZE = "vectorize"
    TRANSPOSE = "transpose"


WarpLoadAlgorithm = WarpLoadStoreAlgorithm
WarpStoreAlgorithm = WarpLoadStoreAlgorithm


_LOAD_ALGORITHM_CPP = {
    WarpLoadAlgorithm.DIRECT: "::cub::WARP_LOAD_DIRECT",
    WarpLoadAlgorithm.STRIPED: "::cub::WARP_LOAD_STRIPED",
    WarpLoadAlgorithm.VECTORIZE: "::cub::WARP_LOAD_VECTORIZE",
    WarpLoadAlgorithm.TRANSPOSE: "::cub::WARP_LOAD_TRANSPOSE",
}
_STORE_ALGORITHM_CPP = {
    WarpStoreAlgorithm.DIRECT: "::cub::WARP_STORE_DIRECT",
    WarpStoreAlgorithm.STRIPED: "::cub::WARP_STORE_STRIPED",
    WarpStoreAlgorithm.VECTORIZE: "::cub::WARP_STORE_VECTORIZE",
    WarpStoreAlgorithm.TRANSPOSE: "::cub::WARP_STORE_TRANSPOSE",
}
_T = Dependency("T")
_ITEMS_PER_THREAD = Dependency("ITEMS_PER_THREAD")
_TEMPLATE_PARAMETERS = (
    TemplateParameter("T"),
    TemplateParameter("ITEMS_PER_THREAD"),
    TemplateParameter("ALGORITHM"),
    TemplateParameter("LOGICAL_WARP_THREADS"),
)


def _algorithm_cpp_map(
    kind: WarpLoadStoreKind,
) -> dict[WarpLoadStoreAlgorithm, str]:
    return (
        _LOAD_ALGORITHM_CPP
        if kind is WarpLoadStoreKind.LOAD
        else _STORE_ALGORITHM_CPP
    )


def _normalize_algorithm(
    kind: WarpLoadStoreKind,
    algorithm: str | WarpLoadStoreAlgorithm,
) -> WarpLoadStoreAlgorithm:
    """Accept an enum, a short name, or the matching CUB operation's token."""

    mapping = _algorithm_cpp_map(kind)
    if isinstance(algorithm, WarpLoadStoreAlgorithm):
        return algorithm
    if isinstance(algorithm, str):
        for candidate, cpp in mapping.items():
            if algorithm in {candidate.value, cpp}:
                return candidate
    raise ValueError(
        f"unsupported Warp{kind.value.title()} algorithm {algorithm!r}"
    )


def _normalize_items_per_thread(items_per_thread: Any) -> int:
    if (
        not isinstance(items_per_thread, int)
        or isinstance(items_per_thread, bool)
        or items_per_thread < 1
    ):
        raise ValueError("items_per_thread must be a positive integer")
    return int(items_per_thread)


def _normalize_logical_warp_threads(threads_in_warp: Any) -> int:
    """Require a power-of-two width that fits within one physical warp."""

    if (
        not isinstance(threads_in_warp, int)
        or isinstance(threads_in_warp, bool)
        or threads_in_warp not in _SUPPORTED_LOGICAL_WARP_THREADS
    ):
        supported = ", ".join(
            str(value) for value in sorted(_SUPPORTED_LOGICAL_WARP_THREADS)
        )
        raise ValueError(
            "Warp Load/Store requires threads_in_warp in "
            f"{{{supported}}}; got {threads_in_warp!r}"
        )
    return threads_in_warp


def _base_parameters(kind: WarpLoadStoreKind) -> list[Any]:
    """Describe scratch, the memory pointer, and the per-thread item array.

    Load and store take the pointer first but reverse the input/output roles.
    Both write through arguments instead of returning an array or pointer.
    """

    if kind is WarpLoadStoreKind.LOAD:
        return [
            TempStorageParameter(),
            Pointer(_T, name="src", is_array_pointer=True, restrict=True),
            Array(
                _T,
                _ITEMS_PER_THREAD,
                name="dst",
                is_output=True,
                is_return=False,
            ),
        ]
    return [
        TempStorageParameter(),
        Pointer(
            _T,
            name="dst",
            is_output=True,
            is_return=False,
            is_array_pointer=True,
            restrict=True,
        ),
        Array(_T, _ITEMS_PER_THREAD, name="src"),
    ]


def _normalize_optional_binding(
    value: bool | ArgumentBinding,
    *,
    name: str,
) -> ArgumentBinding:
    """Use booleans to select signatures and keep explicit bindings."""

    if isinstance(value, ArgumentBinding):
        return value
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool or ArgumentBinding")
    return ArgumentBinding.runtime() if value else ArgumentBinding.omitted()


def _with_pointer_offset(
    parameters: list[Any],
    offset: ArgumentBinding,
) -> tuple[Any, ...]:
    """Append an element offset for the memory pointer after scratch storage.

    The descriptor targets argument zero after the leading storage parameter
    is removed: ``src`` for loads and ``dst`` for stores. Lowering applies
    the offset to that pointer before invoking CUB.
    """

    static_value = offset.value if offset.kind is BindingKind.STATIC else None
    return (
        *parameters,
        PointerOffset(
            INT64,
            name="offset",
            pointer_arg_index=0,
            static_value=static_value,
        ),
    )


@dataclass(frozen=True)
class WarpLoadStoreSemantics:
    """Normalized call variants for one physical or logical warp's tile.

    Construct this description with :func:`make_warp_load_store_semantics`
    to validate options and build the parameter descriptors consumed by a
    backend. The tile contains ``threads_in_warp * items_per_thread`` items.

    Attributes
    ----------
    kind : WarpLoadStoreKind
        Load from memory into per-thread items, or store those items to
        memory.
    dtype : object
        Element type used by the CUB primitive and its per-thread items.
    algorithm : WarpLoadStoreAlgorithm
        Memory-access strategy and per-thread arrangement.
    items_per_thread : int
        Number of items owned by each participating thread.
    threads_in_warp : int
        Logical warp width: one of 1, 2, 4, 8, 16, or 32. A width of 32
        represents a physical warp.
    valid_items : ArgumentBinding
        Omitted, static, or runtime valid-item count for a guarded call.
    oob_default : ArgumentBinding
        Omitted, static, or runtime value for invalid load items.
    has_full_tile : bool
        Whether the described variants include an unguarded full-tile call.
        This records signature availability, not a runtime bounds check.
    pointer_offset : ArgumentBinding
        Omitted, static, or runtime element offset applied to the memory
        pointer. A runtime binding denotes an effective offset supplied by
        the backend; group planning accounts for the warp's tile origin.
    parameters : tuple of tuples
        Parameter descriptors for each selected call variant. Every variant
        starts with temporary storage, the memory pointer, and the
        per-thread array, followed by any bounds, default, and offset
        descriptors. Static descriptors are embedded during lowering rather
        than passed as runtime arguments.
    """

    kind: WarpLoadStoreKind
    dtype: Any
    algorithm: WarpLoadStoreAlgorithm
    items_per_thread: int
    threads_in_warp: int
    valid_items: ArgumentBinding
    oob_default: ArgumentBinding
    has_full_tile: bool
    pointer_offset: ArgumentBinding
    parameters: tuple[tuple[Any, ...], ...]

    @property
    def has_valid_items(self) -> bool:
        return self.valid_items.kind is not BindingKind.OMITTED

    @property
    def has_oob_default(self) -> bool:
        return self.oob_default.kind is not BindingKind.OMITTED

    @property
    def has_pointer_offset(self) -> bool:
        return self.pointer_offset.kind is not BindingKind.OMITTED

    @property
    def requires_runtime_effective_offset(self) -> bool:
        """Whether lowering must supply the effective element offset."""

        return self.pointer_offset.kind is BindingKind.RUNTIME

    @property
    def method_name(self) -> str:
        return self.kind.value.title()

    @property
    def algorithm_cpp(self) -> str:
        return _algorithm_cpp_map(self.kind)[self.algorithm]

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        """Identify the bindings and selected call variants."""

        return (
            f"warp_{self.kind.value}",
            semantic_token(self.dtype),
            self.algorithm.value,
            self.items_per_thread,
            self.threads_in_warp,
            self.valid_items.semantic_key,
            self.oob_default.semantic_key,
            self.has_full_tile,
            self.pointer_offset.semantic_key,
            semantic_token(self.parameters),
        )


def make_warp_load_store_semantics(
    *,
    kind: str | WarpLoadStoreKind,
    dtype: Any,
    items_per_thread: int,
    algorithm: str | WarpLoadStoreAlgorithm,
    threads_in_warp: int = _MAX_WARP_THREADS,
    valid_items: bool | ArgumentBinding = False,
    oob_default: bool | ArgumentBinding = False,
    include_full_tile: bool = False,
    include_pointer_offset: bool | ArgumentBinding = False,
) -> WarpLoadStoreSemantics:
    """Describe CUB Warp Load/Store call variants and their scalar bindings.

    Normalize the algorithm and optional arguments, validate static bounds,
    and build descriptors for the selected full-tile and guarded calls.
    Each invocation moves one logical warp's tile; the caller must provide
    complete participation by that warp. This builds metadata for later
    specialization and lowering.

    Parameters
    ----------
    kind : str or WarpLoadStoreKind
        ``"load"`` or ``"store"``.
    dtype : object
        Element type carried into the CUB specialization and used for any
        out-of-bounds default. Type interpretation belongs to the backend.
    items_per_thread : int
        Positive number of items per thread. Booleans are rejected.
    algorithm : str or WarpLoadStoreAlgorithm
        Algorithm enum, its lowercase value, or the matching fully
        qualified CUB enumerator, such as ``"::cub::WARP_LOAD_TRANSPOSE"``.
        See :class:`WarpLoadStoreAlgorithm` for the per-thread arrangements.
    threads_in_warp : int, optional
        Logical warp width in ``{1, 2, 4, 8, 16, 32}``, defaulting to the
        physical warp width of 32. Booleans are rejected.
    valid_items : bool or ArgumentBinding, optional
        ``False`` omits the guarded signature; ``True`` selects a runtime
        signed 32-bit count. An explicit binding can omit the count, embed
        a static count, or request a runtime count. The count covers the
        first valid items of this warp's tile, not items per thread.
        Static counts must fit signed 32-bit arithmetic and lie between
        zero and ``threads_in_warp * items_per_thread``, inclusive.
        Runtime counts must satisfy the same bounds at the call site.
    oob_default : bool or ArgumentBinding, optional
        Value assigned to invalid items by a guarded load. ``False`` omits
        the default; ``True`` adds a runtime value of ``dtype``. Explicit
        bindings select omitted, static, or runtime values. Static defaults
        must be finite numeric scalars representable as C++ literals.
        Requires ``valid_items`` and is invalid for stores. With no default,
        invalid load items are unspecified; their prior values need not be
        preserved. Guarded stores leave memory outside the valid range
        untouched.
    include_full_tile : bool, optional
        Add an unguarded full-tile variant alongside a guarded variant.
        Setting this requires ``valid_items``. When ``valid_items`` is
        omitted, the full-tile variant is already selected by default.
    include_pointer_offset : bool or ArgumentBinding, optional
        Select element offsets into the memory pointer. ``False`` produces
        only variants without an offset; ``True`` produces both those
        variants and variants with a runtime signed 64-bit offset. An
        explicit static or runtime binding instead produces only variants
        with that offset. An omitted binding produces only variants without
        it. Static offsets must be nonnegative signed 64-bit integers.
        Runtime offsets use signed 64-bit arithmetic; the caller must
        establish pointer validity and bounds.
        Applying an offset does not change the meaning of ``valid_items``:
        the count starts at the adjusted pointer.

    Returns
    -------
    WarpLoadStoreSemantics
        Normalized bindings and parameter descriptors for each requested
        call variant. The descriptors retain template dependencies until
        :func:`make_warp_load_store_specialization` binds them.

    Raises
    ------
    TypeError
        An optional binding is neither a bool nor an ``ArgumentBinding``;
        a static count or offset is not an integer or is a boolean; or a
        static default is not a supported numeric scalar.
    ValueError
        The kind, algorithm, item count, or warp width is unsupported; a
        static count or offset is outside its supported range; a static
        default is nonfinite or its integer literal exceeds 64-bit range;
        or the requested guarded, default, and full-tile variants conflict.

    Notes
    -----
    The offset describes an adjustment to the supplied pointer. This
    builder does not calculate the logical warp's index or add its tile
    origin. Group planning requests a runtime effective offset, and the
    backend combines the group-instance tile origin with the user's offset.
    That planning path records nonnegative, overflow-safe runtime offset
    bounds as caller preconditions.
    """

    pointer_offset_overload_cohort = isinstance(include_pointer_offset, bool)
    kind = WarpLoadStoreKind(kind)
    items_per_thread = _normalize_items_per_thread(items_per_thread)
    threads_in_warp = _normalize_logical_warp_threads(threads_in_warp)
    algorithm = _normalize_algorithm(kind, algorithm)
    valid_items = _normalize_optional_binding(valid_items, name="valid_items")
    valid_items = normalize_i32_binding(valid_items, name="valid_items")
    if valid_items.kind is BindingKind.STATIC:
        tile_items = items_per_thread * threads_in_warp
        if not 0 <= int(valid_items.value) <= tile_items:
            raise ValueError(
                "static valid_items must be between zero and the warp tile "
                f"size ({tile_items})"
            )
    oob_default = _normalize_optional_binding(oob_default, name="oob_default")
    pointer_offset = _normalize_optional_binding(
        include_pointer_offset,
        name="include_pointer_offset",
    )
    pointer_offset = normalize_i64_binding(
        pointer_offset, name="pointer offset"
    )
    if (
        pointer_offset.kind is BindingKind.STATIC
        and int(pointer_offset.value) < 0
    ):
        raise ValueError("static pointer offset must be nonnegative")
    if kind is WarpLoadStoreKind.STORE and (
        oob_default.kind is not BindingKind.OMITTED
    ):
        raise ValueError("oob_default is only valid for WarpLoad")
    if (
        oob_default.kind is not BindingKind.OMITTED
        and valid_items.kind is BindingKind.OMITTED
    ):
        raise ValueError("oob_default requires a valid_items signature")
    if include_full_tile and valid_items.kind is BindingKind.OMITTED:
        raise ValueError("include_full_tile requires a valid_items signature")

    base = _base_parameters(kind)
    has_full_tile = valid_items.kind is BindingKind.OMITTED or include_full_tile
    methods: list[tuple[Any, ...]] = []
    if has_full_tile and (
        pointer_offset.kind is BindingKind.OMITTED
        or pointer_offset_overload_cohort
    ):
        methods.append(tuple(base))
    if valid_items.kind is not BindingKind.OMITTED:
        num_valid_items = i32_parameter(valid_items, name="num_valid_items")
        assert num_valid_items is not None
        partial = [*base, num_valid_items]
        if oob_default.kind is BindingKind.RUNTIME:
            partial.append(Value(dtype, name="oob_default"))
        elif oob_default.kind is BindingKind.STATIC:
            partial.append(
                CxxFunction(
                    cxx_scalar_literal(oob_default.value, name="oob_default"),
                    dtype,
                    name="oob_default",
                )
            )
        if (
            pointer_offset.kind is BindingKind.OMITTED
            or pointer_offset_overload_cohort
        ):
            methods.append(tuple(partial))
        if pointer_offset.kind is not BindingKind.OMITTED:
            methods.append(_with_pointer_offset(partial, pointer_offset))
    if pointer_offset.kind is not BindingKind.OMITTED and has_full_tile:
        methods.append(_with_pointer_offset(base, pointer_offset))

    return WarpLoadStoreSemantics(
        kind=kind,
        dtype=dtype,
        algorithm=algorithm,
        items_per_thread=items_per_thread,
        threads_in_warp=threads_in_warp,
        valid_items=valid_items,
        oob_default=oob_default,
        has_full_tile=has_full_tile,
        pointer_offset=pointer_offset,
        parameters=tuple(methods),
    )


def make_warp_load_store_specialization(
    *,
    kind: str | WarpLoadStoreKind,
    dtype: Any,
    items_per_thread: int,
    algorithm: str | WarpLoadStoreAlgorithm,
    threads_in_warp: int = _MAX_WARP_THREADS,
    valid_items: bool | ArgumentBinding = False,
    oob_default: bool | ArgumentBinding = False,
    include_full_tile: bool = False,
    include_pointer_offset: bool | ArgumentBinding = False,
) -> Algorithm:
    """Bind CUB template arguments for physical or logical Warp Load/Store.

    Normalize and validate the options with
    :func:`make_warp_load_store_semantics`, then construct an ``Algorithm``
    binding ``T``, ``ITEMS_PER_THREAD``, ``ALGORITHM``, and
    ``LOGICAL_WARP_THREADS``. The parameters and validation rules are those
    of that semantics builder.

    Parameters
    ----------
    kind : str or WarpLoadStoreKind
        ``"load"`` or ``"store"``.
    dtype : object
        Element type bound to CUB's ``T``.
    items_per_thread : int
        Positive number of elements per thread, excluding booleans.
    algorithm : str or WarpLoadStoreAlgorithm
        CUB strategy, accepted in the forms documented by
        :func:`make_warp_load_store_semantics`.
    threads_in_warp : int, optional
        Logical warp width in ``{1, 2, 4, 8, 16, 32}``, defaulting to 32.
    valid_items : bool or ArgumentBinding, optional
        Guarded-call count binding, bounded by this warp's tile size.
    oob_default : bool or ArgumentBinding, optional
        Optional invalid-item default for guarded loads.
    include_full_tile : bool, optional
        Include an unguarded overload alongside guarded overloads.
    include_pointer_offset : bool or ArgumentBinding, optional
        Select memory-pointer offset overloads. Boolean and explicit-binding
        selection follow :func:`make_warp_load_store_semantics`.

    Returns
    -------
    Algorithm
        Bound CUB template arguments, header and method, selected parameter
        variants, and effective-offset metadata for group lowering.
        Its tile stride is ``threads_in_warp * items_per_thread`` elements.

    Notes
    -----
    Specialization constructs a description ready for backend materialization.
    It does not compile wrappers, allocate temporary storage, or calculate a
    runtime tile origin. When group planning supplies a runtime effective
    offset, lowering adds the group-instance tile origin to the user's
    offset before invoking the specialized wrapper.
    """

    call = make_warp_load_store_semantics(
        kind=kind,
        dtype=dtype,
        items_per_thread=items_per_thread,
        algorithm=algorithm,
        threads_in_warp=threads_in_warp,
        valid_items=valid_items,
        oob_default=oob_default,
        include_full_tile=include_full_tile,
        include_pointer_offset=include_pointer_offset,
    )
    title = call.kind.value.title()
    return Algorithm(
        struct_name=f"Warp{title}",
        method_name=title,
        c_name=f"warp_{call.kind.value}",
        includes=(f"cub/warp/warp_{call.kind.value}.cuh",),
        template_parameters=_TEMPLATE_PARAMETERS,
        parameters=call.parameters,
        template_arguments={
            "T": dtype,
            "ITEMS_PER_THREAD": call.items_per_thread,
            "ALGORITHM": call.algorithm_cpp,
            "LOGICAL_WARP_THREADS": call.threads_in_warp,
        },
        metadata={
            "scope": "warp",
            "primitive": call.kind.value,
            "algorithm": call.algorithm.value,
            "valid_items": call.has_valid_items,
            "oob_default": call.has_oob_default,
            "full_tile": call.has_full_tile,
            "pointer_offset": call.has_pointer_offset,
            "requires_runtime_effective_offset": (
                call.requires_runtime_effective_offset
            ),
            "effective_offset_origin": "group_instance",
            "effective_offset_stride": (
                call.threads_in_warp * call.items_per_thread
            ),
        },
    )


def make_warp_load_specialization(**kwargs: Any) -> Algorithm:
    """Build a WarpLoad specialization with the shared builder's options.

    See :func:`make_warp_load_store_specialization`; ``kind`` is fixed to
    ``WarpLoadStoreKind.LOAD``.
    """

    return make_warp_load_store_specialization(
        kind=WarpLoadStoreKind.LOAD, **kwargs
    )


def make_warp_store_specialization(**kwargs: Any) -> Algorithm:
    """Build a WarpStore specialization with the shared builder's options.

    See :func:`make_warp_load_store_specialization`; ``kind`` is fixed to
    ``WarpLoadStoreKind.STORE`` and ``oob_default`` must be omitted.
    """

    return make_warp_load_store_specialization(
        kind=WarpLoadStoreKind.STORE, **kwargs
    )


__all__ = [
    "WarpLoadAlgorithm",
    "WarpLoadStoreAlgorithm",
    "WarpLoadStoreKind",
    "WarpLoadStoreSemantics",
    "WarpStoreAlgorithm",
    "make_warp_load_specialization",
    "make_warp_load_store_semantics",
    "make_warp_load_store_specialization",
    "make_warp_store_specialization",
]
