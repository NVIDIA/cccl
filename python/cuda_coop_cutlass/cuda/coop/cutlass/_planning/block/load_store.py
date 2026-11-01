# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Describe CUB BlockLoad and BlockStore wrappers for compiler adapters.

The semantics builder selects full-tile, guarded, and pointer-offset
signatures. The specialization builder then binds the block shape and CUB
template arguments. These descriptions let each backend generate wrappers for
the same operation without importing compiler types into the shared model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral
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
from ._common import normalize_block_dim, normalize_positive_int


class BlockLoadStoreKind(str, Enum):
    """Select CUB's ``BlockLoad.Load`` or ``BlockStore.Store`` operation."""

    LOAD = "load"
    STORE = "store"


class BlockLoadStoreAlgorithm(str, Enum):
    """CUB memory-access strategies and per-thread item arrangements.

    ``DIRECT`` and ``VECTORIZE`` use blocked items: thread ``t`` owns tile
    positions ``t * items_per_thread + i``. ``STRIPED`` uses positions
    ``t + i * block_threads`` instead. ``VECTORIZE`` requests vectorized
    accesses where CUB's type, alignment, and item-count requirements allow.

    The transpose variants also present blocked items to the caller, using
    shared memory to reorder coalesced accesses. ``TRANSPOSE`` exchanges a
    block-striped arrangement; ``WARP_TRANSPOSE`` exchanges a warp-striped
    arrangement. ``WARP_TRANSPOSE_TIMESLICED`` reuses one warp's exchange
    storage across warps to reduce shared memory use. Both warp-transpose
    variants require the block's thread count to be a multiple of 32.

    ``BlockLoadAlgorithm`` and ``BlockStoreAlgorithm`` alias this enum; the
    operation kind selects the corresponding CUB load or store constant.
    """

    DIRECT = "direct"
    STRIPED = "striped"
    VECTORIZE = "vectorize"
    TRANSPOSE = "transpose"
    WARP_TRANSPOSE = "warp_transpose"
    WARP_TRANSPOSE_TIMESLICED = "warp_transpose_timesliced"


BlockLoadAlgorithm = BlockLoadStoreAlgorithm
BlockStoreAlgorithm = BlockLoadStoreAlgorithm


_LOAD_ALGORITHM_CPP = {
    BlockLoadAlgorithm.DIRECT: "::cub::BLOCK_LOAD_DIRECT",
    BlockLoadAlgorithm.STRIPED: "::cub::BLOCK_LOAD_STRIPED",
    BlockLoadAlgorithm.VECTORIZE: "::cub::BLOCK_LOAD_VECTORIZE",
    BlockLoadAlgorithm.TRANSPOSE: "::cub::BLOCK_LOAD_TRANSPOSE",
    BlockLoadAlgorithm.WARP_TRANSPOSE: "::cub::BLOCK_LOAD_WARP_TRANSPOSE",
    BlockLoadAlgorithm.WARP_TRANSPOSE_TIMESLICED: (
        "::cub::BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED"
    ),
}
_STORE_ALGORITHM_CPP = {
    BlockStoreAlgorithm.DIRECT: "::cub::BLOCK_STORE_DIRECT",
    BlockStoreAlgorithm.STRIPED: "::cub::BLOCK_STORE_STRIPED",
    BlockStoreAlgorithm.VECTORIZE: "::cub::BLOCK_STORE_VECTORIZE",
    BlockStoreAlgorithm.TRANSPOSE: "::cub::BLOCK_STORE_TRANSPOSE",
    BlockStoreAlgorithm.WARP_TRANSPOSE: "::cub::BLOCK_STORE_WARP_TRANSPOSE",
    BlockStoreAlgorithm.WARP_TRANSPOSE_TIMESLICED: (
        "::cub::BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED"
    ),
}
_T = Dependency("T")
_ITEMS_PER_THREAD = Dependency("ITEMS_PER_THREAD")
_TEMPLATE_PARAMETERS = (
    TemplateParameter("T"),
    TemplateParameter("BLOCK_DIM_X"),
    TemplateParameter("ITEMS_PER_THREAD"),
    TemplateParameter("ALGORITHM"),
    TemplateParameter("BLOCK_DIM_Y"),
    TemplateParameter("BLOCK_DIM_Z"),
)


def _algorithm_cpp_map(
    kind: BlockLoadStoreKind,
) -> dict[BlockLoadStoreAlgorithm, str]:
    return (
        _LOAD_ALGORITHM_CPP
        if kind is BlockLoadStoreKind.LOAD
        else _STORE_ALGORITHM_CPP
    )


def _normalize_algorithm(
    kind: BlockLoadStoreKind,
    algorithm: str | BlockLoadStoreAlgorithm,
) -> BlockLoadStoreAlgorithm:
    """Accept an enum, a short name, or the matching CUB operation's token."""

    mapping = _algorithm_cpp_map(kind)
    if isinstance(algorithm, BlockLoadStoreAlgorithm):
        return algorithm
    if isinstance(algorithm, str):
        for candidate, cpp in mapping.items():
            if algorithm in {candidate.value, cpp}:
                return candidate
    raise ValueError(
        f"unsupported Block{kind.value.title()} algorithm {algorithm!r}"
    )


def _base_parameters(kind: BlockLoadStoreKind) -> list[Any]:
    """Describe scratch, the global pointer, and the per-thread item array.

    CUB takes the global pointer first for both operations: ``src`` for a
    load and ``dst`` for a store. Outputs are written through arguments;
    neither operation returns the item array or pointer.
    """

    if kind is BlockLoadStoreKind.LOAD:
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
    """Use booleans to select signatures and keep explicit value bindings."""

    if isinstance(value, ArgumentBinding):
        return value
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool or ArgumentBinding")
    return ArgumentBinding.runtime() if value else ArgumentBinding.omitted()


def _with_pointer_offset(
    parameters: list[Any],
    offset: ArgumentBinding,
) -> tuple[Any, ...]:
    """Append an element offset targeting the first non-scratch argument.

    The offset adjusts the global pointer during wrapper generation; it is
    not an extra argument to CUB's ``Load`` or ``Store`` method. A static
    binding embeds the offset, while a runtime binding adds an i64 input.
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
class BlockLoadStoreSemantics:
    """Normalized BlockLoad/BlockStore options and wrapper overloads.

    Built by ``make_block_load_store_semantics`` before block dimensions
    are known. ``make_block_load_store_specialization`` adds those
    dimensions and validates constraints that depend on the tile size.

    Attributes
    ----------
    kind : BlockLoadStoreKind
        Operation to perform and CUB method to invoke.
    dtype : Any
        Element type to bind to CUB's ``T`` template parameter.
    algorithm : BlockLoadStoreAlgorithm
        Memory-access strategy and per-thread item arrangement.
    items_per_thread : int
        Positive number of elements in each thread's item array.
    valid_items, oob_default, pointer_offset : ArgumentBinding
        Omitted, static, or runtime scalar arguments. A static value is
        embedded in the wrapper; a runtime binding describes an input
        without retaining its eventual value.
    has_full_tile : bool
        Whether an overload without a valid-item count is included. This
        describes available signatures, not the size of a particular call.
    parameters : tuple[tuple[Any, ...], ...]
        Ordered wrapper parameter descriptors, one tuple per overload.
        Each starts with scratch storage, the global pointer, and the
        per-thread array, followed by any count, default, and offset.
        Static descriptors do not consume runtime arguments; ``T`` and
        ``ITEMS_PER_THREAD`` dependencies are bound during specialization.
    """

    kind: BlockLoadStoreKind
    dtype: Any
    algorithm: BlockLoadStoreAlgorithm
    items_per_thread: int
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
    def method_name(self) -> str:
        return self.kind.value.title()

    @property
    def algorithm_cpp(self) -> str:
        return _algorithm_cpp_map(self.kind)[self.algorithm]

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        """Identify the normalized options and overloads before dimensions."""

        return (
            f"block_{self.kind.value}",
            semantic_token(self.dtype),
            self.algorithm.value,
            self.items_per_thread,
            self.valid_items.semantic_key,
            self.oob_default.semantic_key,
            self.has_full_tile,
            self.pointer_offset.semantic_key,
            semantic_token(self.parameters),
        )


def make_block_load_store_semantics(
    *,
    kind: str | BlockLoadStoreKind,
    dtype: Any,
    items_per_thread: int,
    algorithm: str | BlockLoadStoreAlgorithm,
    valid_items: bool | ArgumentBinding = False,
    oob_default: bool | ArgumentBinding = False,
    include_full_tile: bool = False,
    include_pointer_offset: bool | ArgumentBinding = False,
) -> BlockLoadStoreSemantics:
    """Normalize BlockLoad/BlockStore options and describe its overloads.

    Assemble parameter descriptors for CUB wrappers before a block shape
    is available. For scalar options, ``False`` omits the argument and
    ``True`` requests a runtime argument. Use ``ArgumentBinding.static``
    to embed a value, including a literal boolean ``oob_default``.

    Parameters
    ----------
    kind : str or BlockLoadStoreKind
        ``"load"`` or ``"store"``; selects the CUB class and method.
    dtype : Any
        Element type for the global pointer, per-thread array, and optional
        load default. Retained for later binding to CUB's ``T``.
    items_per_thread : int
        Positive integral number of elements per thread; booleans are
        rejected.
    algorithm : str or BlockLoadStoreAlgorithm
        Enum member, its lowercase value, or the corresponding fully
        qualified CUB token, such as ``"::cub::BLOCK_LOAD_TRANSPOSE"``.
        A CUB token must match ``kind``.
    valid_items : bool or ArgumentBinding, optional
        Number of valid elements in the whole block tile, measured from
        the possibly offset global pointer. A present binding selects a
        guarded overload. Static counts must fit signed i32; validation
        against the block tile size happens during specialization.
    oob_default : bool or ArgumentBinding, optional
        Value assigned to invalid load positions. Requires a load with
        ``valid_items`` present. Static defaults must be representable as
        finite C++ scalar literals. Without a default, invalid load values
        are unspecified; guarded stores leave out-of-range memory alone.
    include_full_tile : bool, optional
        Add an unguarded overload alongside the guarded overload. Requires
        ``valid_items`` to be present. When ``valid_items`` is omitted,
        the unguarded overload is already included with the default
        ``include_full_tile=False``.
    include_pointer_offset : bool or ArgumentBinding, optional
        Offset in elements applied to the global pointer. ``False`` omits
        it; ``True`` includes both unoffset and runtime-offset overloads.
        An explicit static or runtime binding includes only offset
        overloads; an omitted binding includes only unoffset overloads.
        Static offsets must be nonnegative and fit signed i64.

    Returns
    -------
    BlockLoadStoreSemantics
        Canonical options and ordered parameter descriptors. Runtime
        values, pointer validity, and memory extents are checked or
        established by the caller and later planning/lowering stages.

    Raises
    ------
    TypeError
        A scalar option is neither a boolean nor an ``ArgumentBinding``,
        a static count or offset is not an integer or is a boolean, or a
        static default is not a numeric scalar.
    ValueError
        The kind, algorithm, or item count is invalid; a static scalar is
        out of range or nonfinite; a static offset is negative; or the
        requested default/full-tile overload lacks a guarded load/store
        signature. A default is also rejected for stores.

    See Also
    --------
    make_block_load_store_specialization
        Bind block dimensions and validate the tile-dependent constraints.
    """

    pointer_offset_overload_cohort = isinstance(include_pointer_offset, bool)
    kind = BlockLoadStoreKind(kind)
    items_per_thread = normalize_positive_int(
        "items_per_thread", items_per_thread
    )
    algorithm = _normalize_algorithm(kind, algorithm)
    valid_items = _normalize_optional_binding(valid_items, name="valid_items")
    valid_items = normalize_i32_binding(valid_items, name="valid_items")
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
    if (
        kind is BlockLoadStoreKind.STORE
        and oob_default.kind is not BindingKind.OMITTED
    ):
        raise ValueError("oob_default is only valid for BlockLoad")
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

    return BlockLoadStoreSemantics(
        kind=kind,
        dtype=dtype,
        algorithm=algorithm,
        items_per_thread=items_per_thread,
        valid_items=valid_items,
        oob_default=oob_default,
        has_full_tile=has_full_tile,
        pointer_offset=pointer_offset,
        parameters=tuple(methods),
    )


def make_block_load_store_specialization(
    *,
    kind: str | BlockLoadStoreKind,
    dtype: Any,
    block_dim: tuple[int, int, int],
    items_per_thread: int,
    algorithm: str | BlockLoadStoreAlgorithm,
    valid_items: bool | ArgumentBinding = False,
    oob_default: bool | ArgumentBinding = False,
    include_full_tile: bool = False,
    include_pointer_offset: bool | ArgumentBinding = False,
) -> Algorithm:
    """Bind a BlockLoad/BlockStore operation to a concrete block shape.

    Normalize the options with ``make_block_load_store_semantics``, then
    construct ``Algorithm`` with all CUB template arguments bound. The
    returned description is ready for materialization; it does not compile
    device code, allocate scratch storage, or insert synchronization.

    Parameters
    ----------
    block_dim : tuple[int, int, int]
        Exact ``(x, y, z)`` launch dimensions, each a positive integer.
        Booleans are rejected. The block tile contains
        ``x * y * z * items_per_thread`` elements. ``warp_transpose`` and
        ``warp_transpose_timesliced`` require ``x * y * z`` divisible by 32.

    Other Parameters
    ----------------
    kind : str or BlockLoadStoreKind
        ``"load"`` or ``"store"``.
    dtype : Any
        Element type bound to CUB's ``T``.
    items_per_thread : int
        Positive integral number of elements per thread, excluding booleans.
    algorithm : str or BlockLoadStoreAlgorithm
        CUB strategy, accepted in the forms documented by
        ``make_block_load_store_semantics``.
    valid_items : bool or ArgumentBinding, optional
        Guarded-call count binding. Static counts must additionally lie
        between zero and the block tile size, inclusive. Runtime counts
        must satisfy the same bounds at execution time.
    oob_default : bool or ArgumentBinding, optional
        Optional invalid-item default for guarded loads.
    include_full_tile : bool, optional
        Include an unguarded overload alongside guarded overloads.
    include_pointer_offset : bool or ArgumentBinding, optional
        Select global-pointer offset overloads. Boolean and explicit-binding
        selection follow ``make_block_load_store_semantics``.

    Returns
    -------
    Algorithm
        Bound CUB template arguments, wrapper parameters, and operation
        metadata, ready for backend materialization.

    Raises
    ------
    TypeError
        An argument binding fails the semantics builder's type checks.
    ValueError
        An option fails the semantics builder's validation, the block
        shape is invalid, a static count exceeds the tile bounds, or a
        warp-transpose algorithm is selected for an incomplete warp.
    """

    block_dim = normalize_block_dim(block_dim)
    semantics = make_block_load_store_semantics(
        kind=kind,
        dtype=dtype,
        items_per_thread=items_per_thread,
        algorithm=algorithm,
        valid_items=valid_items,
        oob_default=oob_default,
        include_full_tile=include_full_tile,
        include_pointer_offset=include_pointer_offset,
    )
    if semantics.valid_items.kind is BindingKind.STATIC:
        value = semantics.valid_items.value
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("static valid_items must be an integer")
        value = int(value)
        tile_items = (
            semantics.items_per_thread
            * block_dim[0]
            * block_dim[1]
            * block_dim[2]
        )
        if not 0 <= value <= tile_items:
            raise ValueError(
                "static valid_items must be between zero and the block tile "
                f"size ({tile_items})"
            )
    block_threads = block_dim[0] * block_dim[1] * block_dim[2]
    if (
        semantics.algorithm
        in {
            BlockLoadStoreAlgorithm.WARP_TRANSPOSE,
            BlockLoadStoreAlgorithm.WARP_TRANSPOSE_TIMESLICED,
        }
        and block_threads % 32 != 0
    ):
        raise ValueError(
            f"Block{semantics.kind.value.title()} algorithm "
            f"{semantics.algorithm.value!r} "
            "requires a block size that is a multiple of 32"
        )
    title = semantics.kind.value.title()
    return Algorithm(
        struct_name=f"Block{title}",
        method_name=title,
        c_name=f"block_{semantics.kind.value}",
        includes=(f"cub/block/block_{semantics.kind.value}.cuh",),
        template_parameters=_TEMPLATE_PARAMETERS,
        parameters=semantics.parameters,
        template_arguments={
            "T": dtype,
            "BLOCK_DIM_X": block_dim[0],
            "ITEMS_PER_THREAD": semantics.items_per_thread,
            "ALGORITHM": semantics.algorithm_cpp,
            "BLOCK_DIM_Y": block_dim[1],
            "BLOCK_DIM_Z": block_dim[2],
        },
        metadata={
            "scope": "block",
            "primitive": semantics.kind.value,
            "algorithm": semantics.algorithm.value,
            "valid_items": semantics.has_valid_items,
            "oob_default": semantics.has_oob_default,
            "full_tile": semantics.has_full_tile,
            "pointer_offset": semantics.has_pointer_offset,
        },
    )


def make_block_load_specialization(**kwargs: Any) -> Algorithm:
    """Call ``make_block_load_store_specialization`` with ``kind="load"``.

    Accepts the same keyword arguments except ``kind`` and returns the
    resulting ``Algorithm``.
    """

    return make_block_load_store_specialization(
        kind=BlockLoadStoreKind.LOAD, **kwargs
    )


def make_block_store_specialization(**kwargs: Any) -> Algorithm:
    """Call ``make_block_load_store_specialization`` with ``kind="store"``.

    Accepts the same keyword arguments except ``kind`` and returns the
    resulting ``Algorithm``. ``oob_default`` must be omitted for a store.
    """

    return make_block_load_store_specialization(
        kind=BlockLoadStoreKind.STORE, **kwargs
    )
