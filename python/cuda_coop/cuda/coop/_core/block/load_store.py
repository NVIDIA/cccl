# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral CUB BlockLoad and BlockStore semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Any

from .._algorithm import Algorithm, AlgorithmSpec
from .._bindings import (
    ArgumentBinding,
    BindingKind,
    _cxx_scalar_literal,
    _normalize_i32_binding,
    _normalize_i64_binding,
    i32_parameter,
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
    LOAD = "load"
    STORE = "store"


class BlockLoadStoreAlgorithm(str, Enum):
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
        _LOAD_ALGORITHM_CPP if kind is BlockLoadStoreKind.LOAD else _STORE_ALGORITHM_CPP
    )


def _normalize_algorithm(
    kind: BlockLoadStoreKind,
    algorithm: str | BlockLoadStoreAlgorithm,
) -> BlockLoadStoreAlgorithm:
    mapping = _algorithm_cpp_map(kind)
    if isinstance(algorithm, BlockLoadStoreAlgorithm):
        return algorithm
    if isinstance(algorithm, str):
        for candidate, cpp in mapping.items():
            if algorithm in {candidate.value, cpp}:
                return candidate
    raise ValueError(f"unsupported Block{kind.value.title()} algorithm {algorithm!r}")


def _base_parameters(kind: BlockLoadStoreKind) -> list[Any]:
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
    if isinstance(value, ArgumentBinding):
        return value
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool or ArgumentBinding")
    return ArgumentBinding.runtime() if value else ArgumentBinding.omitted()


def _with_pointer_offset(
    parameters: list[Any],
    offset: ArgumentBinding,
) -> tuple[Any, ...]:
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
    """Dimension-independent BlockLoad or BlockStore call contract."""

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


@dataclass(frozen=True)
class BlockLoadStoreSpec:
    """Fully specialized CUB BlockLoad or BlockStore semantics."""

    specialization: AlgorithmSpec
    call: BlockLoadStoreSemantics
    block_dim: tuple[int, int, int]

    @property
    def kind(self) -> BlockLoadStoreKind:
        return self.call.kind

    @property
    def algorithm(self) -> BlockLoadStoreAlgorithm:
        return self.call.algorithm

    @property
    def items_per_thread(self) -> int:
        return self.call.items_per_thread

    @property
    def has_valid_items(self) -> bool:
        return self.call.has_valid_items

    @property
    def has_oob_default(self) -> bool:
        return self.call.has_oob_default

    @property
    def has_full_tile(self) -> bool:
        return self.call.has_full_tile

    @property
    def has_pointer_offset(self) -> bool:
        return self.call.has_pointer_offset

    @property
    def method_name(self) -> str:
        return self.specialization.method_name

    @property
    def algorithm_cpp(self) -> str:
        return self.call.algorithm_cpp

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.specialization.semantic_key


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
    """Build canonical dimension-independent BlockLoad/BlockStore semantics."""

    pointer_offset_overload_cohort = isinstance(include_pointer_offset, bool)
    kind = BlockLoadStoreKind(kind)
    items_per_thread = normalize_positive_int("items_per_thread", items_per_thread)
    algorithm = _normalize_algorithm(kind, algorithm)
    valid_items = _normalize_optional_binding(valid_items, name="valid_items")
    valid_items = _normalize_i32_binding(valid_items, name="valid_items")
    oob_default = _normalize_optional_binding(oob_default, name="oob_default")
    pointer_offset = _normalize_optional_binding(
        include_pointer_offset,
        name="include_pointer_offset",
    )
    pointer_offset = _normalize_i64_binding(pointer_offset, name="pointer offset")
    if pointer_offset.kind is BindingKind.STATIC and int(pointer_offset.value) < 0:
        raise ValueError("static pointer offset must be nonnegative")
    if kind is BlockLoadStoreKind.STORE and oob_default.kind is not BindingKind.OMITTED:
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
        pointer_offset.kind is BindingKind.OMITTED or pointer_offset_overload_cohort
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
                    _cxx_scalar_literal(oob_default.value, name="oob_default"),
                    dtype,
                    name="oob_default",
                )
            )
        if pointer_offset.kind is BindingKind.OMITTED or pointer_offset_overload_cohort:
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


def make_block_load_store_spec(
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
) -> BlockLoadStoreSpec:
    """Build a fully specialized CUB BlockLoad or BlockStore description."""

    block_dim = normalize_block_dim(block_dim)
    call = make_block_load_store_semantics(
        kind=kind,
        dtype=dtype,
        items_per_thread=items_per_thread,
        algorithm=algorithm,
        valid_items=valid_items,
        oob_default=oob_default,
        include_full_tile=include_full_tile,
        include_pointer_offset=include_pointer_offset,
    )
    if call.valid_items.kind is BindingKind.STATIC:
        value = call.valid_items.value
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("static valid_items must be an integer")
        value = int(value)
        tile_items = call.items_per_thread * block_dim[0] * block_dim[1] * block_dim[2]
        if not 0 <= value <= tile_items:
            raise ValueError(
                "static valid_items must be between zero and the block tile "
                f"size ({tile_items})"
            )
    block_threads = block_dim[0] * block_dim[1] * block_dim[2]
    if (
        call.algorithm
        in {
            BlockLoadStoreAlgorithm.WARP_TRANSPOSE,
            BlockLoadStoreAlgorithm.WARP_TRANSPOSE_TIMESLICED,
        }
        and block_threads % 32 != 0
    ):
        raise ValueError(
            f"Block{call.kind.value.title()} algorithm {call.algorithm.value!r} "
            "requires a block size that is a multiple of 32"
        )
    title = call.kind.value.title()
    specialization = Algorithm(
        struct_name=f"Block{title}",
        method_name=title,
        c_name=f"block_{call.kind.value}",
        includes=(f"cub/block/block_{call.kind.value}.cuh",),
        template_parameters=_TEMPLATE_PARAMETERS,
        parameters=call.parameters,
    ).specialize(
        {
            "T": dtype,
            "BLOCK_DIM_X": block_dim[0],
            "ITEMS_PER_THREAD": call.items_per_thread,
            "ALGORITHM": call.algorithm_cpp,
            "BLOCK_DIM_Y": block_dim[1],
            "BLOCK_DIM_Z": block_dim[2],
        },
        metadata={
            "scope": "block",
            "primitive": call.kind.value,
            "algorithm": call.algorithm.value,
            "valid_items": call.has_valid_items,
            "oob_default": call.has_oob_default,
            "full_tile": call.has_full_tile,
            "pointer_offset": call.has_pointer_offset,
        },
    )
    return BlockLoadStoreSpec(
        specialization=specialization,
        call=call,
        block_dim=block_dim,
    )


def make_block_load_spec(**kwargs: Any) -> BlockLoadStoreSpec:
    return make_block_load_store_spec(kind=BlockLoadStoreKind.LOAD, **kwargs)


def make_block_store_spec(**kwargs: Any) -> BlockLoadStoreSpec:
    return make_block_load_store_spec(kind=BlockLoadStoreKind.STORE, **kwargs)
