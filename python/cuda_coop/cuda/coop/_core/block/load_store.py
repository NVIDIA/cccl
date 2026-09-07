# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral CUB BlockLoad and BlockStore semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .._algorithm import Algorithm, AlgorithmSpec
from .._symbols import semantic_token
from .._types import (
    INT32,
    INT64,
    Array,
    Dependency,
    Pointer,
    PointerOffset,
    Reference,
    TemplateParameter,
    TempStorageParameter,
    Value,
)


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


def _with_pointer_offset(parameters: list[Any]) -> tuple[Any, ...]:
    return (
        *parameters,
        PointerOffset(INT64, name="offset", pointer_arg_index=0),
    )


@dataclass(frozen=True)
class BlockLoadStoreSemantics:
    """Dimension-independent BlockLoad or BlockStore call contract."""

    kind: BlockLoadStoreKind
    dtype: Any
    algorithm: BlockLoadStoreAlgorithm
    items_per_thread: int
    has_valid_items: bool
    has_oob_default: bool
    has_full_tile: bool
    has_pointer_offset: bool
    parameters: tuple[tuple[Any, ...], ...]

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
            self.has_valid_items,
            self.has_oob_default,
            self.has_full_tile,
            self.has_pointer_offset,
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
    valid_items: bool = False,
    oob_default: bool = False,
    include_full_tile: bool = False,
    include_pointer_offset: bool = False,
) -> BlockLoadStoreSemantics:
    """Build canonical dimension-independent BlockLoad/BlockStore semantics."""

    kind = BlockLoadStoreKind(kind)
    if (
        not isinstance(items_per_thread, int)
        or isinstance(items_per_thread, bool)
        or items_per_thread < 1
    ):
        raise ValueError("items_per_thread must be a positive integer")
    algorithm = _normalize_algorithm(kind, algorithm)
    if kind is BlockLoadStoreKind.STORE and oob_default:
        raise ValueError("oob_default is only valid for BlockLoad")
    if oob_default and not valid_items:
        raise ValueError("oob_default requires a valid_items signature")
    if include_full_tile and not valid_items:
        raise ValueError("include_full_tile requires a valid_items signature")

    base = _base_parameters(kind)
    has_full_tile = not valid_items or include_full_tile
    methods: list[tuple[Any, ...]] = []
    if has_full_tile:
        methods.append(tuple(base))
    if valid_items:
        partial = [*base, Value(INT32, name="num_valid_items")]
        if oob_default:
            partial.append(Reference(_T, name="oob_default"))
        methods.append(tuple(partial))
        if include_pointer_offset:
            methods.append(_with_pointer_offset(partial))
    if include_pointer_offset and has_full_tile:
        methods.append(_with_pointer_offset(base))

    return BlockLoadStoreSemantics(
        kind=kind,
        dtype=dtype,
        algorithm=algorithm,
        items_per_thread=items_per_thread,
        has_valid_items=valid_items,
        has_oob_default=oob_default,
        has_full_tile=has_full_tile,
        has_pointer_offset=include_pointer_offset,
        parameters=tuple(methods),
    )


def make_block_load_store_spec(
    *,
    kind: str | BlockLoadStoreKind,
    dtype: Any,
    block_dim: tuple[int, int, int],
    items_per_thread: int,
    algorithm: str | BlockLoadStoreAlgorithm,
    valid_items: bool = False,
    oob_default: bool = False,
    include_full_tile: bool = False,
    include_pointer_offset: bool = False,
) -> BlockLoadStoreSpec:
    """Build a fully specialized CUB BlockLoad or BlockStore description."""

    block_dim = tuple(block_dim)
    if len(block_dim) != 3 or any(dim < 1 for dim in block_dim):
        raise ValueError("block_dim must contain three positive dimensions")
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
            "ITEMS_PER_THREAD": items_per_thread,
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
