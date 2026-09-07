# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral CUB WarpLoad and WarpStore semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
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
from ._common import _validate_items_per_thread, _validate_logical_warp_threads


class WarpLoadStoreKind(str, Enum):
    LOAD = "load"
    STORE = "store"


class WarpLoadStoreAlgorithm(str, Enum):
    DIRECT = "direct"
    STRIPED = "striped"
    VECTORIZE = "vectorize"
    TRANSPOSE = "transpose"


# Scope-specific aliases keep call sites descriptive while sharing the CUB
# algorithm domain intentionally common to WarpLoad and WarpStore.
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
        _LOAD_ALGORITHM_CPP if kind is WarpLoadStoreKind.LOAD else _STORE_ALGORITHM_CPP
    )


@dataclass(frozen=True)
class WarpLoadStoreSpec:
    """Fully specialized WarpLoad or WarpStore call semantics."""

    specialization: AlgorithmSpec
    kind: WarpLoadStoreKind
    algorithm: WarpLoadStoreAlgorithm
    items_per_thread: int
    threads_in_warp: int
    has_valid_items: bool
    has_oob_default: bool
    has_full_tile: bool
    has_pointer_offset: bool

    @property
    def method_name(self) -> str:
        return self.specialization.method_name

    @property
    def algorithm_cpp(self) -> str:
        return _algorithm_cpp_map(self.kind)[self.algorithm]

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.specialization.semantic_key


def _normalize_algorithm(
    kind: WarpLoadStoreKind,
    algorithm: str | WarpLoadStoreAlgorithm,
) -> WarpLoadStoreAlgorithm:
    mapping = _algorithm_cpp_map(kind)
    if isinstance(algorithm, WarpLoadStoreAlgorithm):
        return algorithm
    if isinstance(algorithm, str):
        for candidate, cpp in mapping.items():
            if algorithm in {candidate.value, cpp}:
                return candidate
    raise ValueError(f"unsupported Warp{kind.value.title()} algorithm {algorithm!r}")


def _base_parameters(kind: WarpLoadStoreKind) -> list[Any]:
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


def make_warp_load_store_spec(
    *,
    kind: str | WarpLoadStoreKind,
    dtype: Any,
    items_per_thread: int,
    threads_in_warp: int,
    algorithm: str | WarpLoadStoreAlgorithm,
    valid_items: bool | ArgumentBinding = False,
    oob_default: bool | ArgumentBinding = False,
    include_full_tile: bool = False,
    include_pointer_offset: bool | ArgumentBinding = False,
) -> WarpLoadStoreSpec:
    """Build canonical WarpLoad or WarpStore semantics."""

    pointer_offset_overload_cohort = isinstance(include_pointer_offset, bool)
    kind = WarpLoadStoreKind(kind)
    items_per_thread = _validate_items_per_thread(items_per_thread)
    threads_in_warp = _validate_logical_warp_threads(threads_in_warp)
    algorithm = _normalize_algorithm(kind, algorithm)
    valid_items = _normalize_optional_binding(valid_items, name="valid_items")
    valid_items = _normalize_i32_binding(valid_items, name="valid_items")
    oob_default = _normalize_optional_binding(oob_default, name="oob_default")
    pointer_offset = _normalize_optional_binding(
        include_pointer_offset,
        name="include_pointer_offset",
    )
    pointer_offset = _normalize_i64_binding(pointer_offset, name="pointer offset")
    if valid_items.kind is BindingKind.STATIC:
        value = valid_items.value
        tile_items = items_per_thread * threads_in_warp
        if not 0 <= value <= tile_items:
            raise ValueError(
                "static valid_items must be between zero and the warp tile "
                f"size ({tile_items})"
            )
    if kind is WarpLoadStoreKind.STORE and oob_default.kind is not BindingKind.OMITTED:
        raise ValueError("oob_default is only valid for WarpLoad")
    if (
        oob_default.kind is not BindingKind.OMITTED
        and valid_items.kind is BindingKind.OMITTED
    ):
        raise ValueError("oob_default requires a valid_items signature")
    if include_full_tile and valid_items.kind is BindingKind.OMITTED:
        raise ValueError("include_full_tile requires a valid_items signature")

    base_parameters = _base_parameters(kind)
    has_full_tile = valid_items.kind is BindingKind.OMITTED or include_full_tile
    methods: list[tuple[Any, ...]] = []
    if has_full_tile and (
        pointer_offset.kind is BindingKind.OMITTED or pointer_offset_overload_cohort
    ):
        methods.append(tuple(base_parameters))
    if valid_items.kind is not BindingKind.OMITTED:
        num_valid_items = i32_parameter(valid_items, name="num_valid_items")
        assert num_valid_items is not None
        partial_parameters = [*base_parameters, num_valid_items]
        if oob_default.kind is BindingKind.RUNTIME:
            partial_parameters.append(Value(dtype, name="oob_default"))
        elif oob_default.kind is BindingKind.STATIC:
            partial_parameters.append(
                CxxFunction(
                    _cxx_scalar_literal(oob_default.value, name="oob_default"),
                    dtype,
                    name="oob_default",
                )
            )
        if pointer_offset.kind is BindingKind.OMITTED or pointer_offset_overload_cohort:
            methods.append(tuple(partial_parameters))
        if pointer_offset.kind is not BindingKind.OMITTED:
            methods.append(_with_pointer_offset(partial_parameters, pointer_offset))
    if pointer_offset.kind is not BindingKind.OMITTED and has_full_tile:
        methods.append(_with_pointer_offset(base_parameters, pointer_offset))

    algorithm_cpp = _algorithm_cpp_map(kind)[algorithm]
    title = kind.value.title()
    specialization = Algorithm(
        struct_name=f"Warp{title}",
        method_name=title,
        c_name=f"warp_{kind.value}",
        includes=(f"cub/warp/warp_{kind.value}.cuh",),
        template_parameters=_TEMPLATE_PARAMETERS,
        parameters=tuple(methods),
    ).specialize(
        {
            "T": dtype,
            "ITEMS_PER_THREAD": items_per_thread,
            "ALGORITHM": algorithm_cpp,
            "LOGICAL_WARP_THREADS": threads_in_warp,
        },
        metadata={
            "scope": "warp",
            "primitive": kind.value,
            "algorithm": algorithm,
            "valid_items": valid_items.semantic_key,
            "oob_default": oob_default.semantic_key,
            "full_tile": has_full_tile,
            "pointer_offset": pointer_offset.semantic_key,
        },
    )
    return WarpLoadStoreSpec(
        specialization=specialization,
        kind=kind,
        algorithm=algorithm,
        items_per_thread=items_per_thread,
        threads_in_warp=threads_in_warp,
        has_valid_items=valid_items.kind is not BindingKind.OMITTED,
        has_oob_default=oob_default.kind is not BindingKind.OMITTED,
        has_full_tile=has_full_tile,
        has_pointer_offset=pointer_offset.kind is not BindingKind.OMITTED,
    )


def make_warp_load_spec(
    *,
    dtype: Any,
    items_per_thread: int,
    threads_in_warp: int,
    algorithm: str | WarpLoadStoreAlgorithm,
    valid_items: bool | ArgumentBinding = False,
    oob_default: bool | ArgumentBinding = False,
    include_full_tile: bool = False,
    include_pointer_offset: bool | ArgumentBinding = False,
) -> WarpLoadStoreSpec:
    return make_warp_load_store_spec(
        kind=WarpLoadStoreKind.LOAD,
        dtype=dtype,
        items_per_thread=items_per_thread,
        threads_in_warp=threads_in_warp,
        algorithm=algorithm,
        valid_items=valid_items,
        oob_default=oob_default,
        include_full_tile=include_full_tile,
        include_pointer_offset=include_pointer_offset,
    )


def make_warp_store_spec(
    *,
    dtype: Any,
    items_per_thread: int,
    threads_in_warp: int,
    algorithm: str | WarpLoadStoreAlgorithm,
    valid_items: bool | ArgumentBinding = False,
    oob_default: bool | ArgumentBinding = False,
    include_full_tile: bool = False,
    include_pointer_offset: bool | ArgumentBinding = False,
) -> WarpLoadStoreSpec:
    return make_warp_load_store_spec(
        kind=WarpLoadStoreKind.STORE,
        dtype=dtype,
        items_per_thread=items_per_thread,
        threads_in_warp=threads_in_warp,
        algorithm=algorithm,
        valid_items=valid_items,
        oob_default=oob_default,
        include_full_tile=include_full_tile,
        include_pointer_offset=include_pointer_offset,
    )
