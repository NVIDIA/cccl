# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral CUB physical and logical WarpLoad/Store semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .._algorithm import Algorithm, AlgorithmSpec, TypeDefinition
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

_MAX_WARP_THREADS = 32
_SUPPORTED_LOGICAL_WARP_THREADS = frozenset({1, 2, 4, 8, 16, 32})


class WarpLoadStoreKind(str, Enum):
    LOAD = "load"
    STORE = "store"


class WarpLoadStoreAlgorithm(str, Enum):
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
_PRESERVING_WARP_LOAD = TypeDefinition(
    name="cuda_coop_warp_load_preserving_invalid",
    code=r"""
namespace cub {
template <typename T,
          int ItemsPerThread,
          WarpLoadAlgorithm Algorithm,
          int LogicalWarpThreads>
class CudaCoopWarpLoadPreservingInvalid
{
  static_assert(LogicalWarpThreads > 0 && LogicalWarpThreads <= 32 &&
                  (LogicalWarpThreads & (LogicalWarpThreads - 1)) == 0,
                "cuda.coop WarpLoad requires a power-of-two width in [1, 32]");

  using primitive_type =
    WarpLoad<T, ItemsPerThread, Algorithm, LogicalWarpThreads>;

  primitive_type primitive;

public:
  using TempStorage = typename primitive_type::TempStorage;

  _CCCL_DEVICE _CCCL_FORCEINLINE CudaCoopWarpLoadPreservingInvalid()
      : primitive()
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE
  CudaCoopWarpLoadPreservingInvalid(TempStorage& temp_storage)
      : primitive(temp_storage)
  {}

  template <typename InputIteratorT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Load(
    InputIteratorT warp_iterator,
    T (&items)[ItemsPerThread])
  {
    primitive.Load(warp_iterator, items);
  }

  template <typename InputIteratorT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Load(
    InputIteratorT warp_iterator,
    T (&items)[ItemsPerThread],
    int valid_items)
  {
    T original[ItemsPerThread];
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int item = 0; item < ItemsPerThread; ++item)
    {
      original[item] = items[item];
    }

    primitive.Load(warp_iterator, items, valid_items);

    const int lane = static_cast<int>(::cuda::ptx::get_sreg_laneid()) %
                     LogicalWarpThreads;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int item = 0; item < ItemsPerThread; ++item)
    {
      if (lane * ItemsPerThread + item >= valid_items)
      {
        items[item] = original[item];
      }
    }
  }
};
} // namespace cub
""".strip(),
)
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


def _normalize_items_per_thread(items_per_thread: Any) -> int:
    if (
        not isinstance(items_per_thread, int)
        or isinstance(items_per_thread, bool)
        or items_per_thread < 1
    ):
        raise ValueError("items_per_thread must be a positive integer")
    return int(items_per_thread)


def _normalize_logical_warp_threads(threads_in_warp: Any) -> int:
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


@dataclass(frozen=True)
class WarpLoadStoreSemantics:
    """Provider-facing physical or logical Warp Load/Store contract."""

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
        return self.pointer_offset.kind is BindingKind.RUNTIME

    @property
    def method_name(self) -> str:
        return self.kind.value.title()

    @property
    def algorithm_cpp(self) -> str:
        return _algorithm_cpp_map(self.kind)[self.algorithm]

    @property
    def semantic_key(self) -> tuple[Any, ...]:
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


@dataclass(frozen=True)
class WarpLoadStoreSpec:
    """Fully specialized CUB physical or logical Warp Load/Store semantics."""

    specialization: AlgorithmSpec
    call: WarpLoadStoreSemantics

    @property
    def kind(self) -> WarpLoadStoreKind:
        return self.call.kind

    @property
    def algorithm(self) -> WarpLoadStoreAlgorithm:
        return self.call.algorithm

    @property
    def items_per_thread(self) -> int:
        return self.call.items_per_thread

    @property
    def threads_in_warp(self) -> int:
        return self.call.threads_in_warp

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
    def requires_runtime_effective_offset(self) -> bool:
        return self.call.requires_runtime_effective_offset

    @property
    def method_name(self) -> str:
        return self.specialization.method_name

    @property
    def algorithm_cpp(self) -> str:
        return self.call.algorithm_cpp

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.specialization.semantic_key


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
    """Build canonical provider-facing Warp Load/Store semantics."""

    pointer_offset_overload_cohort = isinstance(include_pointer_offset, bool)
    kind = WarpLoadStoreKind(kind)
    items_per_thread = _normalize_items_per_thread(items_per_thread)
    threads_in_warp = _normalize_logical_warp_threads(threads_in_warp)
    algorithm = _normalize_algorithm(kind, algorithm)
    valid_items = _normalize_optional_binding(valid_items, name="valid_items")
    valid_items = _normalize_i32_binding(valid_items, name="valid_items")
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
    pointer_offset = _normalize_i64_binding(pointer_offset, name="pointer offset")
    if pointer_offset.kind is BindingKind.STATIC and int(pointer_offset.value) < 0:
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


def make_warp_load_store_spec(
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
) -> WarpLoadStoreSpec:
    """Build a fully specialized CUB Warp Load/Store description."""

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
    preserve_invalid_items = (
        call.kind is WarpLoadStoreKind.LOAD
        and call.has_valid_items
        and not call.has_oob_default
        and call.algorithm is WarpLoadStoreAlgorithm.TRANSPOSE
    )
    title = call.kind.value.title()
    specialization = Algorithm(
        struct_name=(
            "CudaCoopWarpLoadPreservingInvalid"
            if preserve_invalid_items
            else f"Warp{title}"
        ),
        method_name=title,
        c_name=f"warp_{call.kind.value}",
        includes=(f"cub/warp/warp_{call.kind.value}.cuh",),
        template_parameters=_TEMPLATE_PARAMETERS,
        parameters=call.parameters,
        type_definitions=((_PRESERVING_WARP_LOAD,) if preserve_invalid_items else ()),
    ).specialize(
        {
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
            "effective_offset_stride": (call.threads_in_warp * call.items_per_thread),
            "preserves_invalid_items": preserve_invalid_items,
        },
    )
    return WarpLoadStoreSpec(specialization=specialization, call=call)


def make_warp_load_spec(**kwargs: Any) -> WarpLoadStoreSpec:
    return make_warp_load_store_spec(kind=WarpLoadStoreKind.LOAD, **kwargs)


def make_warp_store_spec(**kwargs: Any) -> WarpLoadStoreSpec:
    return make_warp_load_store_spec(kind=WarpLoadStoreKind.STORE, **kwargs)


__all__ = [
    "WarpLoadAlgorithm",
    "WarpLoadStoreAlgorithm",
    "WarpLoadStoreKind",
    "WarpLoadStoreSemantics",
    "WarpLoadStoreSpec",
    "WarpStoreAlgorithm",
    "make_warp_load_spec",
    "make_warp_load_store_semantics",
    "make_warp_load_store_spec",
    "make_warp_store_spec",
]
