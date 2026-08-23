# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Dedicated CUTLASS provider for CUB BlockTopK."""

from __future__ import annotations

import dataclasses
import hashlib
import operator
from typing import Any

from cutlass import cute as _cute
from cutlass._mlir.dialects import llvm
from cutlass.base_dsl.typing import Int32, Integer, Uint32
from cutlass.cute.ffi import ffi

from cuda.coop._core import ResultVisibility
from cuda.coop._core.block import (
    BLOCK_TOPK_TYPE,
    ArgumentBinding,
    BlockTopKSpec,
    TopKPayload,
    make_block_topk_spec,
)

from .._compiler import _rendering as _provider_rendering
from .._compiler import _state as _provider_state
from .._compiler import _storage as _provider_storage
from .._compiler import _types as _provider_types
from .._compiler._call_context import get_active_single_phase_context
from .._compiler._types import ALL_PROVIDER_TYPES as _ALL_PROVIDER_TYPES
from .._compiler._types import TYPE_SPECS as _TYPE_SPECS
from .._temp_storage import _topk_cub_temp_storage_requirement
from .._thread_data import ThreadData
from .._value_metadata import (
    attach_thread_data_metadata,
    metadata_for_group,
)

_ROOT_SCOPE = "cuda.coop.cutlass"
_TOPK_KEY_TYPES = _ALL_PROVIDER_TYPES
_TOPK_VALUE_TYPES = _ALL_PROVIDER_TYPES
_resolve_topk_type = _provider_state.make_provider_type_resolver(
    scope=_ROOT_SCOPE,
    root_scope=_ROOT_SCOPE,
    namespace="cutlass",
)


def _is_boolean_control(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        if isinstance(value, np.bool_):
            return True
    try:
        from cutlass.base_dsl.typing import Boolean
    except ImportError:
        return False
    return isinstance(value, Boolean)


def _static_index(value: Any, *, name: str) -> int | None:
    if _is_boolean_control(value):
        raise TypeError(f"{_ROOT_SCOPE}.topk {name} must be an integer")
    if isinstance(value, Integer):
        return None
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{_ROOT_SCOPE}.topk {name} must be an integer") from exc
    if isinstance(normalized, bool):
        raise TypeError(f"{_ROOT_SCOPE}.topk {name} must be an integer")
    return int(normalized)


def _validate_controls(
    *,
    k: Any,
    num_valid: Any,
    begin_bit: Any,
    end_bit: Any | None,
    key_type: type,
    tile_size: int,
) -> Any:
    static_k = _static_index(k, name="k")
    static_valid = (
        tile_size if num_valid is None else _static_index(num_valid, name="valid_items")
    )
    static_begin = _static_index(begin_bit, name="begin_bit")
    width_bits = _TYPE_SPECS[key_type].width_bits
    resolved_end = width_bits if end_bit is None else end_bit
    static_end = _static_index(resolved_end, name="end_bit")

    if static_valid is not None and not 1 <= static_valid <= tile_size:
        raise ValueError(f"{_ROOT_SCOPE}.topk valid_items must be in [1, {tile_size}]")
    if static_k is not None and static_k < 1:
        raise ValueError(f"{_ROOT_SCOPE}.topk k must be positive")
    if static_k is not None and static_valid is not None and static_k > static_valid:
        raise ValueError(f"{_ROOT_SCOPE}.topk k must be <= valid_items")
    if static_begin is not None and not 0 <= static_begin < width_bits:
        raise ValueError(f"{_ROOT_SCOPE}.topk begin_bit must be in [0, {width_bits})")
    if static_end is not None and not 0 < static_end <= width_bits:
        raise ValueError(f"{_ROOT_SCOPE}.topk end_bit must be in (0, {width_bits}]")
    if (
        static_begin is not None
        and static_end is not None
        and static_end <= static_begin
    ):
        raise ValueError(f"{_ROOT_SCOPE}.topk end_bit must exceed begin_bit")
    return resolved_end


@dataclasses.dataclass(frozen=True, eq=False)
class _CubTopKRequest:
    """One core-defined BlockTopK specialization for the CUTLASS bundle."""

    core_spec: BlockTopKSpec
    key_type: type
    value_type: type | None
    external_scratch: bool
    kind: str = "cub_block_topk"

    def __post_init__(self) -> None:
        if not isinstance(self.core_spec, BlockTopKSpec):
            raise TypeError("CUB TopK request requires a BlockTopKSpec")
        if self.key_type not in _TOPK_KEY_TYPES:
            raise TypeError("CUB TopK request has an unsupported key type")
        if self.value_type is not None and self.value_type not in _TOPK_VALUE_TYPES:
            raise TypeError("CUB TopK request has an unsupported value type")
        is_pairs = self.core_spec.payload is TopKPayload.PAIRS
        if is_pairs != (self.value_type is not None):
            raise ValueError("CUB TopK payload does not match its value type")
        if self.core_spec.block_dim[1:] != (1, 1):
            raise ValueError("CUB TopK requires a one-dimensional block")
        if self.core_spec.block_dim[0] > 1024:
            raise ValueError("CUB TopK block thread count must be <= 1024")

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            self.core_spec.semantic_key,
            self.key_type,
            self.value_type,
            self.external_scratch,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _CubTopKRequest):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)

    @property
    def items_per_thread(self) -> int:
        return self.core_spec.items_per_thread

    @property
    def block_threads(self) -> int:
        return self.core_spec.block_dim[0]

    @property
    def symbol_name(self) -> str:
        signature = hashlib.sha256(repr(self.semantic_key).encode()).hexdigest()[:12]
        value_token = (
            "keys"
            if self.value_type is None
            else f"pairs_{_TYPE_SPECS[self.value_type].token}"
        )
        storage_token = "external" if self.external_scratch else "internal"
        return (
            "cuda_coop_cutlass_cub_topk_"
            f"{self.core_spec.selection.value}_{value_token}_"
            f"{_TYPE_SPECS[self.key_type].token}_"
            f"b{self.block_threads}_x{self.items_per_thread}_"
            f"{storage_token}_{signature}"
        )


def _render_cub_topk(request: _CubTopKRequest) -> list[str]:
    request.__post_init__()
    key_spec = _TYPE_SPECS[request.key_type]
    value_spec = None if request.value_type is None else _TYPE_SPECS[request.value_type]
    key_params = [
        f"{key_spec.cpp_type} key{index}" for index in range(request.items_per_thread)
    ]
    value_params = (
        []
        if value_spec is None
        else [
            f"{value_spec.cpp_type} value{index}"
            for index in range(request.items_per_thread)
        ]
    )
    params = [
        *key_params,
        *value_params,
        "int k",
        "int num_valid",
        "int begin_bit",
        "int end_bit",
    ]
    if request.external_scratch:
        params.extend(
            (
                "unsigned int temp_storage_smem_addr",
                "int temp_storage_bytes",
                "int temp_storage_auto_sync",
            )
        )
    params.append(f"{key_spec.cpp_type}* result_keys")
    if value_spec is not None:
        params.append(f"{value_spec.cpp_type}* result_values")

    template_arguments = (
        f"{key_spec.cpp_type}, {request.block_threads}, "
        f"{request.items_per_thread}, "
        f"{value_spec.cpp_type if value_spec is not None else '::cub::NullType'}, "
        "1, 1"
    )
    storage_lines = (
        [
            "  if (temp_storage_bytes < (int)sizeof(",
            "          typename implementation_type::TempStorage)) {",
            '    asm volatile("trap;");',
            "  }",
            "  void* storage_ptr =",
            "      cuda_coop_cutlass_shared_ptr(temp_storage_smem_addr);",
            "  if (((unsigned long long)storage_ptr %",
            "       (unsigned long long)alignof(",
            "           typename implementation_type::TempStorage)) != 0ull) {",
            '    asm volatile("trap;");',
            "  }",
            "  auto& storage = *reinterpret_cast<",
            "      typename implementation_type::TempStorage*>(storage_ptr);",
            "  if (temp_storage_auto_sync != 0) {",
            "    cuda_coop_cutlass_block_sync();",
            "  }",
        ]
        if request.external_scratch
        else [
            "  __shared__ typename implementation_type::TempStorage storage;",
        ]
    )
    barrier_lines = (
        [
            "  if (temp_storage_auto_sync != 0) {",
            "    cuda_coop_cutlass_block_sync();",
            "  }",
        ]
        if request.external_scratch
        else ["  cuda_coop_cutlass_block_sync();"]
    )
    key_values = ", ".join(f"key{index}" for index in range(request.items_per_thread))
    value_values = (
        ""
        if value_spec is None
        else ", ".join(f"value{index}" for index in range(request.items_per_thread))
    )
    call_arguments = ["keys"]
    if value_spec is not None:
        call_arguments.append("values")
    call_arguments.extend(("k", "valid_items", "begin_bit", "end_bit"))

    return [
        "#ifndef CUDA_COOP_CUTLASS_BLOCK_TOPK_COOP_DEFINED",
        "#define CUDA_COOP_CUTLASS_BLOCK_TOPK_COOP_DEFINED",
        "}",
        *BLOCK_TOPK_TYPE.code.splitlines(),
        'extern "C" {',
        "#endif",
        f"void {request.symbol_name}({', '.join(params)}) {{",
        f"  using implementation_type = ::cub::BlockTopKCoop<{template_arguments}>;",
        "  constexpr int tile_items =",
        f"      {request.block_threads} * {request.items_per_thread};",
        "  if (cuda_coop_cutlass_ntid_y() != 1u ||",
        "      cuda_coop_cutlass_ntid_z() != 1u ||",
        "      cuda_coop_cutlass_group_threads() !=",
        f"          (unsigned int){request.block_threads}) {{",
        '    asm volatile("trap;");',
        "  }",
        "  int valid_items = num_valid < 0 ? tile_items : num_valid;",
        "  if (valid_items <= 0 || valid_items > tile_items ||",
        "      k <= 0 || k > valid_items || begin_bit < 0 ||",
        f"      begin_bit >= {key_spec.width_bits} || end_bit <= begin_bit ||",
        f"      end_bit > {key_spec.width_bits}) {{",
        '    asm volatile("trap;");',
        "  }",
        f"  {key_spec.cpp_type} keys[{request.items_per_thread}] = {{{key_values}}};",
        *(
            []
            if value_spec is None
            else [
                f"  {value_spec.cpp_type} values[{request.items_per_thread}] = "
                f"{{{value_values}}};"
            ]
        ),
        *storage_lines,
        f"  implementation_type(storage).{request.core_spec.method_name}("
        f"{', '.join(call_arguments)});",
        *barrier_lines,
        "#pragma unroll",
        f"  for (int i = 0; i < {request.items_per_thread}; ++i) {{",
        "    result_keys[i] = keys[i];",
        *([] if value_spec is None else ["    result_values[i] = values[i];"]),
        "  }",
        "}",
    ]


def _register_renderer() -> None:
    _provider_rendering.register_bundle_renderer(
        "cub_block_topk",
        render=_render_cub_topk,
        include_lines=("#include <cub/block/block_topk.cuh>",),
        cccl_headers=(
            (
                "#include <cub/block/block_topk.cuh>",
                "cub/block/block_topk.cuh",
            ),
        ),
    )


_register_renderer()


def _resolve_topk_thread_data_value_type(
    value: ThreadData,
    *,
    allowed: frozenset[type],
    feature: str,
) -> tuple[type, tuple[Any, ...]]:
    return _provider_types.resolve_thread_data_value_type(
        value,
        allowed=allowed,
        feature=feature,
        scope=_ROOT_SCOPE,
        resolve_type=_resolve_topk_type,
    )


def _resolve_topk_thread_data_pair_types(
    *,
    key: Any,
    value: Any,
    allowed_key_types: frozenset[type],
    allowed_value_types: frozenset[type],
    feature: str,
) -> tuple[type, tuple[Any, ...], ThreadData, type, tuple[Any, ...], ThreadData]:
    return _provider_types.resolve_thread_data_pair_types(
        key=key,
        value=value,
        allowed_key_types=allowed_key_types,
        allowed_value_types=allowed_value_types,
        feature=feature,
        scope=_ROOT_SCOPE,
        resolve_type=_resolve_topk_type,
    )


def _topk_temp_storage_requirement(
    *,
    block_threads: int,
    items_per_thread: int,
    key_type: type,
    value_type: type | None = None,
) -> tuple[int, int]:
    return _topk_cub_temp_storage_requirement(
        block_threads=block_threads,
        items_per_thread=items_per_thread,
        key_bytes=_provider_types.type_size_bytes(key_type),
        value_bytes=(
            0 if value_type is None else _provider_types.type_size_bytes(value_type)
        ),
    )


def _storage_ffi_args(primitive_name: str) -> tuple[bool, tuple[Any, ...]]:
    context = get_active_single_phase_context()
    if context is None or context.temp_storage is None:
        return False, ()
    return True, _provider_storage.temp_storage_ffi_args(
        primitive_name,
        scope=_ROOT_SCOPE,
    )


def _make_request(
    *,
    key_type: type,
    value_type: type | None,
    items_per_thread: int,
    block_threads: int,
    descending: bool,
    num_valid: Any,
    external_scratch: bool,
) -> _CubTopKRequest:
    core_spec = make_block_topk_spec(
        key_dtype=key_type,
        value_dtype=value_type,
        block_dim=(block_threads, 1, 1),
        items_per_thread=items_per_thread,
        selection="max" if descending else "min",
        num_valid=(
            ArgumentBinding.omitted()
            if num_valid is None
            else ArgumentBinding.runtime()
        ),
        begin_bit=ArgumentBinding.runtime(),
        end_bit=ArgumentBinding.runtime(),
    )
    return _CubTopKRequest(
        core_spec=core_spec,
        key_type=key_type,
        value_type=value_type,
        external_scratch=external_scratch,
    )


def _result_payload(
    tensor: Any,
    *,
    source: Any,
    value_type: type,
    group: Any,
) -> Any:
    metadata = metadata_for_group(group, visibility=ResultVisibility.PER_MEMBER)
    if isinstance(source, ThreadData):
        values = tuple(tensor[index] for index in range(source.items_per_thread))
        return attach_thread_data_metadata(
            ThreadData.from_values(
                *values,
                dtype=_provider_types.thread_data_output_dtype(
                    source,
                    value_type,
                ),
            ),
            metadata,
        )
    return _provider_state.remember_scalar_result_type(
        tensor[0],
        value_type,
        scope=_ROOT_SCOPE,
        group_metadata=metadata,
    )


def provider_topk_keys(
    *,
    group: Any,
    key: Any,
    k: Any,
    num_valid: Any,
    begin_bit: Any,
    end_bit: Any | None,
    descending: bool,
    block_threads: int,
    temp_storage_primitive_name: str,
) -> Any:
    if isinstance(key, ThreadData):
        key_type, key_values = _resolve_topk_thread_data_value_type(
            key,
            allowed=_TOPK_KEY_TYPES,
            feature="topk_keys",
        )
        items_per_thread = key.items_per_thread
    else:
        key_type = _resolve_topk_type(
            key,
            allowed=_TOPK_KEY_TYPES,
            feature="topk_keys",
        )
        key_values = (key,)
        items_per_thread = 1
    resolved_end = _validate_controls(
        k=k,
        num_valid=num_valid,
        begin_bit=begin_bit,
        end_bit=end_bit,
        key_type=key_type,
        tile_size=block_threads * items_per_thread,
    )
    external_scratch, storage_args = _storage_ffi_args(temp_storage_primitive_name)
    request = _make_request(
        key_type=key_type,
        value_type=None,
        items_per_thread=items_per_thread,
        block_threads=block_threads,
        descending=descending,
        num_valid=num_valid,
        external_scratch=external_scratch,
    )
    _provider_state.register_request(request)
    result = _cute.make_rmem_tensor(items_per_thread, key_type)
    ffi(
        name=request.symbol_name,
        params_types=[
            *([key_type] * items_per_thread),
            Int32,
            Int32,
            Int32,
            Int32,
            *([Uint32, Int32, Int32] if external_scratch else []),
            llvm.PointerType.get(0),
        ],
        return_type=None,
    )(
        *key_values,
        _provider_types.as_int32(k),
        _provider_types.as_valid_items_arg(num_valid, scope=_ROOT_SCOPE),
        _provider_types.as_int32(begin_bit),
        _provider_types.as_int32(resolved_end),
        *storage_args,
        result.iterator.llvm_ptr,
    )
    return _result_payload(
        result,
        source=key,
        value_type=key_type,
        group=group,
    )


def provider_topk_pairs(
    *,
    group: Any,
    key: Any,
    value: Any,
    k: Any,
    num_valid: Any,
    begin_bit: Any,
    end_bit: Any | None,
    descending: bool,
    block_threads: int,
    temp_storage_primitive_name: str,
) -> tuple[Any, Any]:
    if isinstance(key, ThreadData) or isinstance(value, ThreadData):
        (
            key_type,
            key_values,
            key_data,
            value_type,
            value_values,
            value_data,
        ) = _resolve_topk_thread_data_pair_types(
            key=key,
            value=value,
            allowed_key_types=_TOPK_KEY_TYPES,
            allowed_value_types=_TOPK_VALUE_TYPES,
            feature="topk_pairs",
        )
        items_per_thread = key_data.items_per_thread
    else:
        key_type = _resolve_topk_type(
            key,
            allowed=_TOPK_KEY_TYPES,
            feature="topk_pairs",
        )
        value_type = _resolve_topk_type(
            value,
            allowed=_TOPK_VALUE_TYPES,
            feature="topk_pairs",
        )
        key_values = (key,)
        value_values = (value,)
        key_data = key
        value_data = value
        items_per_thread = 1
    resolved_end = _validate_controls(
        k=k,
        num_valid=num_valid,
        begin_bit=begin_bit,
        end_bit=end_bit,
        key_type=key_type,
        tile_size=block_threads * items_per_thread,
    )
    external_scratch, storage_args = _storage_ffi_args(temp_storage_primitive_name)
    request = _make_request(
        key_type=key_type,
        value_type=value_type,
        items_per_thread=items_per_thread,
        block_threads=block_threads,
        descending=descending,
        num_valid=num_valid,
        external_scratch=external_scratch,
    )
    _provider_state.register_request(request)
    result_keys = _cute.make_rmem_tensor(items_per_thread, key_type)
    result_values = _cute.make_rmem_tensor(items_per_thread, value_type)
    ffi(
        name=request.symbol_name,
        params_types=[
            *([key_type] * items_per_thread),
            *([value_type] * items_per_thread),
            Int32,
            Int32,
            Int32,
            Int32,
            *([Uint32, Int32, Int32] if external_scratch else []),
            llvm.PointerType.get(0),
            llvm.PointerType.get(0),
        ],
        return_type=None,
    )(
        *key_values,
        *value_values,
        _provider_types.as_int32(k),
        _provider_types.as_valid_items_arg(num_valid, scope=_ROOT_SCOPE),
        _provider_types.as_int32(begin_bit),
        _provider_types.as_int32(resolved_end),
        *storage_args,
        result_keys.iterator.llvm_ptr,
        result_values.iterator.llvm_ptr,
    )
    return (
        _result_payload(
            result_keys,
            source=key_data,
            value_type=key_type,
            group=group,
        ),
        _result_payload(
            result_values,
            source=value_data,
            value_type=value_type,
            group=group,
        ),
    )


__all__ = ["_CubTopKRequest", "provider_topk_keys", "provider_topk_pairs"]
