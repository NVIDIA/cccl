# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# ruff: noqa: I001

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Mapping
from typing import Any

import dataclasses
import numpy as np
from cuda.coop._core.block import (
    BLOCK_ROW_REDUCE_INCLUDES,
    ArgumentBinding,
    BlockRowReduceSpec,
    TopKPayload,
    make_block_row_reduce_spec,
    make_block_topk_spec,
    normalize_block_row_reduce_geometry,
)
from cutlass.base_dsl.typing import (
    Int32,
    Uint32,
)
from cutlass.base_dsl.common import DSLRuntimeError
from cutlass.cute.ffi import ffi

from .. import _provider as _provider_support
from .. import _provider_bundle as _provider_bundle_support
from .._provider import ALL_PROVIDER_TYPES as _ALL_PROVIDER_TYPES
from .._provider import SCAN_REDUCE_TYPES as _SCAN_REDUCE_TYPES
from .._provider import TYPE_SPECS as _TYPE_SPECS
from .._provider import TypeSpec as _TypeSpec
from .._provider import as_int32 as _as_int32_impl
from .._provider import as_int32_bool as _as_int32_bool_impl
from .._provider import as_valid_items_arg as _as_valid_items_arg_impl
from .._provider import registered_bundle_headers as _registered_bundle_headers
from .._provider import (
    require_single_item_thread_data as _require_single_item_thread_data_impl,
)
from .._provider import (
    resolve_thread_data_pair_types as _resolve_thread_data_pair_types_impl,
)
from .._provider import type_size_bytes as _type_size_bytes
from .._provider import validate_radix_bit_range as _validate_radix_bit_range_impl
from .._single_phase import get_active_single_phase_context
from .._temp_storage import (
    _block_row_reduce_temp_storage_alignment,
    _block_row_reduce_temp_storage_bytes,
    _topk_cub_temp_storage_requirement,
    infer_group_width as _infer_group_width,
)
from .._thread_data import ThreadData
from .._scope import BLOCK_SCOPE as _SCOPE, ROOT_SCOPE as _ROOT_SCOPE
from ._single_phase import TempStorage

#
# Batched provider JIT state
#

_TOPK_KEY_TYPES = _ALL_PROVIDER_TYPES
_TOPK_VALUE_TYPES = _ALL_PROVIDER_TYPES
_ORDINARY_TOPK_DTYPES = {
    int: _provider_support.Int32,
    float: _provider_support.Float32,
    np.uint8: _provider_support.Uint8,
    np.int32: _provider_support.Int32,
    np.uint32: _provider_support.Uint32,
    np.int64: _provider_support.Int64,
    np.uint64: _provider_support.Uint64,
    np.float32: _provider_support.Float32,
    np.float64: _provider_support.Float64,
}
_IMPLICIT_TEMP_STORAGE_ALIGNMENT = 8
_PROVIDER_DIR = os.path.dirname(__file__)


@dataclasses.dataclass(frozen=True, eq=False)
class _BlockRowReduceRequest:
    """CuTe projection of one core BlockRowReduce specialization."""

    core_spec: BlockRowReduceSpec
    value_type: type

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (self.value_type, self.core_spec.semantic_key)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _BlockRowReduceRequest) and (
            self.semantic_key == other.semantic_key
        )

    def __hash__(self) -> int:
        return hash(self.semantic_key)

    @property
    def kind(self) -> str:
        return "row_sum"

    @property
    def items_per_thread(self) -> int:
        return 1

    @property
    def rows_per_block(self) -> int:
        return self.core_spec.rows_per_block

    @property
    def warps_per_row(self) -> int:
        return self.core_spec.warps_per_row

    @property
    def symbol_name(self) -> str:
        return (
            "cuda_coop_cutlass_row_sum_"
            f"{_TYPE_SPECS[self.value_type].token}"
            f"_r{self.rows_per_block}_w{self.warps_per_row}"
        )


@dataclasses.dataclass(frozen=True)
class _ShimRequest:
    kind: str
    op: str | None = None
    value_type: type | None = None
    key_type: type | None = None
    pair_value_type: type | None = None
    items_per_thread: int = 1
    block_threads: int | None = None

    @property
    def _arity_suffix(self) -> str:
        if self.items_per_thread > 1:
            return f"_x{self.items_per_thread}"
        return ""

    @property
    def symbol_name(self) -> str:
        if self.kind == "topk_keys":
            assert self.key_type is not None
            assert self.op is not None
            assert self.block_threads is not None
            base = (
                f"cuda_coop_cutlass_topk_{self.op}_keys_"
                f"{_TYPE_SPECS[self.key_type].token}_bt{self.block_threads}"
            )
            return f"{base}{self._arity_suffix}"
        if self.kind == "topk_pair_keys":
            assert self.key_type is not None and self.pair_value_type is not None
            assert self.op is not None
            assert self.block_threads is not None
            base = (
                f"cuda_coop_cutlass_topk_{self.op}_pair_keys_"
                f"k{_TYPE_SPECS[self.key_type].token}_"
                f"v{_TYPE_SPECS[self.pair_value_type].token}_"
                f"bt{self.block_threads}"
            )
            return f"{base}{self._arity_suffix}"
        if self.kind == "topk_pair_values":
            assert self.key_type is not None and self.pair_value_type is not None
            assert self.op is not None
            assert self.block_threads is not None
            base = (
                f"cuda_coop_cutlass_topk_{self.op}_pair_values_"
                f"k{_TYPE_SPECS[self.key_type].token}_"
                f"v{_TYPE_SPECS[self.pair_value_type].token}_"
                f"bt{self.block_threads}"
            )
            return f"{base}{self._arity_suffix}"
        raise ValueError(f"Unsupported shim request kind: {self.kind}")


def _get_cute_dsl():
    return _provider_support._get_cute_dsl()


def _ensure_trace_hook_registered() -> None:
    _provider_support.ensure_trace_hook_registered(
        finalizer=_trace_finalize_hook,
        scope=_ROOT_SCOPE,
        get_cute_dsl=_get_cute_dsl,
    )


def _active_session() -> _provider_support.BundleSession:
    return _provider_support.active_bundle_session_for(
        get_cute_dsl=_get_cute_dsl,
        ensure_trace_hook=_ensure_trace_hook_registered,
    )


def _snapshot_active_session_state():
    return _provider_support.snapshot_active_session_state_for(
        get_cute_dsl=_get_cute_dsl,
    )


def _restore_active_session_state(snapshot) -> None:
    _provider_support.restore_active_session_state_for(
        snapshot,
        get_cute_dsl=_get_cute_dsl,
    )


def _materialize_temp_storage_binding(
    temp_storage: TempStorage,
) -> _provider_support._TempStorageBinding:
    return _provider_support.materialize_temp_storage_binding(
        temp_storage,
        scope=_SCOPE,
        active_session_getter=_active_session,
        implicit_alignment=_IMPLICIT_TEMP_STORAGE_ALIGNMENT,
    )


def _temp_storage_ffi_args(primitive_name: str) -> tuple[Any, Any, Any]:
    return _provider_support.temp_storage_ffi_args(
        primitive_name,
        scope=_SCOPE,
        active_session_getter=_active_session,
        implicit_alignment=_IMPLICIT_TEMP_STORAGE_ALIGNMENT,
    )


def _append_external_link_file_attr(module: Any, path: str) -> None:
    _provider_bundle_support.append_external_link_file_attr(module, path)


def _append_link_library_attr(module: Any, path: str) -> None:
    _provider_bundle_support.append_link_library_attr(module, path)


def _remove_managed_bundle_link_options(dsl: Any) -> None:
    """Keep persistent CUTLASS compile options from relinking prior bundles."""

    managed_paths = _provider_bundle_support.managed_bundle_paths()
    if not managed_paths:
        return
    try:
        from cutlass.base_dsl.compiler import LinkLibraries

        options = dsl.compile_options.options
        try:
            option = options[LinkLibraries]
        except KeyError:
            return
        paths = [path for path in str(option.value).split(",") if path]
    except (AttributeError, ImportError, TypeError) as exc:
        raise DSLRuntimeError(
            f"{_SCOPE} provider requires a CUTLASS DSL with mutable "
            "link-library compile options. Install a compatible CUTLASS DSL "
            "runtime separately; repository qualification uses the internal "
            "nightly.",
            cause=exc,
        ) from exc

    filtered_paths = [
        path for path in paths if os.path.realpath(path) not in managed_paths
    ]
    if filtered_paths != paths:
        options[LinkLibraries] = LinkLibraries(",".join(filtered_paths))


def _remember_scalar_result_type(value: Any, value_type: type) -> Any:
    return _provider_support.remember_scalar_result_type(
        value,
        value_type,
        scope=_SCOPE,
        compile_options_getter=lambda: _get_cute_dsl().compile_options,
    )


_resolve_type = _provider_support.make_provider_type_resolver(
    scope=_SCOPE,
    root_scope=_ROOT_SCOPE,
    namespace="block",
)


def _configured_gpu_arch() -> str:
    return _provider_bundle_support.configured_gpu_arch(_get_cute_dsl)


def _resolve_nvrtc_arch() -> str:
    return _provider_bundle_support.resolve_nvrtc_arch(_SCOPE, _configured_gpu_arch)


def _resolve_nvrtc_sm_arch() -> str:
    return _provider_bundle_support.resolve_nvrtc_sm_arch(_SCOPE, _configured_gpu_arch)


def _select_bundle_format() -> str:
    return _provider_bundle_support.select_bundle_format(_SCOPE)


_CCCL_CUB_HEADERS = {
    "#include <cub/block/block_topk.cuh>": "cub/block/block_topk.cuh",
    **{f"#include <{include}>": include for include in BLOCK_ROW_REDUCE_INCLUDES},
}


def _registered_cccl_headers() -> dict[str, str]:
    headers = dict(_CCCL_CUB_HEADERS)
    headers.update(_registered_bundle_headers())
    return headers


def _compile_bundle_source(
    source: str,
    symbols: tuple[str, ...],
    *,
    initial_phase_timings_ns: Mapping[str, int] | None = None,
    resolution_started_ns: int | None = None,
) -> str:
    return _provider_bundle_support.compile_bundle_source(
        source,
        scope=_SCOPE,
        provider_dir=_PROVIDER_DIR,
        registered_headers=_registered_cccl_headers,
        select_bundle_format=_select_bundle_format,
        resolve_nvrtc_sm_arch=_resolve_nvrtc_sm_arch,
        resolve_nvrtc_arch=_resolve_nvrtc_arch,
        symbols=symbols,
        which=shutil.which,
        initial_phase_timings_ns=initial_phase_timings_ns,
        resolution_started_ns=resolution_started_ns,
    )


def _compile_bundle_source_with_layouts(
    source: str,
    probes: dict[object, _provider_support.ScratchLayoutProbe],
    symbols: tuple[str, ...],
    *,
    initial_phase_timings_ns: Mapping[str, int] | None = None,
    resolution_started_ns: int | None = None,
) -> _provider_bundle_support.BundleCompilation:
    return _provider_bundle_support.compile_bundle_source_with_layouts(
        source,
        layout_probes=(
            _provider_bundle_support.LayoutProbe(
                key=key,
                size_expression=probe.size_expression,
                alignment_expression=probe.alignment_expression,
            )
            for key, probe in probes.items()
        ),
        scope=_SCOPE,
        provider_dir=_PROVIDER_DIR,
        registered_headers=_registered_cccl_headers,
        select_bundle_format=_select_bundle_format,
        resolve_nvrtc_sm_arch=_resolve_nvrtc_sm_arch,
        resolve_nvrtc_arch=_resolve_nvrtc_arch,
        symbols=symbols,
        which=shutil.which,
        initial_phase_timings_ns=initial_phase_timings_ns,
        resolution_started_ns=resolution_started_ns,
    )


def _block_row_reduce_request(
    *,
    value_type: type,
    rows_per_block: int,
    warps_per_row: int,
) -> _BlockRowReduceRequest:
    core_spec = make_block_row_reduce_spec(
        dtype=value_type,
        rows_per_block=rows_per_block,
        warps_per_row=warps_per_row,
    )
    return _BlockRowReduceRequest(core_spec=core_spec, value_type=value_type)


def _block_row_reduce_template_arguments(
    request: _BlockRowReduceRequest,
    type_spec: _TypeSpec,
) -> str:
    rendered: list[str] = []
    for name, value in request.core_spec.specialization.ordered_template_arguments:
        if name == "T":
            if value is not request.value_type:
                raise ValueError("row_sum request dtype must match its core spec")
            rendered.append(type_spec.cpp_type)
        elif isinstance(value, int) and not isinstance(value, bool):
            rendered.append(str(value))
        else:
            raise TypeError(f"cannot render row_sum template argument {name}={value!r}")
    return ", ".join(rendered)


def _render_cub_row_sum(request: _BlockRowReduceRequest) -> list[str]:
    algorithm = request.core_spec.specialization
    type_spec = _TYPE_SPECS[request.value_type]
    template_arguments = _block_row_reduce_template_arguments(request, type_spec)
    planned_temp_bytes = _block_row_reduce_temp_storage_bytes(
        request.core_spec.geometry,
        type_spec.width_bits // 8,
    )
    planned_temp_alignment = _block_row_reduce_temp_storage_alignment(
        type_spec.width_bits // 8
    )
    return [
        (
            f"{type_spec.cpp_type} {request.symbol_name}("
            f"{type_spec.cpp_type} value, "
            "unsigned int temp_storage_smem_addr, int temp_storage_bytes, "
            "int temp_storage_auto_sync) {"
        ),
        "  (void)temp_storage_auto_sync;",
        (
            f"  using cub_row_reduce_t = cub::{algorithm.struct_name}<"
            f"{template_arguments}>;"
        ),
        f"  constexpr unsigned long long planned_temp_bytes = {planned_temp_bytes}ull;",
        f"  constexpr unsigned long long planned_temp_alignment = {planned_temp_alignment}ull;",
        "  constexpr unsigned long long required_temp_bytes =",
        "      (unsigned long long)sizeof(typename cub_row_reduce_t::TempStorage);",
        "  static_assert(planned_temp_bytes >= required_temp_bytes,",
        '                "CuTe row_sum scratch estimate is smaller than CUB TempStorage");',
        "  static_assert(planned_temp_alignment >=",
        "                (unsigned long long)alignof(typename cub_row_reduce_t::TempStorage),",
        '                "CuTe row_sum scratch alignment is weaker than CUB TempStorage");',
        "  if (temp_storage_bytes <= 0 ||",
        "      (unsigned long long)temp_storage_bytes < required_temp_bytes) {",
        '    asm volatile("trap;");',
        "  }",
        "  void* cub_storage_ptr = cuda_coop_cutlass_shared_ptr(temp_storage_smem_addr);",
        "  auto* cub_storage = reinterpret_cast<",
        "      typename cub_row_reduce_t::TempStorage*>(cub_storage_ptr);",
        f"  return cub_row_reduce_t(*cub_storage).{algorithm.method_name}(value);",
        "}",
    ]


_BLOCK_MULTI_ITEM_KINDS = frozenset(
    {
        "topk_keys",
        "topk_pair_keys",
        "topk_pair_values",
    },
)


def _block_bundle_include_lines(requests: list[Any]) -> list[str]:
    needs_cub_topk = any(
        request.kind in {"topk_keys", "topk_pair_keys", "topk_pair_values"}
        for request in requests
    )
    row_reduce_includes = tuple(
        f"#include <{include}>"
        for request in requests
        if isinstance(request, _BlockRowReduceRequest)
        for include in request.core_spec.specialization.includes
    )
    return [
        "#include <cub/block/block_topk.cuh>" if needs_cub_topk else "",
        *dict.fromkeys(row_reduce_includes),
    ]


def _render_block_bundle_request(request: Any) -> list[str]:
    lines: list[str] = []
    if request.kind == "row_sum":
        if not isinstance(request, _BlockRowReduceRequest):
            raise TypeError("row_sum bundle requests must carry a core spec")
        lines.extend(_render_cub_row_sum(request))
        return lines

    if request.kind in {"topk_keys", "topk_pair_keys", "topk_pair_values"}:
        assert request.key_type is not None
        assert request.op in {"max", "min"}
        assert request.block_threads is not None
        key_spec = _TYPE_SPECS[request.key_type]
        is_pairs = request.kind in {"topk_pair_keys", "topk_pair_values"}
        if is_pairs:
            assert request.pair_value_type is not None
            value_spec = _TYPE_SPECS[request.pair_value_type]
            value_cpp_type = value_spec.cpp_type
        else:
            value_spec = None
            value_cpp_type = "cub::NullType"
        planned_temp_bytes, planned_temp_alignment = _topk_cub_temp_storage_requirement(
            block_threads=request.block_threads,
            items_per_thread=request.items_per_thread,
            key_bytes=max(1, key_spec.width_bits // 8),
            value_bytes=(
                0 if value_spec is None else max(1, value_spec.width_bits // 8)
            ),
        )
        key_params = ", ".join(
            f"{key_spec.cpp_type} key{idx}" for idx in range(request.items_per_thread)
        )
        key_values = ", ".join(f"key{idx}" for idx in range(request.items_per_thread))
        if is_pairs:
            assert value_spec is not None
            value_params = ", ".join(
                f"{value_spec.cpp_type} value{idx}"
                for idx in range(request.items_per_thread)
            )
            value_values = ", ".join(
                f"value{idx}" for idx in range(request.items_per_thread)
            )
            params = f"{key_params}, {value_params}"
        else:
            value_params = ""
            value_values = ""
            params = key_params
        return_spec = value_spec if request.kind == "topk_pair_values" else key_spec
        method = f"{request.op}_{'pairs' if is_pairs else 'keys'}"
        full_call_args = (
            "keys, values, k, tile_items, begin_bit, end_bit"
            if is_pairs
            else "keys, k, tile_items, begin_bit, end_bit"
        )
        partial_call_args = (
            "keys, values, k, valid_items, begin_bit, end_bit"
            if is_pairs
            else "keys, k, valid_items, begin_bit, end_bit"
        )
        value_cache_defs = []
        value_cache_addr_defs = []
        value_cache_stores = []
        value_cache_required = ""
        if is_pairs:
            assert value_spec is not None
            value_cache_defs = [
                "  constexpr unsigned long long value_cache_offset =",
                "      (key_cache_offset + key_cache_bytes +",
                "       (unsigned long long)alignof(ValueT) - 1ull) &",
                "      ~((unsigned long long)alignof(ValueT) - 1ull);",
                "  constexpr unsigned long long value_cache_bytes =",
                "      (unsigned long long)tile_items *",
                "      (unsigned long long)sizeof(ValueT);",
            ]
            value_cache_addr_defs = [
                "  unsigned int value_cache_addr =",
                "      temp_storage_smem_addr + (unsigned int)value_cache_offset;",
            ]
            value_cache_stores = [
                f"    {value_spec.smem_store}(",
                "        value_cache_addr, cache_base_idx + (unsigned int)i, values[i]);",
            ]
            value_cache_required = "value_cache_offset + value_cache_bytes"
        else:
            value_cache_required = "key_cache_offset + key_cache_bytes"
        lines.extend(
            [
                (
                    f"{return_spec.cpp_type} {request.symbol_name}("
                    f"{params}, int k, int num_valid, int begin_bit, "
                    f"int end_bit, {key_spec.cpp_type} seed_key, int output_item, "
                    "unsigned int temp_storage_smem_addr, "
                    "int temp_storage_bytes, int temp_storage_auto_sync) {"
                ),
                "  (void)temp_storage_auto_sync;",
                "  if (cuda_coop_cutlass_ntid_y() != 1u ||",
                "      cuda_coop_cutlass_ntid_z() != 1u ||",
                "      cuda_coop_cutlass_group_threads() != "
                f"{request.block_threads}) {{",
                '    asm volatile("trap;");',
                "  }",
                f"  using KeyT = {key_spec.cpp_type};",
                f"  using ValueT = {value_cpp_type};",
                "  using BlockTopKT = cub::detail::block_topk<",
                f"      {key_spec.cpp_type}, {request.block_threads}, "
                f"{request.items_per_thread}, {value_cpp_type}>;",
                "  using TempStorageT = typename BlockTopKT::TempStorage;",
                "  constexpr int tile_items =",
                f"      {request.block_threads} * {request.items_per_thread};",
                "  constexpr unsigned long long key_cache_offset =",
                "      ((unsigned long long)sizeof(TempStorageT) +",
                "       (unsigned long long)alignof(KeyT) - 1ull) &",
                "      ~((unsigned long long)alignof(KeyT) - 1ull);",
                "  constexpr unsigned long long key_cache_bytes =",
                "      (unsigned long long)tile_items *",
                "      (unsigned long long)sizeof(KeyT);",
                *value_cache_defs,
                "  constexpr unsigned long long required_temp_bytes =",
                f"      {value_cache_required};",
                "  constexpr unsigned long long planned_temp_bytes =",
                f"      {planned_temp_bytes}ull;",
                "  constexpr unsigned long long planned_temp_alignment =",
                f"      {planned_temp_alignment}ull;",
                "  static_assert(planned_temp_bytes >= required_temp_bytes,",
                '                "CuTe TopK scratch estimate is smaller than CUB storage plus result caches");',
                "  static_assert(planned_temp_alignment >=",
                "                (unsigned long long)alignof(TempStorageT),",
                '                "CuTe TopK scratch alignment is weaker than CUB TempStorage");',
                "  static_assert(planned_temp_alignment >=",
                "                (unsigned long long)alignof(KeyT),",
                '                "CuTe TopK scratch alignment is weaker than the key cache");',
                *(
                    [
                        "  static_assert(planned_temp_alignment >=",
                        "                (unsigned long long)alignof(ValueT),",
                        '                "CuTe TopK scratch alignment is weaker than the value cache");',
                    ]
                    if is_pairs
                    else []
                ),
                "  if (temp_storage_bytes <= 0 ||",
                "      (unsigned long long)temp_storage_bytes < required_temp_bytes) {",
                '    asm volatile("trap;");',
                "  }",
                "  int valid_items = num_valid < 0 ? tile_items : num_valid;",
                "  if (valid_items <= 0 || valid_items > tile_items) {",
                '    asm volatile("trap;");',
                "  }",
                "  if (k <= 0 || k > valid_items) {",
                '    asm volatile("trap;");',
                "  }",
                "  int out_item = output_item;",
                "  if (out_item < 0) {",
                "    out_item = 0;",
                "  }",
                f"  if (out_item >= {request.items_per_thread}) {{",
                f"    out_item = {request.items_per_thread - 1};",
                "  }",
                "  unsigned int linear_tid = cuda_coop_cutlass_linear_tid();",
                "  unsigned int cache_base_idx =",
                f"      linear_tid * (unsigned int){request.items_per_thread};",
                "  unsigned int key_cache_addr =",
                "      temp_storage_smem_addr + (unsigned int)key_cache_offset;",
                *value_cache_addr_defs,
                "  volatile KeyT seed_anchor = seed_key;",
                "  (void)seed_anchor;",
            ]
        )
        if request.kind == "topk_pair_values":
            assert value_spec is not None
            lines.extend(
                [
                    f"  return {value_spec.smem_load}(",
                    "      value_cache_addr, cache_base_idx + (unsigned int)out_item);",
                    "}",
                ]
            )
            return lines
        lines.extend(
            [
                "  if (out_item != 0) {",
                f"    return {key_spec.smem_load}(",
                "        key_cache_addr, cache_base_idx + (unsigned int)out_item);",
                "  }",
                f"  KeyT keys[{request.items_per_thread}] = {{{key_values}}};",
            ]
        )
        if is_pairs:
            assert value_spec is not None
            lines.append(
                f"  ValueT values[{request.items_per_thread}] = {{{value_values}}};"
            )
        lines.extend(
            [
                "  TempStorageT& temp_storage = *reinterpret_cast<TempStorageT*>(",
                "      cuda_coop_cutlass_shared_ptr(temp_storage_smem_addr));",
                "  if (temp_storage_auto_sync != 0) {",
                f"    cuda_coop_cutlass_group_sync({request.block_threads});",
                "  }",
                "  BlockTopKT topk(temp_storage);",
                "  if (num_valid < 0) {",
                f"    topk.template {method}<true>({full_call_args});",
                "  } else {",
                f"    topk.template {method}<false>({partial_call_args});",
                "  }",
                "  if (temp_storage_auto_sync != 0) {",
                f"    cuda_coop_cutlass_group_sync({request.block_threads});",
                "  }",
                "#pragma unroll",
                f"  for (int i = 0; i < {request.items_per_thread}; ++i) {{",
                f"    {key_spec.smem_store}(",
                "        key_cache_addr, cache_base_idx + (unsigned int)i, keys[i]);",
                *value_cache_stores,
                "  }",
                f"  return {key_spec.smem_load}(",
                "      key_cache_addr, cache_base_idx + (unsigned int)out_item);",
                "}",
            ]
        )
        return lines

    raise ValueError(f"Unsupported request kind: {request.kind}")


def _render_bundle_source(requests: list[Any]) -> str:
    return _provider_support.render_bundle_source(
        requests,
        scope=_SCOPE,
        include_lines=_block_bundle_include_lines(requests),
        multi_item_kinds=_BLOCK_MULTI_ITEM_KINDS,
        render_local_request=_render_block_bundle_request,
    )


def _trace_finalize_hook(dsl, module, function_name) -> None:
    session = _provider_support.pop_bundle_session(dsl.compile_options)
    _remove_managed_bundle_link_options(dsl)
    del dsl, function_name
    if (
        session is None
        or not session.belongs_to_trace_module(getattr(module, "operation", module))
        or session.is_empty()
    ):
        return

    requests = session.request_list()
    events = session.deferred_temp_storage_event_list()
    probes = _provider_support.bundle_scratch_layout_probes(requests)
    missing_probe_keys = {
        event.requirement_key for event in events if event.requirement_key not in probes
    }
    if missing_probe_keys:
        raise DSLRuntimeError(
            "Deferred TempStorage events have no registered exact-layout probe: "
            f"{sorted(map(repr, missing_probe_keys))}"
        )

    resolution_started_ns = time.perf_counter_ns()
    render_started_ns = time.perf_counter_ns()
    source = _render_bundle_source(requests)
    render_duration_ns = max(0, time.perf_counter_ns() - render_started_ns)
    initial_phase_timings_ns = {"render": render_duration_ns}
    symbols = tuple(sorted(request.symbol_name for request in requests))
    if probes:
        compilation = _compile_bundle_source_with_layouts(
            source,
            probes,
            symbols,
            initial_phase_timings_ns=initial_phase_timings_ns,
            resolution_started_ns=resolution_started_ns,
        )
        bundle_path = compilation.path
        layouts = {
            key: _provider_support.ScratchLayout(
                size_in_bytes=layout.size_in_bytes,
                alignment=layout.alignment,
            )
            for key, layout in compilation.layouts.items()
        }
    else:
        bundle_path = _compile_bundle_source(
            source,
            symbols,
            initial_phase_timings_ns=initial_phase_timings_ns,
            resolution_started_ns=resolution_started_ns,
        )
        layouts = {}

    plans = _provider_support.plan_deferred_temp_storage_events(events, layouts)
    _provider_support.materialize_deferred_temp_storage_plans(plans, module)
    _append_link_library_attr(module, bundle_path)


_provider_support.register_bundle_finalizer(_trace_finalize_hook, scope=_ROOT_SCOPE)


def _register_request(request: _ShimRequest | _BlockRowReduceRequest) -> None:
    _active_session().add(request)


_as_int32_bool = _as_int32_bool_impl
_as_int32 = _as_int32_impl


def _as_valid_items_arg(value: Any) -> Any:
    return _as_valid_items_arg_impl(value, scope=_SCOPE)


_validate_radix_bit_range = _validate_radix_bit_range_impl


def _resolve_topk_type(
    value: Any,
    *,
    allowed: frozenset[type],
    feature: str,
) -> type:
    if isinstance(value, np.dtype):
        value = value.type
    candidate = value if isinstance(value, type) else type(value)
    value = _ORDINARY_TOPK_DTYPES.get(candidate, value)
    return _resolve_type(value, allowed=allowed, feature=feature)


_resolve_thread_data_value_type = (
    _provider_support.make_thread_data_value_type_resolver(
        scope=_SCOPE,
        resolve_type=_resolve_type,
    )
)

_resolve_topk_thread_data_value_type = (
    _provider_support.make_thread_data_value_type_resolver(
        scope=_SCOPE,
        resolve_type=_resolve_topk_type,
    )
)


def _resolve_thread_data_pair_types(
    *,
    key: Any,
    value: Any,
    allowed_key_types: frozenset[type],
    allowed_value_types: frozenset[type],
    feature: str,
) -> tuple[type, tuple[Any, ...], ThreadData, type, tuple[Any, ...], ThreadData]:
    return _resolve_thread_data_pair_types_impl(
        key=key,
        value=value,
        allowed_key_types=allowed_key_types,
        allowed_value_types=allowed_value_types,
        feature=feature,
        scope=_SCOPE,
        resolve_type=_resolve_type,
    )


def _resolve_topk_thread_data_pair_types(
    *,
    key: Any,
    value: Any,
    allowed_key_types: frozenset[type],
    allowed_value_types: frozenset[type],
    feature: str,
) -> tuple[type, tuple[Any, ...], ThreadData, type, tuple[Any, ...], ThreadData]:
    return _resolve_thread_data_pair_types_impl(
        key=key,
        value=value,
        allowed_key_types=allowed_key_types,
        allowed_value_types=allowed_value_types,
        feature=feature,
        scope=_SCOPE,
        resolve_type=_resolve_topk_type,
    )


def _require_single_item_thread_data(
    feature: str,
    *thread_data_args: ThreadData,
) -> None:
    _require_single_item_thread_data_impl(
        feature,
        *thread_data_args,
        scope=_SCOPE,
    )


def _provider_thread_data_unary(
    *,
    primitive_name: str,
    request_kind: str,
    value: ThreadData,
    allowed_types: frozenset[type],
    output_type: type | None = None,
    temp_storage_primitive_name: str | None = None,
) -> ThreadData:
    value_type, values = _resolve_thread_data_value_type(
        value,
        allowed=allowed_types,
        feature=primitive_name,
    )
    out_type = value_type if output_type is None else output_type
    _require_single_item_thread_data(primitive_name, value)
    request = _ShimRequest(
        kind=request_kind,
        value_type=value_type,
    )
    _register_request(request)
    temp_ptr_arg, temp_size_arg, temp_sync_arg = _temp_storage_ffi_args(
        primitive_name
        if temp_storage_primitive_name is None
        else temp_storage_primitive_name
    )
    result = ffi(
        name=request.symbol_name,
        params_types=[value_type, Uint32, Int32, Int32],
        return_type=out_type,
    )(
        values[0],
        temp_ptr_arg,
        temp_size_arg,
        temp_sync_arg,
    )
    out_dtype = (
        value.dtype if output_type is None and value.dtype is not None else out_type
    )
    return ThreadData.from_values(result, dtype=out_dtype)


def _provider_row_sum_after_launch_validation(
    *,
    value: Any,
    rows_per_block: int,
    warps_per_row: int,
) -> Any:
    """Emit row-sum after the registered provider validates the launch width.

    This terminal FFI hook deliberately has no launch metadata. Callers must go
    through ``_reduce._row_sum_provider``, which repeats the launch check after
    mandatory TempStorage planning and before reaching this function.
    """

    if isinstance(value, ThreadData):
        raise TypeError(f"{_SCOPE}.row_sum currently expects a scalar value")

    # Re-assert the terminal shim's static-geometry precondition independently
    # of its caller while preserving geometry-before-dtype diagnostics.
    geometry = normalize_block_row_reduce_geometry(
        rows_per_block=rows_per_block,
        warps_per_row=warps_per_row,
    )

    value_type = _resolve_type(
        value,
        allowed=_SCAN_REDUCE_TYPES,
        feature="row_sum",
    )
    request = _block_row_reduce_request(
        value_type=value_type,
        rows_per_block=geometry.rows_per_block,
        warps_per_row=geometry.warps_per_row,
    )
    _register_request(request)
    temp_ptr_arg, temp_size_arg, temp_sync_arg = _temp_storage_ffi_args("row_sum")
    result = ffi(
        name=request.symbol_name,
        params_types=[value_type, Uint32, Int32, Int32],
        return_type=value_type,
    )(
        value,
        temp_ptr_arg,
        temp_size_arg,
        temp_sync_arg,
    )
    return _remember_scalar_result_type(result, value_type)


def infer_topk_block_threads(kwargs: dict[str, Any]) -> int:
    try:
        block_threads = _infer_group_width(kwargs, default=None)
    except ValueError as exc:
        raise ValueError(
            f"{_SCOPE}.TopK requires launch_metadata or kernel "
            "reqntid/maxntid attributes with a positive integer thread count "
            "so the provider can instantiate CUB BlockTopK with a "
            "compile-time block size"
        ) from exc
    if block_threads <= 0:
        raise ValueError(f"{_SCOPE}.TopK block_threads must be positive")
    if block_threads > 1024:
        raise ValueError(f"{_SCOPE}.TopK block_threads must be <= 1024")
    return block_threads


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
        key_bytes=_type_size_bytes(key_type),
        value_bytes=0 if value_type is None else _type_size_bytes(value_type),
    )


def _topk_temp_storage_ffi_args(
    primitive_name: str,
    *,
    block_threads: int,
    items_per_thread: int,
    key_type: type,
    value_type: type | None = None,
) -> tuple[Any, Any, Any]:
    context = get_active_single_phase_context()
    temp_storage = context.temp_storage if context is not None else None
    if temp_storage is not None:
        return _temp_storage_ffi_args(primitive_name)

    required_size, required_alignment = _topk_temp_storage_requirement(
        block_threads=block_threads,
        items_per_thread=items_per_thread,
        key_type=key_type,
        value_type=value_type,
    )
    return _provider_support.temp_storage_ffi_args_for_size(
        required_size,
        required_alignment,
    )


def _topk_request(
    *,
    kind: str,
    select: str,
    key_type: type,
    value_type: type | None,
    items_per_thread: int,
    block_threads: int,
    num_valid: Any,
) -> _ShimRequest:
    core_spec = make_block_topk_spec(
        key_dtype=key_type,
        value_dtype=value_type,
        block_dim=(block_threads, 1, 1),
        items_per_thread=items_per_thread,
        selection=select,
        num_valid=(
            ArgumentBinding.omitted()
            if num_valid is None
            else ArgumentBinding.runtime()
        ),
        begin_bit=ArgumentBinding.runtime(),
        end_bit=ArgumentBinding.runtime(),
    )
    template_arguments = core_spec.specialization.template_arguments
    return _ShimRequest(
        kind=kind,
        op=core_spec.selection.value,
        key_type=template_arguments["KeyT"],
        pair_value_type=(
            template_arguments["ValueT"]
            if core_spec.payload is TopKPayload.PAIRS
            else None
        ),
        items_per_thread=core_spec.items_per_thread,
        block_threads=core_spec.block_dim[0],
    )


def _provider_thread_data_topk_keys(
    *,
    key: ThreadData,
    k: Any,
    num_valid: Any,
    begin_bit: Any,
    end_bit: Any | None,
    descending: bool,
    block_threads: int,
    temp_storage_primitive_name: str,
) -> ThreadData:
    key_type, key_values = _resolve_topk_thread_data_value_type(
        key,
        allowed=_TOPK_KEY_TYPES,
        feature="topk_keys",
    )
    resolved_end_bit = _validate_radix_bit_range(begin_bit, end_bit, key_type)
    request = _topk_request(
        kind="topk_keys",
        select="max" if descending else "min",
        key_type=key_type,
        value_type=None,
        items_per_thread=key.items_per_thread,
        block_threads=block_threads,
        num_valid=num_valid,
    )
    _register_request(request)
    temp_ptr_arg, temp_size_arg, temp_sync_arg = _topk_temp_storage_ffi_args(
        temp_storage_primitive_name,
        block_threads=block_threads,
        items_per_thread=key.items_per_thread,
        key_type=key_type,
    )
    params_types = [
        *([key_type] * key.items_per_thread),
        Int32,
        Int32,
        Int32,
        Int32,
        key_type,
        Int32,
        Uint32,
        Int32,
        Int32,
    ]
    args = [
        *key_values,
        _as_int32(k),
        _as_valid_items_arg(num_valid),
        _as_int32(begin_bit),
        _as_int32(resolved_end_bit),
    ]
    result_values = []
    seed_key = key_values[0]
    for item_idx in range(key.items_per_thread):
        result = ffi(
            name=request.symbol_name,
            params_types=params_types,
            return_type=key_type,
        )(
            *args,
            seed_key,
            _as_int32(item_idx),
            temp_ptr_arg,
            temp_size_arg,
            temp_sync_arg,
        )
        result_values.append(result)
        if item_idx == 0:
            seed_key = result
    out_dtype = key.dtype if key.dtype is not None else key_type
    return ThreadData.from_values(*result_values, dtype=out_dtype)


def _provider_thread_data_topk_pairs(
    *,
    key: ThreadData,
    value: ThreadData,
    k: Any,
    num_valid: Any,
    begin_bit: Any,
    end_bit: Any | None,
    descending: bool,
    block_threads: int,
    temp_storage_primitive_name: str,
) -> tuple[ThreadData, ThreadData]:
    key_type, key_values, key_td, value_type, value_values, value_td = (
        _resolve_topk_thread_data_pair_types(
            key=key,
            value=value,
            allowed_key_types=_TOPK_KEY_TYPES,
            allowed_value_types=_TOPK_VALUE_TYPES,
            feature="topk_pairs",
        )
    )
    resolved_end_bit = _validate_radix_bit_range(begin_bit, end_bit, key_type)
    select = "max" if descending else "min"
    key_request = _topk_request(
        kind="topk_pair_keys",
        select=select,
        key_type=key_type,
        value_type=value_type,
        items_per_thread=key_td.items_per_thread,
        block_threads=block_threads,
        num_valid=num_valid,
    )
    value_request = _topk_request(
        kind="topk_pair_values",
        select=select,
        key_type=key_type,
        value_type=value_type,
        items_per_thread=key_td.items_per_thread,
        block_threads=block_threads,
        num_valid=num_valid,
    )
    _register_request(key_request)
    _register_request(value_request)
    temp_ptr_arg, temp_size_arg, temp_sync_arg = _topk_temp_storage_ffi_args(
        temp_storage_primitive_name,
        block_threads=block_threads,
        items_per_thread=key_td.items_per_thread,
        key_type=key_type,
        value_type=value_type,
    )
    params_types = [
        *([key_type] * key_td.items_per_thread),
        *([value_type] * value_td.items_per_thread),
        Int32,
        Int32,
        Int32,
        Int32,
        key_type,
        Int32,
        Uint32,
        Int32,
        Int32,
    ]
    args = [
        *key_values,
        *value_values,
        _as_int32(k),
        _as_valid_items_arg(num_valid),
        _as_int32(begin_bit),
        _as_int32(resolved_end_bit),
    ]
    sorted_keys = []
    sorted_values = []
    seed_key = key_values[0]
    for item_idx in range(key_td.items_per_thread):
        call_args = [
            *args,
            seed_key,
            _as_int32(item_idx),
            temp_ptr_arg,
            temp_size_arg,
            temp_sync_arg,
        ]
        sorted_key = ffi(
            name=key_request.symbol_name,
            params_types=params_types,
            return_type=key_type,
        )(*call_args)
        sorted_keys.append(sorted_key)
        if item_idx == 0:
            seed_key = sorted_key
        value_call_args = [
            *args,
            sorted_key,
            _as_int32(item_idx),
            temp_ptr_arg,
            temp_size_arg,
            temp_sync_arg,
        ]
        sorted_values.append(
            ffi(
                name=value_request.symbol_name,
                params_types=params_types,
                return_type=value_type,
            )(*value_call_args)
        )
    out_key_dtype = key_td.dtype if key_td.dtype is not None else key_type
    out_value_dtype = value_td.dtype if value_td.dtype is not None else value_type
    return ThreadData.from_values(
        *sorted_keys, dtype=out_key_dtype
    ), ThreadData.from_values(*sorted_values, dtype=out_value_dtype)


def provider_topk_keys(
    *,
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
        return _provider_thread_data_topk_keys(
            key=key,
            k=k,
            num_valid=num_valid,
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=descending,
            block_threads=block_threads,
            temp_storage_primitive_name=temp_storage_primitive_name,
        )

    key_type = _resolve_topk_type(
        key,
        allowed=_TOPK_KEY_TYPES,
        feature="topk_keys",
    )
    resolved_end_bit = _validate_radix_bit_range(begin_bit, end_bit, key_type)
    request = _topk_request(
        kind="topk_keys",
        select="max" if descending else "min",
        key_type=key_type,
        value_type=None,
        items_per_thread=1,
        block_threads=block_threads,
        num_valid=num_valid,
    )
    _register_request(request)
    temp_ptr_arg, temp_size_arg, temp_sync_arg = _topk_temp_storage_ffi_args(
        temp_storage_primitive_name,
        block_threads=block_threads,
        items_per_thread=1,
        key_type=key_type,
    )
    result = ffi(
        name=request.symbol_name,
        params_types=[
            key_type,
            Int32,
            Int32,
            Int32,
            Int32,
            key_type,
            Int32,
            Uint32,
            Int32,
            Int32,
        ],
        return_type=key_type,
    )(
        key,
        _as_int32(k),
        _as_valid_items_arg(num_valid),
        _as_int32(begin_bit),
        _as_int32(resolved_end_bit),
        key,
        _as_int32(0),
        temp_ptr_arg,
        temp_size_arg,
        temp_sync_arg,
    )
    return _remember_scalar_result_type(result, key_type)


def provider_topk_pairs(
    *,
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
        return _provider_thread_data_topk_pairs(
            key=key,
            value=value,
            k=k,
            num_valid=num_valid,
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=descending,
            block_threads=block_threads,
            temp_storage_primitive_name=temp_storage_primitive_name,
        )

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
    resolved_end_bit = _validate_radix_bit_range(begin_bit, end_bit, key_type)
    select = "max" if descending else "min"
    key_request = _topk_request(
        kind="topk_pair_keys",
        select=select,
        key_type=key_type,
        value_type=value_type,
        items_per_thread=1,
        block_threads=block_threads,
        num_valid=num_valid,
    )
    value_request = _topk_request(
        kind="topk_pair_values",
        select=select,
        key_type=key_type,
        value_type=value_type,
        items_per_thread=1,
        block_threads=block_threads,
        num_valid=num_valid,
    )
    _register_request(key_request)
    _register_request(value_request)
    temp_ptr_arg, temp_size_arg, temp_sync_arg = _topk_temp_storage_ffi_args(
        temp_storage_primitive_name,
        block_threads=block_threads,
        items_per_thread=1,
        key_type=key_type,
        value_type=value_type,
    )
    params_types = [
        key_type,
        value_type,
        Int32,
        Int32,
        Int32,
        Int32,
        key_type,
        Int32,
        Uint32,
        Int32,
        Int32,
    ]
    args = [
        key,
        value,
        _as_int32(k),
        _as_valid_items_arg(num_valid),
        _as_int32(begin_bit),
        _as_int32(resolved_end_bit),
    ]
    sorted_key = ffi(
        name=key_request.symbol_name,
        params_types=params_types,
        return_type=key_type,
    )(
        *args,
        key,
        _as_int32(0),
        temp_ptr_arg,
        temp_size_arg,
        temp_sync_arg,
    )
    sorted_value = ffi(
        name=value_request.symbol_name,
        params_types=params_types,
        return_type=value_type,
    )(
        *args,
        sorted_key,
        _as_int32(0),
        temp_ptr_arg,
        temp_size_arg,
        temp_sync_arg,
    )
    return (
        _remember_scalar_result_type(sorted_key, key_type),
        _remember_scalar_result_type(sorted_value, value_type),
    )
