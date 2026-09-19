# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typed shared-planner providers for stable block Radix Sort and Rank."""

import hashlib
from dataclasses import dataclass, replace
from enum import Enum
from numbers import Integral

import numpy as np
from cutlass._mlir.dialects import llvm
from cutlass.base_dsl.typing import Float32, Float64, Int32, Int64, Uint32, Uint64
from cutlass.cute.ffi import ffi

from cuda.coop._core import (
    AlgorithmSpec,
    ArgumentBinding,
    GroupLoweringPlan,
    GroupLoweringTarget,
    GroupOperandKind,
    StorageOwnership,
    SynchronizationScope,
    make_group_primitive_call,
    plan_group_primitive,
)
from cuda.coop._core._types import INT32
from cuda.coop._core.api._dispatch import _common_root_operation_name
from cuda.coop._core.api.radix import _radix_bounds
from cuda.coop._core.block.radix_rank import (
    block_radix_rank_bins_per_thread,
    make_block_radix_rank_semantics,
)
from cuda.coop._core.block.radix_sort import make_block_radix_sort_semantics
from cuda.coop._core.group.radix import GroupRadixRankSemantics, GroupRadixSortSemantics

from .._compiler import _rendering, _state, _storage, _types
from .._temp_storage import TempStorage
from .._thread_data import ThreadData, _make_rmem_tensor

_SCOPE = "cuda.coop.cutlass"
_INTEGER_KEYS = frozenset({Int32, Int64, Uint32, Uint64})
_SORT_KEYS = _INTEGER_KEYS | {Float32, Float64}
_resolve_type = _types.make_provider_type_resolver(
    scope=_SCOPE, root_scope=_SCOPE, namespace="thread_group"
)


def _bit_binding(value, name):
    if isinstance(value, (bool, np.bool_, Enum)):
        raise TypeError(f"radix sort {name} must be an integer, not bool or Enum")
    if isinstance(value, Integral):
        return ArgumentBinding.static(int(value))
    spec = _types.TYPE_SPECS.get(_types.canonical_dsl_type(value))
    if spec is None or spec.token[0] not in {"i", "u"} or spec.token == "u64":
        raise TypeError(
            f"radix sort {name} requires a signed integer up to 64 bits or an unsigned integer up to 32 bits"
        )
    return ArgumentBinding.runtime()


def _plan(group, launch, operation, temp_storage=None):
    source = (
        "common_root" if _common_root_operation_name() is not None else "cutlass_root"
    )
    plan = plan_group_primitive(
        make_group_primitive_call(group, operation, source=source), launch
    ).require_supported()
    if operation.primitive.items_per_thread * launch.exact_block_threads > 65535:
        raise ValueError("CUB radix operations require at most 65535 items per block")
    return replace(
        plan,
        temp_storage=replace(
            plan.temp_storage,
            ownership=StorageOwnership.IMPLEMENTATION
            if temp_storage is None
            else StorageOwnership.CALLER,
            exact_layout_required=True,
            sharing=None if temp_storage is None else temp_storage.sharing,
            requested_size_in_bytes=None
            if temp_storage is None
            else temp_storage.size_in_bytes,
            requested_alignment=None
            if temp_storage is None
            else temp_storage.alignment,
            auto_sync=True if temp_storage is None else temp_storage.auto_sync,
        ),
        synchronization=replace(
            plan.synchronization,
            storage_reuse_barrier=SynchronizationScope.BLOCK
            if temp_storage is None or temp_storage.auto_sync
            else SynchronizationScope.NONE,
        ),
    )


def _sort_plan(
    *,
    group,
    launch,
    key_type,
    value_type,
    items,
    scalar,
    begin_bit,
    end_bit,
    descending,
    blocked_to_striped,
    temp_storage=None,
):
    primitive = make_block_radix_sort_semantics(
        key_dtype=key_type,
        value_dtype=value_type,
        items_per_thread=items,
        descending=descending,
        blocked_to_striped=blocked_to_striped,
        begin_bit=_bit_binding(begin_bit, "begin_bit"),
        end_bit=_bit_binding(end_bit, "end_bit"),
        key_bit_width=_types.TYPE_SPECS[key_type].width_bits,
        bit_policy="both",
    )
    return _plan(
        group,
        launch,
        GroupRadixSortSemantics(
            primitive, GroupOperandKind.SCALAR if scalar else GroupOperandKind.ARRAY
        ),
        temp_storage,
    )


def _rank_plan(
    *,
    group,
    launch,
    key_type,
    items,
    scalar,
    begin_bit,
    end_bit,
    descending,
    prefix_items,
):
    primitive = make_block_radix_rank_semantics(
        key_dtype=key_type,
        items_per_thread=items,
        begin_bit=begin_bit,
        end_bit=end_bit,
        key_bit_width=_types.TYPE_SPECS[key_type].width_bits,
        descending=descending,
        block_threads=launch.exact_block_threads,
        exclusive_digit_prefix_items_per_thread=prefix_items,
    )
    return _plan(
        group,
        launch,
        GroupRadixRankSemantics(
            primitive, GroupOperandKind.SCALAR if scalar else GroupOperandKind.ARRAY
        ),
    )


@dataclass(frozen=True, eq=False)
class _CubRadixRequest:
    plan: GroupLoweringPlan
    kind: str = "cub_group_radix"

    def __post_init__(self):
        self.plan.require_supported()
        if (
            self.plan.target is not GroupLoweringTarget.CUB_BLOCK
            or not isinstance(
                self.plan.call.operation,
                (GroupRadixSortSemantics, GroupRadixRankSemantics),
            )
            or not isinstance(self.plan.implementation, AlgorithmSpec)
        ):
            raise ValueError("Radix requires a shared CUB block Sort or Rank plan")
        p, spec = self.primitive, self.implementation
        if p.key_dtype not in (_INTEGER_KEYS if self.is_rank else _SORT_KEYS):
            raise TypeError("Radix key dtype is not supported")
        if (
            not self.is_rank
            and p.value_dtype is not None
            and p.value_dtype not in _types.ALL_PROVIDER_TYPES
        ):
            raise TypeError("Radix value dtype is not supported")
        expected_class = "BlockRadixRank" if self.is_rank else "CudaCoopBlockRadixSort"
        expected_method = "RankKeys" if self.is_rank else p.method_name
        if spec.struct_name != expected_class or spec.method_name != expected_method:
            raise ValueError("Radix implementation does not match its plan")
        args = spec.template_arguments
        if (
            args.get("KeyT") is not p.key_dtype
            or args.get("ITEMS_PER_THREAD") != p.items_per_thread
        ):
            raise ValueError("Radix template payload does not match its plan")
        if (
            tuple(args.get(f"BLOCK_DIM_{axis}") for axis in "XYZ")
            != self.plan.participation.exact_block_dim
        ):
            raise ValueError("Radix block dimensions do not match its plan")
        if self.is_rank:
            if (
                args.get("RADIX_BITS") != p.radix_bits
                or args.get("IS_DESCENDING") != p.order.cpp_bool
            ):
                raise ValueError("Radix Rank digit policy does not match its plan")
        elif args.get("ValueT") != (
            p.value_dtype if p.value_dtype is not None else "::cub::NullType"
        ):
            raise ValueError("Radix Sort value dtype does not match its plan")
        expected = (
            (INT32,)
            if self.is_rank
            else (
                (p.key_dtype,)
                if p.value_dtype is None
                else (p.key_dtype, p.value_dtype)
            )
        )
        if (
            self.plan.result is None
            or tuple(item.dtype for item in self.plan.result.values) != expected
        ):
            raise ValueError("Radix result dtypes do not match its plan")
        if any(
            item.items_per_member != p.items_per_thread
            or item.operand_kind is not self.plan.call.operation.operand_kind
            for item in self.plan.result.values
        ):
            raise ValueError("Radix result shape does not match its plan")
        storage, sync = self.plan.temp_storage, self.plan.synchronization
        if (
            storage is None
            or not storage.exact_layout_required
            or storage.instances != 1
            or sync is None
        ):
            raise ValueError("Radix requires one exact block scratch layout")
        if sync.storage_reuse_barrier is not (
            SynchronizationScope.BLOCK
            if storage.auto_sync
            else SynchronizationScope.NONE
        ):
            raise ValueError("Radix scratch synchronization does not match its plan")
        if self.is_rank and (
            storage.ownership is not StorageOwnership.IMPLEMENTATION
            or not storage.auto_sync
        ):
            raise ValueError("Radix Rank owns scratch with trailing synchronization")

    @property
    def is_rank(self):
        return isinstance(self.plan.call.operation, GroupRadixRankSemantics)

    @property
    def primitive(self):
        return self.plan.call.operation.primitive

    @property
    def implementation(self):
        return self.plan.implementation

    @property
    def cpp_type(self):
        values = []
        for name, value in self.implementation.ordered_template_arguments:
            if name in {"KeyT", "ValueT"} and value in _types.TYPE_SPECS:
                values.append(_types.TYPE_SPECS[value].cpp_type)
            elif isinstance(value, (int, str)) and not isinstance(value, bool):
                values.append(str(value))
            else:
                raise TypeError(f"Unsupported Radix template argument {name}")
        return f"::cub::{self.implementation.struct_name}<{', '.join(values)}>"

    @property
    def scratch_requirement_key(self):
        return "cub_radix_storage", self.cpp_type

    @property
    def symbol_name(self):
        digest = hashlib.sha256(repr(self.plan.artifact_key).encode()).hexdigest()[:16]
        return f"cuda_coop_cutlass_radix_{digest}"

    def __eq__(self, other):
        return (
            isinstance(other, _CubRadixRequest)
            and self.plan.artifact_key == other.plan.artifact_key
        )

    def __hash__(self):
        return hash(self.plan.artifact_key)


def _render_radix(request):
    request.__post_init__()
    p = request.primitive
    params, inputs, outputs = [], [], []
    payloads = [("keys", p.key_dtype)]
    if not request.is_rank and p.value_dtype is not None:
        payloads.append(("values", p.value_dtype))
    for name, dtype in payloads:
        cpp = _types.TYPE_SPECS[dtype].cpp_type
        params.extend(f"{cpp} {name}{i}" for i in range(p.items_per_thread))
        items = [f"{name}{i}" for i in range(p.items_per_thread)]
        if request.is_rank:
            unsigned = (
                "unsigned int"
                if _types.TYPE_SPECS[dtype].width_bits == 32
                else "unsigned long long"
            )
            flip = "0x80000000u" if dtype is Int32 else "0x8000000000000000ull"
            items = [
                f"static_cast<{unsigned}>({item})"
                + (f" ^ {flip}" if dtype in {Int32, Int64} else "")
                for item in items
            ]
            cpp = unsigned
        inputs.append(f"  {cpp} {name}[{p.items_per_thread}] = {{{', '.join(items)}}};")
    if request.is_rank:
        inputs.append(f"  int ranks[{p.items_per_thread}];")
        unsigned = (
            "unsigned int"
            if _types.TYPE_SPECS[p.key_dtype].width_bits == 32
            else "unsigned long long"
        )
        args = [
            "keys",
            "ranks",
            f"::cub::BFEDigitExtractor<{unsigned}>({p.bit_range.static_begin_bit}, {p.radix_bits})",
        ]
        result_names = [("ranks", "int", p.items_per_thread)]
        if p.has_exclusive_digit_prefix:
            extent = p.exclusive_digit_prefix_items_per_thread
            inputs.extend(
                (
                    f"  int prefix[{extent}];",
                    f"  for (int i = 0; i < {extent}; ++i) {{ prefix[i] = -1; }}",
                )
            )
            args.append("prefix")
            result_names.append(("prefix", "int", extent))
    else:
        params.extend(("long long begin_bit", "long long end_bit"))
        args = [name for name, _ in payloads] + ["begin_bit", "end_bit"]
        result_names = [
            (name, _types.TYPE_SPECS[dtype].cpp_type, p.items_per_thread)
            for name, dtype in payloads
        ]
    params.extend(
        ("unsigned int storage_address", "int storage_bytes", "int storage_auto_sync")
    )
    for name, cpp, extent in result_names:
        params.append(f"{cpp}* result_{name}")
        outputs.extend(f"  result_{name}[{i}] = {name}[{i}];" for i in range(extent))
    return [
        f"void {request.symbol_name}({', '.join(params)}) {{",
        f"  using implementation_type = {request.cpp_type};",
        "  using storage_type = typename implementation_type::TempStorage;",
        "  if (storage_bytes <= 0 || (unsigned long long)storage_bytes < sizeof(storage_type) ||",
        "      (storage_address & (alignof(storage_type) - 1u)) != 0u) {",
        '    asm volatile("trap;");',
        "  }",
        "  unsigned long long generic_address;",
        '  asm("cvta.shared.u64 %0, %1;" : "=l"(generic_address) : "l"((unsigned long long)storage_address));',
        "  auto& storage = *reinterpret_cast<storage_type*>(generic_address);",
        *inputs,
        f"  implementation_type(storage).{request.implementation.method_name}({', '.join(args)});",
        "  if (storage_auto_sync != 0) { __syncthreads(); }",
        *outputs,
        "}",
    ]


_rendering.register_bundle_renderer(
    "cub_group_radix",
    render=_render_radix,
    include_lines=(
        "#include <cub/block/block_radix_sort.cuh>",
        "#include <cub/block/block_radix_rank.cuh>",
        "#include <cub/util_type.cuh>",
    ),
    cccl_headers=tuple(
        (f"#include <{header}>", header)
        for header in (
            "cub/block/block_radix_sort.cuh",
            "cub/block/block_radix_rank.cuh",
            "cub/util_type.cuh",
        )
    ),
    scratch_layout_probe=lambda request: _rendering.make_scratch_layout_probe(
        request.scratch_requirement_key, f"typename {request.cpp_type}::TempStorage"
    ),
)


def _resolve_payload(value, *, allowed, feature):
    scalar = not isinstance(value, ThreadData)
    payload = ThreadData(1, values=[value]) if scalar else value
    dtype, items = _types.resolve_thread_data_value_type(
        payload,
        allowed=allowed,
        feature=feature,
        scope=_SCOPE,
        resolve_type=_resolve_type,
    )
    return payload, dtype, items, scalar


def _typed_item(value, dtype):
    if isinstance(value, np.generic):
        value = value.item()
    converted = _types.coerce_plain_scalar(
        value, dtype, name="radix item", scope=_SCOPE, allow_nonfinite=True
    )
    return dtype(value) if converted is _types._NOT_PLAIN_SCALAR else converted


def _materialize(
    request, payloads, resolved, *, bounds=(), temp_storage=None, prefix=None
):
    arguments, parameter_types = [], []
    for dtype, items in resolved:
        arguments.extend(_typed_item(item, dtype) for item in items)
        parameter_types.extend([dtype] * len(items))
    for value in bounds:
        arguments.append(Int64(value))
        parameter_types.append(Int64)
    result_types = [Int32] if request.is_rank else [dtype for dtype, _ in resolved]
    tensors = [
        _make_rmem_tensor(payload.items_per_thread, dtype, payload.alignment)
        for payload, dtype in zip(payloads, result_types)
    ]
    prefix_tensor = (
        None
        if prefix is None
        else _make_rmem_tensor(prefix.items_per_thread, Int32, prefix.alignment)
    )
    snapshot = _state.snapshot_active_session_state()
    try:
        _state.register_request(request)
        descriptor = TempStorage() if temp_storage is None else temp_storage
        arguments.extend(
            _storage.register_deferred_temp_storage_event(
                descriptor,
                primitive_name="radix_rank" if request.is_rank else "radix_sort",
                requirement_key=request.scratch_requirement_key,
            )
        )
        parameter_types.extend((Uint32, Int32, Int32))
        output_tensors = [*tensors, *([] if prefix_tensor is None else [prefix_tensor])]
        arguments.extend(tensor.iterator.llvm_ptr for tensor in output_tensors)
        parameter_types.extend([llvm.PointerType.get(0)] * len(output_tensors))
        ffi(name=request.symbol_name, params_types=parameter_types, return_type=None)(
            *arguments
        )
        results = tuple(
            ThreadData(
                payload.items_per_thread,
                dtype=dtype
                if request.is_rank
                else _types.thread_data_output_dtype(payload, dtype),
                values=[dtype(tensor[i]) for i in range(payload.items_per_thread)],
                alignment=payload.alignment,
            )
            for payload, dtype, tensor in zip(payloads, result_types, tensors)
        )
        if prefix is not None:
            values = [Int32(prefix_tensor[i]) for i in range(prefix.items_per_thread)]
            prefix.dtype = Int32
            for index, value in enumerate(values):
                prefix[index] = value
        if request.plan.call.operation.operand_kind is GroupOperandKind.SCALAR:
            results = tuple(result[0] for result in results)
        return results[0] if len(results) == 1 else results
    except BaseException:
        _state.restore_active_session_state(snapshot)
        raise


def provider_radix_sort(
    *,
    group,
    launch,
    keys,
    values,
    begin_bit,
    end_bit,
    descending,
    blocked_to_striped,
    temp_storage,
):
    keys, key_type, key_items, scalar = _resolve_payload(
        keys, allowed=_SORT_KEYS, feature="radix_sort"
    )
    payloads, resolved = [keys], [(key_type, key_items)]
    if values is not None:
        values, value_type, value_items, _ = _resolve_payload(
            values, allowed=_types.ALL_PROVIDER_TYPES, feature="radix_sort"
        )
        payloads.append(values)
        resolved.append((value_type, value_items))
    else:
        value_type = None
    end_bit = _types.TYPE_SPECS[key_type].width_bits if end_bit is None else end_bit
    plan = _sort_plan(
        group=group,
        launch=launch,
        key_type=key_type,
        value_type=value_type,
        items=keys.items_per_thread,
        scalar=scalar,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        blocked_to_striped=blocked_to_striped,
        temp_storage=temp_storage,
    )
    return _materialize(
        _CubRadixRequest(plan),
        payloads,
        resolved,
        bounds=(begin_bit, end_bit),
        temp_storage=temp_storage,
    )


def provider_radix_rank(
    *,
    group,
    launch,
    keys,
    begin_bit,
    end_bit,
    radix_bits,
    descending,
    exclusive_digit_prefix,
):
    keys, key_type, key_items, scalar = _resolve_payload(
        keys, allowed=_INTEGER_KEYS, feature="radix_rank"
    )
    begin_bit, end_bit = _radix_bounds(
        "radix_rank",
        _types.TYPE_SPECS[key_type].width_bits,
        begin_bit,
        end_bit,
        radix_bits,
    )
    prefix = exclusive_digit_prefix
    if prefix is not None:
        if not isinstance(prefix, ThreadData):
            raise TypeError(
                "radix_rank exclusive_digit_prefix must be writable ThreadData"
            )
        expected = block_radix_rank_bins_per_thread(
            end_bit - begin_bit, launch.exact_block_threads
        )
        if prefix.items_per_thread != expected:
            raise ValueError(
                f"radix_rank exclusive_digit_prefix must contain {expected} items per thread"
            )
        if (
            prefix.dtype is not None
            and _types.canonical_dsl_type(prefix.dtype) is not Int32
        ):
            raise TypeError("radix_rank exclusive_digit_prefix dtype must be Int32")
    plan = _rank_plan(
        group=group,
        launch=launch,
        key_type=key_type,
        items=keys.items_per_thread,
        scalar=scalar,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        prefix_items=None if prefix is None else prefix.items_per_thread,
    )
    return _materialize(
        _CubRadixRequest(plan), [keys], [(key_type, key_items)], prefix=prefix
    )
