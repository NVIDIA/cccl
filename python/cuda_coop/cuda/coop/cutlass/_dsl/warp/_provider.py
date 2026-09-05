# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cutlass import cute as _cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.ffi import ffi

from cuda.coop._core import CxxOperator, Dependency, Reference
from cuda.coop._core.warp import (
    WarpReduceOperation,
    WarpScanMode,
    make_warp_exchange_spec,
    make_warp_reduce_spec,
    make_warp_scan_spec,
)

from .. import _provider as _provider_support
from .._launch import resolve_threads_in_warp as _resolve_threads_in_warp
from .._scope import ROOT_SCOPE as _ROOT_SCOPE
from .._scope import WARP_SCOPE as _SCOPE
from .._thread_data import ThreadData


@dataclass(frozen=True)
class _WarpShimRequest:
    kind: str
    value_type: type
    logical_warp_threads: int
    op: str | None = None
    has_valid_items: bool = False
    has_warp_aggregate: bool = False
    items_per_thread: int = 1

    @property
    def _arity_suffix(self) -> str:
        if self.items_per_thread > 1:
            return f"_x{self.items_per_thread}"
        return ""

    @property
    def symbol_name(self) -> str:
        spec = _provider_support.TYPE_SPECS[self.value_type]
        valid_suffix = "_valid" if self.has_valid_items else ""
        aggregate_suffix = "_aggregate" if self.has_warp_aggregate else ""
        if self.op is None:
            base = (
                f"cuda_coop_cutlass_{self.kind}_{spec.token}"
                f"{valid_suffix}{aggregate_suffix}"
            )
            return f"{base}{self._arity_suffix}_w{self.logical_warp_threads}"
        base = (
            f"cuda_coop_cutlass_{self.kind}_{self.op}_{spec.token}"
            f"{valid_suffix}{aggregate_suffix}"
        )
        return f"{base}{self._arity_suffix}_w{self.logical_warp_threads}"


@dataclass(frozen=True)
class _WarpThreadDataExchangeRequest:
    kind: str
    mode: str
    value_type: type
    logical_warp_threads: int
    block_threads: int
    items_per_thread: int
    rank_type: type | None = None

    @property
    def symbol_name(self) -> str:
        value_spec = _provider_support.TYPE_SPECS[self.value_type]
        base = f"cuda_coop_cutlass_warp_exchange_{self.mode}_{value_spec.token}"
        if self.mode == "scatter_to_striped":
            assert self.rank_type is not None
            rank_spec = _provider_support.TYPE_SPECS[self.rank_type]
            base += f"_rank_{rank_spec.token}"
        return (
            f"{base}_x{self.items_per_thread}_w{self.logical_warp_threads}"
            f"_b{self.block_threads}"
        )


_WARP_REDUCE_KINDS = frozenset({"warp_reduce", "warp_sum"})
_WARP_SCAN_KINDS = frozenset(
    {
        "warp_exclusive_scan",
        "warp_exclusive_sum",
        "warp_inclusive_scan",
        "warp_inclusive_sum",
    }
)
_WARP_EXCHANGE_MODES = frozenset(
    {
        "blocked_to_striped",
        "scatter_to_striped",
        "striped_to_blocked",
    }
)
_WARP_EXCHANGE_TYPES = _provider_support.ALL_PROVIDER_TYPES


def _core_cxx_operator(op: str, *, name: str) -> CxxOperator:
    """Describe a provider operator using the shared CUB operator spelling."""

    cpp = _provider_support.cub_op_expr(op).removesuffix("{}")
    return CxxOperator(
        cpp=cpp.replace("<>", "<T>", 1),
        dtype=Dependency("T"),
        name=name,
    )


def _warp_reduce_request(
    *,
    request_kind: str,
    value_type: type,
    op: str,
    threads_in_warp: int,
    has_valid_items: bool,
    items_per_thread: int = 1,
) -> _WarpShimRequest:
    """Translate canonical scalar WarpReduce semantics into a CuTe request."""

    if request_kind not in _WARP_REDUCE_KINDS:
        raise ValueError(f"unsupported warp reduce request kind {request_kind!r}")
    if (request_kind == "warp_sum") != (op == "sum"):
        raise ValueError(
            f"warp reduce request kind {request_kind!r} does not match op {op!r}"
        )
    if request_kind == "warp_sum":
        operation = WarpReduceOperation.SUM
        reduce_operator = None
    elif op in {"min", "max"}:
        operation = WarpReduceOperation(op)
        reduce_operator = None
    else:
        operation = WarpReduceOperation.REDUCE
        reduce_operator = _core_cxx_operator(op, name="reduce_op")

    core_spec = make_warp_reduce_spec(
        dtype=value_type,
        threads_in_warp=threads_in_warp,
        operation=operation,
        reduce_operator=reduce_operator,
        valid_items=has_valid_items,
    )
    template_arguments = core_spec.specialization.template_arguments
    assert template_arguments["T"] is value_type
    canonical_kind = (
        "warp_sum" if core_spec.operation is WarpReduceOperation.SUM else "warp_reduce"
    )
    assert canonical_kind == request_kind
    if core_spec.operation in {WarpReduceOperation.MIN, WarpReduceOperation.MAX}:
        assert core_spec.operation.value == op
    return _WarpShimRequest(
        kind=request_kind,
        op=None if core_spec.operation is WarpReduceOperation.SUM else op,
        value_type=template_arguments["T"],
        logical_warp_threads=core_spec.threads_in_warp,
        has_valid_items=core_spec.has_valid_items,
        items_per_thread=items_per_thread,
    )


def _warp_scan_request(
    *,
    request_kind: str,
    value_type: type,
    op: str,
    threads_in_warp: int,
    has_valid_items: bool,
    has_warp_aggregate: bool,
    items_per_thread: int = 1,
) -> _WarpShimRequest:
    """Translate canonical scalar WarpScan semantics into a CuTe request."""

    if request_kind not in _WARP_SCAN_KINDS:
        raise ValueError(f"unsupported warp scan request kind {request_kind!r}")
    mode = (
        WarpScanMode.EXCLUSIVE
        if request_kind.startswith("warp_exclusive_")
        else WarpScanMode.INCLUSIVE
    )
    is_sum = request_kind.endswith("_sum")
    use_sum_method = is_sum and not has_valid_items
    core_spec = make_warp_scan_spec(
        dtype=value_type,
        threads_in_warp=threads_in_warp,
        mode=mode,
        scan_operator=(
            None if use_sum_method else _core_cxx_operator(op, name="scan_op")
        ),
        initial_value=(
            Reference(Dependency("T"), name="initial_value")
            if request_kind == "warp_exclusive_scan"
            else None
        ),
        valid_items=has_valid_items,
        warp_aggregate=has_warp_aggregate,
    )
    canonical_kind = f"warp_{core_spec.mode.value}_{'sum' if is_sum else 'scan'}"
    assert canonical_kind == request_kind
    template_arguments = core_spec.specialization.template_arguments
    assert template_arguments["T"] is value_type
    return _WarpShimRequest(
        kind=canonical_kind,
        op=None if is_sum else op,
        value_type=template_arguments["T"],
        logical_warp_threads=core_spec.threads_in_warp,
        has_valid_items=core_spec.has_valid_items,
        has_warp_aggregate=core_spec.has_warp_aggregate,
        items_per_thread=items_per_thread,
    )


def _warp_exchange_request(
    *,
    mode: str,
    value_type: type,
    threads_in_warp: int,
    block_threads: int,
    items_per_thread: int,
    rank_type: type | None,
) -> _WarpThreadDataExchangeRequest:
    """Translate canonical WarpExchange semantics into a CuTe request."""

    if block_threads <= 0:
        raise ValueError(f"{_SCOPE}.exchange block_threads must be positive")
    core_spec = make_warp_exchange_spec(
        dtype=value_type,
        items_per_thread=items_per_thread,
        threads_in_warp=threads_in_warp,
        mode=mode,
        # CuTe exchange always writes to a distinct ThreadData destination.
        value_form="out_of_place",
        rank_dtype=rank_type,
    )
    if block_threads % core_spec.threads_in_warp != 0:
        raise ValueError(
            f"{_SCOPE}.exchange requires block_threads to be a multiple of "
            "threads_in_warp"
        )
    template_arguments = core_spec.specialization.template_arguments
    return _WarpThreadDataExchangeRequest(
        kind="warp_thread_data_exchange",
        mode=core_spec.mode.value,
        value_type=template_arguments["T"],
        rank_type=core_spec.rank_dtype,
        logical_warp_threads=core_spec.threads_in_warp,
        block_threads=block_threads,
        items_per_thread=core_spec.items_per_thread,
    )


def _render_cub_warp_reduce(request: _WarpShimRequest) -> list[str]:
    spec = _provider_support.TYPE_SPECS[request.value_type]
    input_value = "value"
    method = None
    if request.kind == "warp_sum":
        method = "Sum"
    elif request.kind == "warp_reduce" and request.op == "min":
        method = "Min"
    elif request.kind == "warp_reduce" and request.op == "max":
        method = "Max"

    if request.items_per_thread > 1:
        item_params = ", ".join(
            f"{spec.cpp_type} item{idx}" for idx in range(request.items_per_thread)
        )
        signature = (
            f"{spec.cpp_type} {request.symbol_name}({item_params}"
            f"{', int valid_items' if request.has_valid_items else ''}) {{"
        )
        op = "sum" if request.kind == "warp_sum" else request.op
        assert op is not None
        thread_reduction_lines = [f"  {spec.cpp_type} thread_total = item0;"]
        for item_idx in range(1, request.items_per_thread):
            thread_reduction_lines.append(
                "  thread_total = "
                f"{_provider_support.reduce_op_expr(op, 'thread_total', f'item{item_idx}')};"
            )
        input_value = "thread_total"
    else:
        signature = (
            f"{spec.cpp_type} {request.symbol_name}({spec.cpp_type} value"
            f"{', int valid_items' if request.has_valid_items else ''}) {{"
        )
        thread_reduction_lines = []

    lines = [
        signature,
        (
            "  using cub_warp_reduce_t = cub::WarpReduce<"
            f"{spec.cpp_type}, {request.logical_warp_threads}>;"
        ),
        "  __shared__ typename cub_warp_reduce_t::TempStorage temp_storage[",
        f"      (1024 + {request.logical_warp_threads} - 1) / "
        f"{request.logical_warp_threads}];",
        "  unsigned int logical_warp_id =",
        f"      cuda_coop_cutlass_linear_tid() / {request.logical_warp_threads}u;",
    ]
    lines.extend(thread_reduction_lines)
    if method is not None:
        valid_arg = ", valid_items" if request.has_valid_items else ""
        lines.append(
            "  return cub_warp_reduce_t(temp_storage[logical_warp_id])."
            f"{method}({input_value}{valid_arg});"
        )
    else:
        assert request.op is not None
        valid_arg = ", valid_items" if request.has_valid_items else ""
        lines.append(
            "  return cub_warp_reduce_t(temp_storage[logical_warp_id]).Reduce("
            f"{input_value}, {_provider_support.cub_op_expr(request.op)}{valid_arg});"
        )
    lines.append("}")
    return lines


def _render_cub_warp_scan(request: _WarpShimRequest) -> list[str]:
    spec = _provider_support.TYPE_SPECS[request.value_type]
    if request.items_per_thread > 1:
        item_params = ", ".join(
            f"{spec.cpp_type} item{idx}" for idx in range(request.items_per_thread)
        )
        initial_param = (
            f", {spec.cpp_type} initial"
            if request.kind == "warp_exclusive_scan"
            else ""
        )
        valid_param = ", int valid_items" if request.has_valid_items else ""
        aggregate_param = (
            f"{spec.cpp_type}* warp_aggregate" if request.has_warp_aggregate else ""
        )
        output_params = ", ".join(
            param
            for param in (
                aggregate_param,
                f"{spec.cpp_type}* result_items",
            )
            if param
        )
        op = "sum" if request.kind.endswith("_sum") else request.op
        assert op is not None
        scan_expr = _provider_support.reduce_op_expr(op, "local_total", "peer")
        thread_total_lines = [f"  {spec.cpp_type} thread_total = item0;"]
        for item_idx in range(1, request.items_per_thread):
            thread_total_lines.append(
                "  thread_total = "
                f"{_provider_support.reduce_op_expr(op, 'thread_total', f'item{item_idx}')};"
            )

        lines = [
            (
                f"void {request.symbol_name}({item_params}{initial_param}"
                f"{valid_param}, {output_params}) {{"
            ),
            (
                "  using cub_warp_scan_t = cub::WarpScan<"
                f"{spec.cpp_type}, {request.logical_warp_threads}>;"
            ),
            "  __shared__ typename cub_warp_scan_t::TempStorage temp_storage[",
            f"      (1024 + {request.logical_warp_threads} - 1) / "
            f"{request.logical_warp_threads}];",
            "  unsigned int linear_tid = cuda_coop_cutlass_linear_tid();",
            "  unsigned int logical_warp_id =",
            f"      linear_tid / {request.logical_warp_threads}u;",
            "  unsigned int lane_in_logical_warp =",
            f"      linear_tid - logical_warp_id * {request.logical_warp_threads}u;",
            *thread_total_lines,
            f"  {spec.cpp_type} lane_prefix;",
        ]
        if request.has_warp_aggregate:
            lines.append(f"  {spec.cpp_type} aggregate;")

        valid_arg = ", valid_items" if request.has_valid_items else ""
        aggregate_arg = ", aggregate" if request.has_warp_aggregate else ""
        if request.kind == "warp_exclusive_sum":
            if request.has_valid_items:
                lines.append(
                    "  cub_warp_scan_t(temp_storage[logical_warp_id])."
                    "ExclusiveScanPartial(thread_total, lane_prefix, "
                    f"{spec.zero_literal}, {_provider_support.cub_op_expr('sum')}"
                    f"{valid_arg}{aggregate_arg});"
                )
            else:
                lines.append(
                    "  cub_warp_scan_t(temp_storage[logical_warp_id])."
                    f"ExclusiveSum(thread_total, lane_prefix{aggregate_arg});"
                )
        elif request.kind == "warp_inclusive_sum":
            if request.has_valid_items:
                lines.append(
                    "  cub_warp_scan_t(temp_storage[logical_warp_id])."
                    "ExclusiveScanPartial(thread_total, lane_prefix, "
                    f"{spec.zero_literal}, {_provider_support.cub_op_expr('sum')}"
                    f"{valid_arg}{aggregate_arg});"
                )
            else:
                lines.append(
                    "  cub_warp_scan_t(temp_storage[logical_warp_id])."
                    f"ExclusiveSum(thread_total, lane_prefix{aggregate_arg});"
                )
        elif request.kind == "warp_exclusive_scan":
            assert request.op is not None
            method = (
                "ExclusiveScanPartial" if request.has_valid_items else "ExclusiveScan"
            )
            lines.append(
                f"  cub_warp_scan_t(temp_storage[logical_warp_id]).{method}("
                f"thread_total, lane_prefix, initial, "
                f"{_provider_support.cub_op_expr(request.op)}"
                f"{valid_arg}{aggregate_arg});"
            )
        elif request.kind == "warp_inclusive_scan":
            assert request.op is not None
            method = (
                "ExclusiveScanPartial" if request.has_valid_items else "ExclusiveScan"
            )
            lines.append(
                f"  cub_warp_scan_t(temp_storage[logical_warp_id]).{method}("
                f"thread_total, lane_prefix, "
                f"{_provider_support.cub_op_expr(request.op)}"
                f"{valid_arg}{aggregate_arg});"
            )
        else:
            raise NotImplementedError(f"Unsupported CUB warp scan kind: {request.kind}")

        if request.has_warp_aggregate:
            lines.append("  *warp_aggregate = aggregate;")
        if request.kind.startswith("warp_exclusive"):
            lines.append(f"  {spec.cpp_type} local_total = lane_prefix;")
            for item_idx in range(request.items_per_thread):
                lines.extend(
                    [
                        f"  result_items[{item_idx}] = local_total;",
                        f"  {{ {spec.cpp_type} peer = item{item_idx};",
                        f"    local_total = {scan_expr}; }}",
                    ]
                )
            lines.append("}")
            return lines

        if request.kind == "warp_inclusive_sum":
            lines.extend(
                [
                    f"  {spec.cpp_type} local_total = lane_prefix;",
                    "  local_total = local_total + item0;",
                    "  result_items[0] = local_total;",
                ]
            )
        else:
            lines.extend(
                [
                    "  int has_prior_lane = lane_in_logical_warp != 0u;",
                    f"  {spec.cpp_type} local_total = item0;",
                    "  if (has_prior_lane) {",
                    f"    {spec.cpp_type} peer = item0;",
                    f"    local_total = {scan_expr};",
                    "  }",
                    "  result_items[0] = local_total;",
                ]
            )
        for item_idx in range(1, request.items_per_thread):
            lines.extend(
                [
                    f"  {{ {spec.cpp_type} peer = item{item_idx};",
                    f"    local_total = {scan_expr};",
                    f"    result_items[{item_idx}] = local_total; }}",
                ]
            )
        lines.append("}")
        return lines

    lines = [
        (
            f"{spec.cpp_type} {request.symbol_name}("
            f"{spec.cpp_type} value"
            + (
                f", {spec.cpp_type} initial"
                if request.kind == "warp_exclusive_scan"
                else ""
            )
            + (", int valid_items" if request.has_valid_items else "")
            + (
                f", {spec.cpp_type}* warp_aggregate"
                if request.has_warp_aggregate
                else ""
            )
            + ") {"
        ),
        (
            "  using cub_warp_scan_t = cub::WarpScan<"
            f"{spec.cpp_type}, {request.logical_warp_threads}>;"
        ),
        "  __shared__ typename cub_warp_scan_t::TempStorage temp_storage[",
        f"      (1024 + {request.logical_warp_threads} - 1) / "
        f"{request.logical_warp_threads}];",
        "  unsigned int logical_warp_id =",
        f"      cuda_coop_cutlass_linear_tid() / {request.logical_warp_threads}u;",
        f"  {spec.cpp_type} result;",
    ]
    if request.has_warp_aggregate:
        lines.append(f"  {spec.cpp_type} aggregate;")
    valid_arg = ", valid_items" if request.has_valid_items else ""
    aggregate_arg = ", aggregate" if request.has_warp_aggregate else ""
    if request.kind == "warp_exclusive_sum":
        if request.has_valid_items:
            lines.append(
                "  cub_warp_scan_t(temp_storage[logical_warp_id])."
                "ExclusiveScanPartial(value, result, "
                f"{spec.zero_literal}, {_provider_support.cub_op_expr('sum')}"
                f"{valid_arg}{aggregate_arg});"
            )
        else:
            lines.append(
                "  cub_warp_scan_t(temp_storage[logical_warp_id])."
                f"ExclusiveSum(value, result{aggregate_arg});"
            )
    elif request.kind == "warp_inclusive_sum":
        if request.has_valid_items:
            lines.append(
                "  cub_warp_scan_t(temp_storage[logical_warp_id])."
                "InclusiveScanPartial(value, result, "
                f"{_provider_support.cub_op_expr('sum')}"
                f"{valid_arg}{aggregate_arg});"
            )
        else:
            lines.append(
                "  cub_warp_scan_t(temp_storage[logical_warp_id])."
                f"InclusiveSum(value, result{aggregate_arg});"
            )
    elif request.kind == "warp_exclusive_scan":
        assert request.op is not None
        method = "ExclusiveScanPartial" if request.has_valid_items else "ExclusiveScan"
        lines.append(
            f"  cub_warp_scan_t(temp_storage[logical_warp_id]).{method}("
            f"value, result, initial, {_provider_support.cub_op_expr(request.op)}"
            f"{valid_arg}{aggregate_arg});"
        )
    elif request.kind == "warp_inclusive_scan":
        assert request.op is not None
        method = "InclusiveScanPartial" if request.has_valid_items else "InclusiveScan"
        lines.append(
            f"  cub_warp_scan_t(temp_storage[logical_warp_id]).{method}("
            f"value, result, {_provider_support.cub_op_expr(request.op)}"
            f"{valid_arg}{aggregate_arg});"
        )
    else:
        raise NotImplementedError(f"Unsupported CUB warp scan kind: {request.kind}")
    if request.has_warp_aggregate:
        lines.append("  *warp_aggregate = aggregate;")
    lines.extend(["  return result;", "}"])
    return lines


def _render_cub_warp_exchange(request: _WarpThreadDataExchangeRequest) -> list[str]:
    spec = _provider_support.TYPE_SPECS[request.value_type]
    item_count = request.items_per_thread
    item_params = ", ".join(f"{spec.cpp_type} item{idx}" for idx in range(item_count))
    rank_params = ""
    if request.mode == "scatter_to_striped":
        assert request.rank_type is not None
        rank_spec = _provider_support.TYPE_SPECS[request.rank_type]
        rank_params = ", " + ", ".join(
            f"{rank_spec.cpp_type} rank{idx}" for idx in range(item_count)
        )

    lines = [
        (
            f"void {request.symbol_name}("
            f"{item_params}{rank_params}, {spec.cpp_type}* result_items) {{"
        ),
        (
            "  using cub_warp_exchange_t = cub::WarpExchange<"
            f"{spec.cpp_type}, {item_count}, {request.logical_warp_threads}>;"
        ),
        "  __shared__ typename cub_warp_exchange_t::TempStorage temp_storage[",
        f"      {request.block_threads // request.logical_warp_threads}];",
        "  unsigned int logical_warp_id =",
        f"      cuda_coop_cutlass_linear_tid() / {request.logical_warp_threads}u;",
        f"  {spec.cpp_type} input_items[{item_count}] = {{",
        "      " + ", ".join(f"item{idx}" for idx in range(item_count)),
        "  };",
        f"  {spec.cpp_type} output_items[{item_count}];",
    ]
    if request.mode == "striped_to_blocked":
        lines.append(
            "  cub_warp_exchange_t(temp_storage[logical_warp_id])."
            "StripedToBlocked(input_items, output_items);"
        )
    elif request.mode == "blocked_to_striped":
        lines.append(
            "  cub_warp_exchange_t(temp_storage[logical_warp_id])."
            "BlockedToStriped(input_items, output_items);"
        )
    elif request.mode == "scatter_to_striped":
        assert request.rank_type is not None
        rank_spec = _provider_support.TYPE_SPECS[request.rank_type]
        lines.extend(
            [
                f"  {rank_spec.cpp_type} ranks[{item_count}] = {{",
                "      " + ", ".join(f"rank{idx}" for idx in range(item_count)),
                "  };",
                (
                    "  cub_warp_exchange_t(temp_storage[logical_warp_id])."
                    "ScatterToStriped(input_items, output_items, ranks);"
                ),
            ]
        )
    else:
        raise NotImplementedError(f"Unsupported CUB warp exchange mode: {request.mode}")
    lines.extend(
        [
            "  __syncwarp(cuda_coop_cutlass_active_mask());",
            f"  #pragma unroll {item_count}",
            f"  for (int i = 0; i < {item_count}; ++i) {{",
            "    result_items[i] = output_items[i];",
            "  }",
            "}",
        ]
    )
    return lines


def _register_renderers() -> None:
    for kind in _WARP_REDUCE_KINDS:
        _provider_support.register_bundle_renderer(
            kind,
            render=_render_cub_warp_reduce,
            include_lines=(
                "#include <cub/warp/warp_reduce.cuh>",
                "#include <cuda/__functional/maximum.h>",
                "#include <cuda/__functional/minimum.h>",
            ),
            cccl_headers=(
                ("#include <cub/warp/warp_reduce.cuh>", "cub/warp/warp_reduce.cuh"),
            ),
        )
    for kind in _WARP_SCAN_KINDS:
        _provider_support.register_bundle_renderer(
            kind,
            render=_render_cub_warp_scan,
            include_lines=(
                "#include <cub/warp/warp_scan.cuh>",
                "#include <cuda/__functional/maximum.h>",
                "#include <cuda/__functional/minimum.h>",
            ),
            cccl_headers=(
                ("#include <cub/warp/warp_scan.cuh>", "cub/warp/warp_scan.cuh"),
            ),
        )
    _provider_support.register_bundle_renderer(
        "warp_thread_data_exchange",
        render=_render_cub_warp_exchange,
        include_lines=("#include <cub/warp/warp_exchange.cuh>",),
        cccl_headers=(
            ("#include <cub/warp/warp_exchange.cuh>", "cub/warp/warp_exchange.cuh"),
        ),
    )


_resolve_type = _provider_support.make_provider_type_resolver(
    scope=_SCOPE,
    root_scope=_ROOT_SCOPE,
    namespace="warp",
)
_resolve_scalar_type = _provider_support.make_scalar_type_resolver(
    scope=_SCOPE,
    resolve_type=_resolve_type,
)
_resolve_thread_data_value_type = (
    _provider_support.make_thread_data_value_type_resolver(
        scope=_SCOPE,
        resolve_type=_resolve_type,
    )
)


def _validate_thread_data_output(
    *,
    output: Any,
    value: ThreadData,
    value_type: type,
) -> ThreadData | None:
    return _provider_support.validate_thread_data_output(
        output=output,
        expected_items_per_thread=value.items_per_thread,
        resolved_dtype=value_type,
        scope=_SCOPE,
        primitive_name="exchange",
        output_name="output",
        resolve_type=_resolve_type,
        assigned_dtype=_provider_support.thread_data_output_dtype(value, value_type),
        type_label=f"{_ROOT_SCOPE}.ThreadData",
        item_count_message=(
            f"{_SCOPE}.exchange output must have matching items_per_thread"
        ),
    )


def _validate_warp_aggregate_output(
    *,
    output: Any,
    value_type: type,
    primitive_name: str,
) -> ThreadData | None:
    return _provider_support.validate_thread_data_output(
        output=output,
        expected_items_per_thread=1,
        resolved_dtype=value_type,
        scope=_SCOPE,
        primitive_name=primitive_name,
        output_name="warp_aggregate",
        resolve_type=_resolve_type,
        type_label=f"{_ROOT_SCOPE}.ThreadData",
    )


def _make_warp_aggregate_tensor(
    warp_aggregate: ThreadData | None,
    value_type: type,
) -> Any | None:
    if warp_aggregate is None:
        return None
    return _cute.make_rmem_tensor(1, value_type)


def _populate_warp_aggregate(
    warp_aggregate: ThreadData | None,
    aggregate_tensor: Any | None,
) -> None:
    if warp_aggregate is None:
        return
    assert aggregate_tensor is not None
    warp_aggregate[0] = aggregate_tensor[0]


def provider_exchange(
    *,
    value: Any,
    mode: str,
    output: Any = None,
    ranks: Any = None,
    block_threads: int,
    threads_in_warp: int = 32,
) -> ThreadData:
    if not isinstance(value, ThreadData):
        raise TypeError(f"{_SCOPE}.exchange value must be ThreadData")
    if mode not in _WARP_EXCHANGE_MODES:
        raise ValueError(
            f"{_SCOPE}.exchange mode must be a supported warp-exchange mode"
        )
    threads_in_warp = _resolve_threads_in_warp(
        _SCOPE,
        "exchange",
        threads_in_warp,
    )
    if value.items_per_thread > 4:
        raise NotImplementedError(
            f"{_SCOPE}.exchange currently supports at most 4 items per lane"
        )
    value_type, values = _resolve_thread_data_value_type(
        value,
        allowed=_WARP_EXCHANGE_TYPES,
        feature="exchange",
    )
    out_dtype = value.dtype if value.dtype is not None else value_type
    output_td = _validate_thread_data_output(
        output=output,
        value=value,
        value_type=value_type,
    )

    rank_type = None
    rank_values: tuple[Any, ...] = ()
    if mode == "scatter_to_striped":
        if ranks is None:
            raise ValueError(f"{_SCOPE}.exchange scatter_to_striped requires ranks")
        if not isinstance(ranks, ThreadData):
            raise TypeError(f"{_SCOPE}.exchange ranks must be ThreadData")
        if ranks.items_per_thread != value.items_per_thread:
            raise ValueError(
                f"{_SCOPE}.exchange ranks must have matching items_per_thread"
            )
        rank_type, rank_values = _resolve_thread_data_value_type(
            ranks,
            allowed=frozenset({_provider_support.Int32}),
            feature="exchange",
        )
    elif ranks is not None:
        raise ValueError(
            f"{_SCOPE}.exchange ranks are only valid for scatter_to_striped"
        )

    request = _warp_exchange_request(
        mode=mode,
        value_type=value_type,
        rank_type=rank_type,
        threads_in_warp=threads_in_warp,
        block_threads=block_threads,
        items_per_thread=value.items_per_thread,
    )
    _provider_support.register_request(request)
    params_types = [
        *([value_type] * value.items_per_thread),
        *([rank_type] * value.items_per_thread if rank_type is not None else []),
        llvm.PointerType.get(0),
    ]
    ffi_args = [*values, *rank_values]
    result_tensor = _cute.make_rmem_tensor(value.items_per_thread, value_type)
    ffi(
        name=request.symbol_name,
        params_types=params_types,
        return_type=None,
    )(
        *ffi_args,
        result_tensor.iterator.llvm_ptr,
    )
    result_values = [
        result_tensor[item_idx] for item_idx in range(value.items_per_thread)
    ]

    if output_td is None:
        return ThreadData.from_values(*result_values, dtype=out_dtype)
    for item_idx, item_value in enumerate(result_values):
        output_td[item_idx] = item_value
    return output_td


def _provider_thread_data_reduce(
    *,
    primitive_name: str,
    request_kind: str,
    value: ThreadData,
    op: str,
    threads_in_warp: int,
    valid_items: Any = None,
) -> ThreadData:
    threads_in_warp = _resolve_threads_in_warp(
        _SCOPE,
        primitive_name,
        threads_in_warp,
    )
    value_type, values = _resolve_thread_data_value_type(
        value,
        allowed=_provider_support.SCAN_REDUCE_TYPES,
        feature=primitive_name,
    )
    _provider_support.validate_scan_reduce_op_for_type(
        op,
        value_type,
        root_scope=_ROOT_SCOPE,
        feature="reduce" if primitive_name != "sum" else "sum",
        namespace="warp",
    )
    request = _warp_reduce_request(
        request_kind=request_kind,
        value_type=value_type,
        op=op,
        threads_in_warp=threads_in_warp,
        has_valid_items=valid_items is not None,
        items_per_thread=value.items_per_thread,
    )
    _provider_support.register_request(request)
    valid_args = (
        [_provider_support.as_valid_items_arg(valid_items, scope=_SCOPE)]
        if valid_items is not None
        else []
    )
    result = ffi(
        name=request.symbol_name,
        params_types=[
            *([value_type] * value.items_per_thread),
            *([_provider_support.Int32] if valid_items is not None else []),
        ],
        return_type=value_type,
    )(
        *values,
        *valid_args,
    )
    return ThreadData.from_values(
        result,
        dtype=_provider_support.thread_data_output_dtype(value, value_type),
    )


def _provider_thread_data_scan(
    *,
    primitive_name: str,
    request_kind: str,
    value: ThreadData,
    op: str,
    initial_value: Any = None,
    threads_in_warp: int,
    valid_items: Any = None,
    warp_aggregate: Any = None,
) -> ThreadData:
    threads_in_warp = _resolve_threads_in_warp(
        _SCOPE,
        primitive_name,
        threads_in_warp,
    )
    value_type, values = _resolve_thread_data_value_type(
        value,
        allowed=_provider_support.SCAN_REDUCE_TYPES,
        feature=primitive_name,
    )
    _provider_support.validate_scan_reduce_op_for_type(
        op,
        value_type,
        root_scope=_ROOT_SCOPE,
        feature=primitive_name,
        namespace="warp",
    )
    warp_aggregate_td = _validate_warp_aggregate_output(
        output=warp_aggregate,
        value_type=value_type,
        primitive_name=primitive_name,
    )
    request = _warp_scan_request(
        request_kind=request_kind,
        value_type=value_type,
        op=op,
        threads_in_warp=threads_in_warp,
        has_valid_items=valid_items is not None,
        has_warp_aggregate=warp_aggregate_td is not None,
        items_per_thread=value.items_per_thread,
    )
    _provider_support.register_request(request)
    initial_args = []
    if request_kind == "warp_exclusive_scan":
        initial_args.append(
            _provider_support.coerce_scan_initial_value(
                initial_value=initial_value,
                value_type=value_type,
                root_scope=_ROOT_SCOPE,
                feature=primitive_name,
                namespace="warp",
            )
        )
    valid_args = (
        [_provider_support.as_valid_items_arg(valid_items, scope=_SCOPE)]
        if valid_items is not None
        else []
    )
    aggregate_tensor = _make_warp_aggregate_tensor(warp_aggregate_td, value_type)
    aggregate_args = [aggregate_tensor.iterator] if aggregate_tensor is not None else []
    result_tensor = _cute.make_rmem_tensor(value.items_per_thread, value_type)
    ffi(
        name=request.symbol_name,
        params_types=[
            *([value_type] * value.items_per_thread),
            *([value_type] if request_kind == "warp_exclusive_scan" else []),
            *([_provider_support.Int32] if valid_items is not None else []),
            *([llvm.PointerType.get(0)] if aggregate_tensor is not None else []),
            llvm.PointerType.get(0),
        ],
        return_type=None,
    )(
        *values,
        *initial_args,
        *valid_args,
        *aggregate_args,
        result_tensor.iterator,
    )
    _populate_warp_aggregate(warp_aggregate_td, aggregate_tensor)
    return ThreadData.from_values(
        *(result_tensor[item_idx] for item_idx in range(value.items_per_thread)),
        dtype=_provider_support.thread_data_output_dtype(value, value_type),
    )


def provider_sum(
    *,
    value: Any,
    threads_in_warp: int = 32,
    valid_items: Any = None,
) -> Any:
    if isinstance(value, ThreadData):
        return _provider_thread_data_reduce(
            primitive_name="sum",
            request_kind="warp_sum",
            value=value,
            op="sum",
            threads_in_warp=threads_in_warp,
            valid_items=valid_items,
        )

    threads_in_warp = _resolve_threads_in_warp(
        _SCOPE,
        "sum",
        threads_in_warp,
    )
    value_type = _resolve_scalar_type(value, feature="sum")
    request = _warp_reduce_request(
        request_kind="warp_sum",
        value_type=value_type,
        op="sum",
        threads_in_warp=threads_in_warp,
        has_valid_items=valid_items is not None,
    )
    _provider_support.register_request(request)
    valid_args = (
        [_provider_support.as_valid_items_arg(valid_items, scope=_SCOPE)]
        if valid_items is not None
        else []
    )
    result = ffi(
        name=request.symbol_name,
        params_types=[
            value_type,
            *([_provider_support.Int32] if valid_items is not None else []),
        ],
        return_type=value_type,
    )(
        value,
        *valid_args,
    )
    return _provider_support.remember_scalar_result_type(
        result,
        value_type,
        scope=_SCOPE,
    )


def provider_reduce(
    *,
    value: Any,
    op: str = "sum",
    threads_in_warp: int = 32,
    valid_items: Any = None,
) -> Any:
    if op == "sum":
        return provider_sum(
            value=value,
            threads_in_warp=threads_in_warp,
            valid_items=valid_items,
        )
    if isinstance(value, ThreadData):
        return _provider_thread_data_reduce(
            primitive_name="reduce",
            request_kind="warp_reduce",
            value=value,
            op=op,
            threads_in_warp=threads_in_warp,
            valid_items=valid_items,
        )

    threads_in_warp = _resolve_threads_in_warp(
        _SCOPE,
        "reduce",
        threads_in_warp,
    )
    value_type = _resolve_scalar_type(value, feature="reduce")
    _provider_support.validate_scan_reduce_op_for_type(
        op,
        value_type,
        root_scope=_ROOT_SCOPE,
        feature="reduce",
        namespace="warp",
    )
    request = _warp_reduce_request(
        request_kind="warp_reduce",
        value_type=value_type,
        op=op,
        threads_in_warp=threads_in_warp,
        has_valid_items=valid_items is not None,
    )
    _provider_support.register_request(request)
    valid_args = (
        [_provider_support.as_valid_items_arg(valid_items, scope=_SCOPE)]
        if valid_items is not None
        else []
    )
    result = ffi(
        name=request.symbol_name,
        params_types=[
            value_type,
            *([_provider_support.Int32] if valid_items is not None else []),
        ],
        return_type=value_type,
    )(
        value,
        *valid_args,
    )
    return _provider_support.remember_scalar_result_type(
        result,
        value_type,
        scope=_SCOPE,
    )


def provider_exclusive_sum(
    *,
    value: Any,
    threads_in_warp: int = 32,
    valid_items: Any = None,
    warp_aggregate: Any = None,
) -> Any:
    if isinstance(value, ThreadData):
        return _provider_thread_data_scan(
            primitive_name="exclusive_sum",
            request_kind="warp_exclusive_sum",
            value=value,
            op="sum",
            threads_in_warp=threads_in_warp,
            valid_items=valid_items,
            warp_aggregate=warp_aggregate,
        )

    threads_in_warp = _resolve_threads_in_warp(
        _SCOPE,
        "exclusive_sum",
        threads_in_warp,
    )
    value_type = _resolve_scalar_type(value, feature="exclusive_sum")
    warp_aggregate_td = _validate_warp_aggregate_output(
        output=warp_aggregate,
        value_type=value_type,
        primitive_name="exclusive_sum",
    )
    request = _warp_scan_request(
        request_kind="warp_exclusive_sum",
        value_type=value_type,
        op="sum",
        threads_in_warp=threads_in_warp,
        has_valid_items=valid_items is not None,
        has_warp_aggregate=warp_aggregate_td is not None,
    )
    _provider_support.register_request(request)
    valid_args = (
        [_provider_support.as_valid_items_arg(valid_items, scope=_SCOPE)]
        if valid_items is not None
        else []
    )
    aggregate_tensor = _make_warp_aggregate_tensor(warp_aggregate_td, value_type)
    aggregate_args = [aggregate_tensor.iterator] if aggregate_tensor is not None else []
    result = ffi(
        name=request.symbol_name,
        params_types=[
            value_type,
            *([_provider_support.Int32] if valid_items is not None else []),
            *([llvm.PointerType.get(0)] if aggregate_tensor is not None else []),
        ],
        return_type=value_type,
    )(
        value,
        *valid_args,
        *aggregate_args,
    )
    _populate_warp_aggregate(warp_aggregate_td, aggregate_tensor)
    return _provider_support.remember_scalar_result_type(
        result,
        value_type,
        scope=_SCOPE,
    )


def provider_exclusive_scan(
    *,
    value: Any,
    op: str = "sum",
    initial_value: Any = None,
    threads_in_warp: int = 32,
    valid_items: Any = None,
    warp_aggregate: Any = None,
) -> Any:
    if op == "sum" and initial_value is None:
        return provider_exclusive_sum(
            value=value,
            threads_in_warp=threads_in_warp,
            valid_items=valid_items,
            warp_aggregate=warp_aggregate,
        )
    if initial_value is None:
        raise ValueError(
            f"{_SCOPE}.exclusive_scan requires initial_value for non-default scans"
        )
    if isinstance(value, ThreadData):
        return _provider_thread_data_scan(
            primitive_name="exclusive_scan",
            request_kind="warp_exclusive_scan",
            value=value,
            op=op,
            initial_value=initial_value,
            threads_in_warp=threads_in_warp,
            valid_items=valid_items,
            warp_aggregate=warp_aggregate,
        )

    threads_in_warp = _resolve_threads_in_warp(
        _SCOPE,
        "exclusive_scan",
        threads_in_warp,
    )
    value_type = _resolve_scalar_type(value, feature="exclusive_scan")
    _provider_support.validate_scan_reduce_op_for_type(
        op,
        value_type,
        root_scope=_ROOT_SCOPE,
        feature="exclusive_scan",
        namespace="warp",
    )
    warp_aggregate_td = _validate_warp_aggregate_output(
        output=warp_aggregate,
        value_type=value_type,
        primitive_name="exclusive_scan",
    )
    request = _warp_scan_request(
        request_kind="warp_exclusive_scan",
        value_type=value_type,
        op=op,
        threads_in_warp=threads_in_warp,
        has_valid_items=valid_items is not None,
        has_warp_aggregate=warp_aggregate_td is not None,
    )
    _provider_support.register_request(request)
    valid_args = (
        [_provider_support.as_valid_items_arg(valid_items, scope=_SCOPE)]
        if valid_items is not None
        else []
    )
    aggregate_tensor = _make_warp_aggregate_tensor(warp_aggregate_td, value_type)
    aggregate_args = [aggregate_tensor.iterator] if aggregate_tensor is not None else []
    result = ffi(
        name=request.symbol_name,
        params_types=[
            value_type,
            value_type,
            *([_provider_support.Int32] if valid_items is not None else []),
            *([llvm.PointerType.get(0)] if aggregate_tensor is not None else []),
        ],
        return_type=value_type,
    )(
        value,
        _provider_support.coerce_scan_initial_value(
            initial_value=initial_value,
            value_type=value_type,
            root_scope=_ROOT_SCOPE,
            feature="exclusive_scan",
            namespace="warp",
        ),
        *valid_args,
        *aggregate_args,
    )
    _populate_warp_aggregate(warp_aggregate_td, aggregate_tensor)
    return _provider_support.remember_scalar_result_type(
        result,
        value_type,
        scope=_SCOPE,
    )


def provider_inclusive_sum(
    *,
    value: Any,
    threads_in_warp: int = 32,
    valid_items: Any = None,
    warp_aggregate: Any = None,
) -> Any:
    if isinstance(value, ThreadData):
        return _provider_thread_data_scan(
            primitive_name="inclusive_sum",
            request_kind="warp_inclusive_sum",
            value=value,
            op="sum",
            threads_in_warp=threads_in_warp,
            valid_items=valid_items,
            warp_aggregate=warp_aggregate,
        )

    threads_in_warp = _resolve_threads_in_warp(
        _SCOPE,
        "inclusive_sum",
        threads_in_warp,
    )
    value_type = _resolve_scalar_type(value, feature="inclusive_sum")
    warp_aggregate_td = _validate_warp_aggregate_output(
        output=warp_aggregate,
        value_type=value_type,
        primitive_name="inclusive_sum",
    )
    request = _warp_scan_request(
        request_kind="warp_inclusive_sum",
        value_type=value_type,
        op="sum",
        threads_in_warp=threads_in_warp,
        has_valid_items=valid_items is not None,
        has_warp_aggregate=warp_aggregate_td is not None,
    )
    _provider_support.register_request(request)
    valid_args = (
        [_provider_support.as_valid_items_arg(valid_items, scope=_SCOPE)]
        if valid_items is not None
        else []
    )
    aggregate_tensor = _make_warp_aggregate_tensor(warp_aggregate_td, value_type)
    aggregate_args = [aggregate_tensor.iterator] if aggregate_tensor is not None else []
    result = ffi(
        name=request.symbol_name,
        params_types=[
            value_type,
            *([_provider_support.Int32] if valid_items is not None else []),
            *([llvm.PointerType.get(0)] if aggregate_tensor is not None else []),
        ],
        return_type=value_type,
    )(
        value,
        *valid_args,
        *aggregate_args,
    )
    _populate_warp_aggregate(warp_aggregate_td, aggregate_tensor)
    return _provider_support.remember_scalar_result_type(
        result,
        value_type,
        scope=_SCOPE,
    )


def provider_inclusive_scan(
    *,
    value: Any,
    op: str = "sum",
    threads_in_warp: int = 32,
    valid_items: Any = None,
    warp_aggregate: Any = None,
) -> Any:
    if op == "sum":
        return provider_inclusive_sum(
            value=value,
            threads_in_warp=threads_in_warp,
            valid_items=valid_items,
            warp_aggregate=warp_aggregate,
        )
    if isinstance(value, ThreadData):
        return _provider_thread_data_scan(
            primitive_name="inclusive_scan",
            request_kind="warp_inclusive_scan",
            value=value,
            op=op,
            threads_in_warp=threads_in_warp,
            valid_items=valid_items,
            warp_aggregate=warp_aggregate,
        )

    threads_in_warp = _resolve_threads_in_warp(
        _SCOPE,
        "inclusive_scan",
        threads_in_warp,
    )
    value_type = _resolve_scalar_type(value, feature="inclusive_scan")
    _provider_support.validate_scan_reduce_op_for_type(
        op,
        value_type,
        root_scope=_ROOT_SCOPE,
        feature="inclusive_scan",
        namespace="warp",
    )
    warp_aggregate_td = _validate_warp_aggregate_output(
        output=warp_aggregate,
        value_type=value_type,
        primitive_name="inclusive_scan",
    )
    request = _warp_scan_request(
        request_kind="warp_inclusive_scan",
        value_type=value_type,
        op=op,
        threads_in_warp=threads_in_warp,
        has_valid_items=valid_items is not None,
        has_warp_aggregate=warp_aggregate_td is not None,
    )
    _provider_support.register_request(request)
    valid_args = (
        [_provider_support.as_valid_items_arg(valid_items, scope=_SCOPE)]
        if valid_items is not None
        else []
    )
    aggregate_tensor = _make_warp_aggregate_tensor(warp_aggregate_td, value_type)
    aggregate_args = [aggregate_tensor.iterator] if aggregate_tensor is not None else []
    result = ffi(
        name=request.symbol_name,
        params_types=[
            value_type,
            *([_provider_support.Int32] if valid_items is not None else []),
            *([llvm.PointerType.get(0)] if aggregate_tensor is not None else []),
        ],
        return_type=value_type,
    )(
        value,
        *valid_args,
        *aggregate_args,
    )
    _populate_warp_aggregate(warp_aggregate_td, aggregate_tensor)
    return _provider_support.remember_scalar_result_type(
        result,
        value_type,
        scope=_SCOPE,
    )


_register_renderers()
