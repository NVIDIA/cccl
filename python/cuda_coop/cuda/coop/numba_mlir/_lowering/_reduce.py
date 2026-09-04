# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUB and CUDAX reduction provider lowering."""

from __future__ import annotations

import hashlib
import operator
import re
from typing import Any

import numpy as np
from numba_cuda_mlir import cuda, types

from cuda.coop._core import (
    BindingKind,
    CxxOperator,
    Dependency,
    PythonOperator,
    SynchronizationScope,
    ThreadGroup,
    make_block_reduce_spec,
    make_warp_reduce_spec,
    normalize_block_reduce_algorithm,
    render_group_decl_lines,
    render_hierarchy_decl,
)

from .._compiler import _nvrtc
from .._compiler._operations import (
    StorageABI,
    factory_operation,
    register_factory,
)
from .._compiler._parameters import (
    _validate_common_numeric_dtype,
    normalize_dim_param,
    normalize_dtype_param,
)
from .._semantic import _normalize_numba_callable
from .._types import (
    NUMBA_TYPES_TO_CPP,
    Array,
    BoundedInteger,
    RawCAbiInvocable,
    make_invocable_from_specialization,
    numba_type_to_wrapper,
)
from ._core import NumbaMlirCoreAdapter, _optional_binding

_BUILTIN_REDUCE_OPERATORS = {
    "multiplies": "::cuda::std::multiplies<T>",
    "min": "::cuda::minimum<T>",
    "max": "::cuda::maximum<T>",
    "bit_and": "::cuda::std::bit_and<T>",
    "bit_or": "::cuda::std::bit_or<T>",
    "bit_xor": "::cuda::std::bit_xor<T>",
}
_CUDAX_REDUCE_OPERATORS = {
    "sum": "::cuda::std::plus<>{}",
    "multiplies": "::cuda::std::multiplies<>{}",
    "min": "::cuda::minimum<>{}",
    "max": "::cuda::maximum<>{}",
    "bit_and": "::cuda::std::bit_and<>{}",
    "bit_or": "::cuda::std::bit_or<>{}",
    "bit_xor": "::cuda::std::bit_xor<>{}",
}
_REDUCE_OPERATOR_ALIASES = {
    None: "sum",
    "+": "sum",
    "sum": "sum",
    "add": "sum",
    "plus": "sum",
    "*": "multiplies",
    "mul": "multiplies",
    "multiply": "multiplies",
    "multiplies": "multiplies",
    "min": "min",
    "minimum": "min",
    "max": "max",
    "maximum": "max",
    "&": "bit_and",
    "bit_and": "bit_and",
    "|": "bit_or",
    "bit_or": "bit_or",
    "^": "bit_xor",
    "bit_xor": "bit_xor",
}
_CALLABLE_REDUCE_OPERATOR_ALIASES = {
    operator.add: "sum",
    operator.mul: "multiplies",
    operator.and_: "bit_and",
    operator.or_: "bit_or",
    operator.xor: "bit_xor",
    np.add: "sum",
    np.multiply: "multiplies",
    np.minimum: "min",
    np.maximum: "max",
    np.bitwise_and: "bit_and",
    np.bitwise_or: "bit_or",
    np.bitwise_xor: "bit_xor",
}
_BITWISE_REDUCE_OPERATORS = frozenset({"bit_and", "bit_or", "bit_xor"})
_CUDAX_INCLUDE_LINES = (
    "#define _CUDAX_ENABLE_GROUP_FEATURES_IN_LIBCUDACXX",
    "#define _CUDAX_DISABLE_COOPERATIVE_GROUPS_INTEROP",
    "#include <cuda/barrier>",
    "#include <cuda/devices>",
    "#include <cuda/functional>",
    "#include <cuda/hierarchy>",
    "#include <cuda/std/cstdint>",
    "#include <cuda/std/functional>",
    "#include <cuda/std/type_traits>",
    "#include <cuda/experimental/coop.cuh>",
    "#include <cuda/experimental/group.cuh>",
)


def normalize_reduce_operation(binary_op: Any) -> str:
    """Return the canonical built-in reduction token."""

    try:
        return _REDUCE_OPERATOR_ALIASES[binary_op]
    except (KeyError, TypeError):
        pass
    try:
        return _CALLABLE_REDUCE_OPERATOR_ALIASES[binary_op]
    except (KeyError, TypeError):
        pass
    raise NotImplementedError(
        "cuda.coop.numba_mlir.reduce supports sum, multiplies, min, max, "
        "bit_and, bit_or, and bit_xor built-ins, or a stateless device "
        "callback on direct CUB forms"
    )


def validate_reduce_operator_dtype(operation: str, dtype: Any) -> Any:
    """Validate the payload dtype required by one built-in operator."""

    dtype = _validate_common_numeric_dtype(
        dtype,
        operation="reduce",
        parameter="value",
    )
    if operation in _BITWISE_REDUCE_OPERATORS and not isinstance(dtype, types.Integer):
        raise TypeError(
            f"cuda.coop.numba_mlir.reduce {operation} requires an integer dtype"
        )
    return dtype


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _provider_metadata(factory: Any, *, namespace: str) -> dict[str, Any]:
    registered = factory_operation(factory)
    if registered is None:
        raise RuntimeError(f"unregistered cuda.coop provider {factory!r}")
    if registered.namespace != namespace:
        raise RuntimeError(f"invalid reduction provider registration {registered!r}")
    return {
        "storage_abi": registered.storage_abi,
        "execution_scope": registered.execution_scope,
        "synchronization_scope": registered.synchronization_scope,
    }


def _block_reduce(
    provider_factory: Any,
    dtype: Any,
    threads_per_block: Any = None,
    binary_op: Any = None,
    items_per_thread: int = 1,
    value_kind: str | None = None,
    algorithm: Any = "warp_reductions",
    num_valid: Any = None,
    *,
    callback: bool = False,
) -> Any:
    if threads_per_block is None:
        raise ValueError("threads_per_block must be provided")
    block_dim = normalize_dim_param(threads_per_block)
    items_per_thread = _positive_int(items_per_thread, name="items_per_thread")
    if value_kind is None:
        value_kind = "scalar" if items_per_thread == 1 else "array"
    if value_kind not in {"array", "scalar"}:
        raise ValueError("value_kind must be 'array' or 'scalar'")
    if value_kind == "scalar" and items_per_thread != 1:
        raise ValueError("scalar reduce requires items_per_thread == 1")
    valid_items = _optional_binding(num_valid)
    if valid_items.kind is not BindingKind.OMITTED and value_kind == "array":
        raise ValueError("num_valid is not supported for array inputs")
    dtype = normalize_dtype_param(dtype)
    reduce_operator = None
    operation = "sum"
    if provider_factory is not sum:
        operation = "reduce"
        if callback:
            if not callable(binary_op):
                raise TypeError("binary_op must be a stateless device callable")
            reduce_operator = PythonOperator(
                ret_dtype=Dependency("T"),
                arg_dtypes=(Dependency("T"), Dependency("T")),
                op=_normalize_numba_callable(binary_op),
                name="binary_op",
            )
        else:
            canonical = normalize_reduce_operation(binary_op)
            if canonical == "sum":
                raise ValueError("block_reduce_builtin does not accept sum")
            dtype = validate_reduce_operator_dtype(canonical, dtype)
            reduce_operator = CxxOperator(
                cpp=_BUILTIN_REDUCE_OPERATORS[canonical],
                dtype=Dependency("T"),
                name="binary_op",
            )
    else:
        dtype = validate_reduce_operator_dtype("sum", dtype)

    value_abis = {}
    if valid_items.kind is BindingKind.RUNTIME:
        block_threads = block_dim.x * block_dim.y * block_dim.z
        value_abis["num_valid"] = BoundedInteger(
            types.int32,
            minimum=1,
            maximum=block_threads,
        )
    adapter = NumbaMlirCoreAdapter(value_abis=value_abis)
    core_spec = make_block_reduce_spec(
        dtype=adapter.core_dtype(dtype),
        block_dim=tuple(block_dim),
        items_per_thread=items_per_thread,
        operation=operation,
        algorithm=normalize_block_reduce_algorithm(algorithm),
        value_kind=value_kind,
        reduce_operator=reduce_operator,
        valid_items=valid_items,
    )
    specialization = adapter.materialize(
        core_spec.specialization,
        **_provider_metadata(provider_factory, namespace="block"),
        extra_type_definitions=(numba_type_to_wrapper(dtype),),
    )
    return make_invocable_from_specialization(specialization)


def sum(
    dtype: Any,
    threads_per_block: Any = None,
    items_per_thread: int = 1,
    value_kind: str | None = None,
    algorithm: Any = "warp_reductions",
    num_valid: Any = None,
) -> Any:
    """Build a direct CUB BlockReduce Sum invocable."""

    return _block_reduce(
        sum,
        dtype,
        threads_per_block,
        items_per_thread=items_per_thread,
        value_kind=value_kind,
        algorithm=algorithm,
        num_valid=num_valid,
    )


def block_reduce_builtin(
    dtype: Any,
    threads_per_block: Any = None,
    binary_op: Any = None,
    items_per_thread: int = 1,
    value_kind: str | None = None,
    algorithm: Any = "warp_reductions",
    num_valid: Any = None,
) -> Any:
    """Build a direct CUB BlockReduce invocable with a C++ operator."""

    return _block_reduce(
        block_reduce_builtin,
        dtype,
        threads_per_block,
        binary_op,
        items_per_thread,
        value_kind,
        algorithm,
        num_valid,
    )


def reduce(
    dtype: Any,
    threads_per_block: Any = None,
    binary_op: Any = None,
    items_per_thread: int = 1,
    value_kind: str | None = None,
    algorithm: Any = "warp_reductions",
    num_valid: Any = None,
) -> Any:
    """Build a direct CUB BlockReduce invocable with a callback."""

    return _block_reduce(
        reduce,
        dtype,
        threads_per_block,
        binary_op,
        items_per_thread,
        value_kind,
        algorithm,
        num_valid,
        callback=True,
    )


def _warp_reduce(
    provider_factory: Any,
    dtype: Any,
    binary_op: Any = None,
    threads_in_warp: int = 32,
    valid_items: Any = None,
    threads_per_block: Any = None,
    *,
    callback: bool = False,
) -> Any:
    if threads_per_block is None:
        raise ValueError("threads_per_block must be provided")
    block_dim = normalize_dim_param(threads_per_block)
    threads_in_warp = _positive_int(threads_in_warp, name="threads_in_warp")
    valid_items_binding = _optional_binding(valid_items)
    dtype = normalize_dtype_param(dtype)
    reduce_operator = None
    operation = "sum"
    if provider_factory is not warp_sum:
        operation = "reduce"
        if callback:
            if not callable(binary_op):
                raise TypeError("binary_op must be a stateless device callable")
            reduce_operator = PythonOperator(
                ret_dtype=Dependency("T"),
                arg_dtypes=(Dependency("T"), Dependency("T")),
                op=_normalize_numba_callable(binary_op),
                name="binary_op",
            )
        else:
            canonical = normalize_reduce_operation(binary_op)
            if canonical == "sum":
                raise ValueError("warp_reduce_builtin does not accept sum")
            dtype = validate_reduce_operator_dtype(canonical, dtype)
            reduce_operator = CxxOperator(
                cpp=_BUILTIN_REDUCE_OPERATORS[canonical],
                dtype=Dependency("T"),
                name="binary_op",
            )
    else:
        dtype = validate_reduce_operator_dtype("sum", dtype)

    value_abis = {}
    if valid_items_binding.kind is BindingKind.RUNTIME:
        value_abis["valid_items"] = BoundedInteger(
            types.int32,
            minimum=1,
            maximum=threads_in_warp,
        )
    adapter = NumbaMlirCoreAdapter(value_abis=value_abis)
    core_spec = make_warp_reduce_spec(
        dtype=adapter.core_dtype(dtype),
        threads_in_warp=threads_in_warp,
        operation=operation,
        reduce_operator=reduce_operator,
        valid_items=valid_items_binding,
        include_full_warp=False,
    )
    specialization = adapter.materialize(
        core_spec.specialization,
        **_provider_metadata(provider_factory, namespace="warp"),
        extra_type_definitions=(numba_type_to_wrapper(dtype),),
    )
    return make_invocable_from_specialization(
        specialization,
        threads=threads_in_warp,
        block_threads=block_dim,
    )


def warp_sum(
    dtype: Any,
    threads_in_warp: int = 32,
    valid_items: Any = None,
    threads_per_block: Any = None,
) -> Any:
    """Build a direct CUB WarpReduce Sum invocable."""

    return _warp_reduce(
        warp_sum,
        dtype,
        threads_in_warp=threads_in_warp,
        valid_items=valid_items,
        threads_per_block=threads_per_block,
    )


def warp_reduce_builtin(
    dtype: Any,
    binary_op: Any,
    threads_in_warp: int = 32,
    valid_items: Any = None,
    threads_per_block: Any = None,
) -> Any:
    """Build a direct CUB WarpReduce invocable with a C++ operator."""

    return _warp_reduce(
        warp_reduce_builtin,
        dtype,
        binary_op,
        threads_in_warp,
        valid_items,
        threads_per_block,
    )


def warp_reduce(
    dtype: Any,
    binary_op: Any,
    threads_in_warp: int = 32,
    valid_items: Any = None,
    threads_per_block: Any = None,
) -> Any:
    """Build a direct CUB WarpReduce invocable with a callback."""

    return _warp_reduce(
        warp_reduce,
        dtype,
        binary_op,
        threads_in_warp,
        valid_items,
        threads_per_block,
        callback=True,
    )


def _symbol_component(value: Any) -> str:
    component = re.sub(r"\W+", "_", str(value)).strip("_")
    return component or "anon"


def _group_symbol_component(group: ThreadGroup) -> str:
    digest = hashlib.sha1(repr(group.semantic_key).encode()).hexdigest()[:16]
    return f"{_symbol_component(group.kind)}_{digest}"


def _cpp_type(dtype: Any) -> str:
    dtype = normalize_dtype_param(dtype)
    try:
        return NUMBA_TYPES_TO_CPP[dtype]
    except KeyError as exc:
        raise TypeError(
            "cuda.coop.numba_mlir CUDAX reduce supports built-in numeric "
            f"dtypes; got {dtype!r}"
        ) from exc


def _group_prelude(group: ThreadGroup) -> list[str]:
    hierarchy = group.hierarchy
    if hierarchy.implicit:
        return render_group_decl_lines(group)
    return [
        *render_hierarchy_decl(hierarchy),
        *render_group_decl_lines(group),
    ]


def render_group_reduce_source(
    *,
    group: ThreadGroup,
    dtype: Any,
    items_per_thread: int,
    value_kind: str,
    operation: str,
    broadcast: bool,
    symbol: str,
) -> str:
    """Render one storage-free CUDAX reduction helper."""

    cpp_type = _cpp_type(dtype)
    if value_kind == "scalar":
        parameter = f"{cpp_type} item"
        thread_data = f"  {cpp_type} thread_data[1] = {{item}};"
    else:
        parameter = "void* raw_items"
        thread_data = (
            f"  auto& thread_data = *reinterpret_cast<{cpp_type} "
            f"(*)[{items_per_thread}]>(raw_items);"
        )
    lines = [
        *_CUDAX_INCLUDE_LINES,
        "",
        f'extern "C" __device__ {cpp_type} {symbol}({parameter}) {{',
        *_group_prelude(group),
    ]
    if group.mapping is not None and group.complete_membership is False:
        lines.extend(
            (
                "  if (!::cuda::gpu_thread.is_part_of(group)) {",
                f"    return {cpp_type}{{}};",
                "  }",
            )
        )
    lines.append(thread_data)
    operator_cpp = _CUDAX_REDUCE_OPERATORS[operation]
    if broadcast:
        lines.extend(
            (
                "  auto reduced = ::cuda::experimental::coop::reduce(",
                "      ::cuda::experimental::broadcasted, group, thread_data,",
                f"      {operator_cpp});",
                f"  {cpp_type} result = reduced;",
            )
        )
    else:
        lines.extend(
            (
                "  auto reduced = ::cuda::experimental::coop::reduce(",
                f"      group, thread_data, {operator_cpp});",
                f"  {cpp_type} result = reduced.value_or({cpp_type}{{}});",
            )
        )
    lines.extend(("  return result;", "}", ""))
    return "\n".join(lines)


def _expected_cudax_scope(group: ThreadGroup) -> SynchronizationScope:
    return {
        "thread": SynchronizationScope.NONE,
        "warp": SynchronizationScope.WARP,
        "threads_within_warp": SynchronizationScope.WARP,
        "block": SynchronizationScope.BLOCK,
        "warps_within_block": SynchronizationScope.GROUP,
        "cluster": SynchronizationScope.GROUP,
    }[group.kind]


def _group_reduce(
    provider_factory: Any,
    dtype: Any,
    group: ThreadGroup,
    binary_op: Any = None,
    items_per_thread: int = 1,
    value_kind: str | None = None,
    broadcast: bool = True,
    _compile_context: _nvrtc.CompileContext | None = None,
) -> RawCAbiInvocable:
    if not isinstance(group, ThreadGroup):
        raise TypeError("cuda.coop.numba_mlir.reduce group must be a ThreadGroup")
    if group.kind == "grid":
        raise NotImplementedError(
            "cuda.coop.numba_mlir.reduce grid groups require a hidden "
            "per-launch provider workspace"
        )
    if not isinstance(broadcast, bool):
        raise TypeError("cuda.coop.numba_mlir.reduce broadcast must be a bool")
    items_per_thread = _positive_int(items_per_thread, name="items_per_thread")
    if value_kind is None:
        value_kind = "scalar" if items_per_thread == 1 else "array"
    if value_kind not in {"array", "scalar"}:
        raise ValueError("value_kind must be 'array' or 'scalar'")
    if value_kind == "scalar" and items_per_thread != 1:
        raise ValueError("scalar reduce requires items_per_thread == 1")
    operation = normalize_reduce_operation(binary_op)
    dtype = validate_reduce_operator_dtype(operation, dtype)
    registered = factory_operation(provider_factory)
    if registered is None:
        raise RuntimeError(f"unregistered cuda.coop provider {provider_factory!r}")
    expected_scope = _expected_cudax_scope(group)
    if (
        registered.storage_abi is not StorageABI.NONE
        or registered.execution_scope is not expected_scope
        or registered.synchronization_scope is not SynchronizationScope.NONE
    ):
        raise ValueError(
            "CUDAX reduction provider metadata does not match the group scope"
        )
    device = cuda.get_current_device()
    cc = int(device.compute_capability[0]) * 10 + int(device.compute_capability[1])
    compile_context = (
        _nvrtc.resolve_compile_context()
        if _compile_context is None
        else _compile_context
    )
    mode = "broadcast" if broadcast else "root"
    symbol = (
        "cuda_coop_numba_mlir_group_reduce_"
        f"{_group_symbol_component(group)}_{operation}_"
        f"{_symbol_component(dtype)}_"
        f"{value_kind}_x{items_per_thread}_{mode}_cc{cc}_ctx_"
        f"{compile_context.symbol_suffix}"
    )
    source = render_group_reduce_source(
        group=group,
        dtype=dtype,
        items_per_thread=items_per_thread,
        value_kind=value_kind,
        operation=operation,
        broadcast=broadcast,
        symbol=symbol,
    )
    parameters: tuple[Any, ...]
    transforms: tuple[str, ...]
    if value_kind == "scalar":
        parameters = (dtype,)
        transforms = ("value",)
    else:
        parameters = (Array(dtype, items_per_thread),)
        transforms = ("ptr",)
    return RawCAbiInvocable(
        source=source,
        symbol=symbol,
        return_type=dtype,
        parameters=parameters,
        abi_transforms=transforms,
        cc=cc,
        compile_context=compile_context,
        storage_abi=registered.storage_abi,
        execution_scope=registered.execution_scope,
        synchronization_scope=registered.synchronization_scope,
    )


def group_reduce_none(**kwargs: Any) -> RawCAbiInvocable:
    """Build a current-thread CUDAX reduction provider."""

    return _group_reduce(group_reduce_none, **kwargs)


def group_reduce_warp(**kwargs: Any) -> RawCAbiInvocable:
    """Build a physical or logical-warp CUDAX reduction provider."""

    return _group_reduce(group_reduce_warp, **kwargs)


def group_reduce_block(**kwargs: Any) -> RawCAbiInvocable:
    """Build a block CUDAX reduction provider."""

    return _group_reduce(group_reduce_block, **kwargs)


def group_reduce_group(**kwargs: Any) -> RawCAbiInvocable:
    """Build a mapped-warp or cluster CUDAX reduction provider."""

    return _group_reduce(group_reduce_group, **kwargs)


for _factory, _operation in (
    (sum, "block_sum"),
    (block_reduce_builtin, "block_reduce_builtin"),
    (reduce, "block_reduce_callback"),
):
    register_factory(
        _factory,
        operation=_operation,
        namespace="block",
        storage_abi=StorageABI.LEADING_POINTER,
        execution_scope=SynchronizationScope.BLOCK,
        synchronization_scope=SynchronizationScope.BLOCK,
    )
for _factory, _operation in (
    (warp_sum, "warp_sum"),
    (warp_reduce_builtin, "warp_reduce_builtin"),
    (warp_reduce, "warp_reduce_callback"),
):
    register_factory(
        _factory,
        operation=_operation,
        namespace="warp",
        storage_abi=StorageABI.LEADING_POINTER,
        execution_scope=SynchronizationScope.WARP,
        synchronization_scope=SynchronizationScope.WARP,
    )
for _factory, _namespace, _scope in (
    (group_reduce_none, "cudax_none", SynchronizationScope.NONE),
    (group_reduce_warp, "cudax_warp", SynchronizationScope.WARP),
    (group_reduce_block, "cudax_block", SynchronizationScope.BLOCK),
    (group_reduce_group, "cudax_group", SynchronizationScope.GROUP),
):
    register_factory(
        _factory,
        operation="group_reduce",
        namespace=_namespace,
        storage_abi=StorageABI.NONE,
        execution_scope=_scope,
        synchronization_scope=SynchronizationScope.NONE,
    )
del _factory, _namespace, _operation, _scope


__all__: tuple[str, ...] = ()
