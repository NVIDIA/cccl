# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUB BlockScan and WarpScan provider lowering."""

from __future__ import annotations

import operator
from typing import Any

import numpy as np
from numba_cuda_mlir import types

from cuda.coop._core import (
    BindingKind,
    CxxFunction,
    CxxOperator,
    Dependency,
    PythonOperator,
    Reference,
    SynchronizationScope,
    make_block_scan_spec,
    make_warp_scan_spec,
    normalize_block_scan_algorithm,
)
from cuda.coop._core.scan import normalize_scan_operator_alias

from .._compiler._operations import (
    StorageABI,
    factory_operation,
    register_factory,
)
from .._compiler._parameters import (
    _validate_common_numeric_dtype,
    coerce_static_scalar,
    make_typed_cpp_literal,
    normalize_dim_param,
    normalize_dtype_param,
)
from .._semantic import _normalize_numba_callable
from .._types import (
    BoundedInteger,
    make_invocable_from_specialization,
    numba_type_to_wrapper,
)
from ._core import NumbaMlirCoreAdapter, _optional_binding

_BUILTIN_SCAN_OPERATORS = {
    "multiplies": "::cuda::std::multiplies<T>",
    "min": "::cuda::minimum<T>",
    "max": "::cuda::maximum<T>",
    "bit_and": "::cuda::std::bit_and<T>",
    "bit_or": "::cuda::std::bit_or<T>",
    "bit_xor": "::cuda::std::bit_xor<T>",
}
_CALLABLE_SCAN_OPERATOR_ALIASES = {
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
_BITWISE_SCAN_OPERATORS = frozenset({"bit_and", "bit_or", "bit_xor"})


def normalize_scan_operation(scan_op: Any) -> str | None:
    """Return a canonical built-in token or ``None`` for a callback."""

    if scan_op is None:
        return "sum"
    if isinstance(scan_op, str):
        operation = normalize_scan_operator_alias(scan_op)
        if operation is not None:
            return operation
        raise ValueError(
            "cuda.coop.numba_mlir scan_op must name sum, multiplies, min, "
            "max, bit_and, bit_or, or bit_xor"
        )
    try:
        return _CALLABLE_SCAN_OPERATOR_ALIASES[scan_op]
    except (KeyError, TypeError):
        pass
    if callable(scan_op):
        return None
    raise TypeError(
        "cuda.coop.numba_mlir scan_op must be a string or stateless device callback"
    )


def validate_scan_operator_dtype(scan_op: Any, dtype: Any) -> Any:
    """Validate a Scan operator against the portable numeric dtype profile."""

    dtype = _validate_common_numeric_dtype(
        dtype,
        operation="scan",
        parameter="value",
    )
    operation = normalize_scan_operation(scan_op)
    if operation in _BITWISE_SCAN_OPERATORS and not isinstance(dtype, types.Integer):
        raise TypeError(
            f"cuda.coop.numba_mlir scan {operation} requires an integer dtype"
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


def _block_scan_algorithm(algorithm: Any) -> Any:
    if not isinstance(algorithm, str):
        raise TypeError("block scan algorithm must be a string")
    token = algorithm.strip().lower().replace("-", "_")
    if token not in {"raking", "raking_memoize", "warp_scans"}:
        raise ValueError(
            "block scan algorithm must be one of: raking, raking_memoize, warp_scans"
        )
    return normalize_block_scan_algorithm(token)


def _scan_mode(mode: Any) -> str:
    if not isinstance(mode, str):
        raise TypeError("scan mode must be a string")
    token = mode.strip().lower().replace("-", "_")
    if token not in {"exclusive", "inclusive"}:
        raise ValueError("scan mode must be 'exclusive' or 'inclusive'")
    return token


def _provider_metadata(factory: Any, *, namespace: str) -> dict[str, Any]:
    registered = factory_operation(factory)
    if registered is None:
        raise RuntimeError(f"unregistered cuda.coop provider {factory!r}")
    if registered.namespace != namespace:
        raise RuntimeError(f"invalid scan provider registration {registered!r}")
    return {
        "storage_abi": registered.storage_abi,
        "execution_scope": registered.execution_scope,
        "synchronization_scope": registered.synchronization_scope,
    }


def _scan_operator(scan_op: Any, *, force_sum_operator: bool) -> Any:
    operation = normalize_scan_operation(scan_op)
    if operation == "sum" and not force_sum_operator:
        return None
    if operation is None:
        return PythonOperator(
            ret_dtype=Dependency("T"),
            arg_dtypes=(Dependency("T"), Dependency("T")),
            op=_normalize_numba_callable(scan_op),
            name="scan_op",
        )
    cpp = (
        "::cuda::std::plus<T>"
        if operation == "sum"
        else _BUILTIN_SCAN_OPERATORS[operation]
    )
    return CxxOperator(cpp=cpp, dtype=Dependency("T"), name="scan_op")


def _initial_value(binding: Any, dtype: Any) -> Any:
    binding = _optional_binding(binding)
    if binding.kind is BindingKind.OMITTED:
        return None
    if binding.kind is BindingKind.RUNTIME:
        return Reference(Dependency("T"), name="initial_value")
    value = coerce_static_scalar(
        binding.value,
        dtype,
        operation="scan",
        parameter="initial_value",
    )
    return CxxFunction(
        make_typed_cpp_literal(value, dtype),
        Dependency("T"),
        name="initial_value",
    )


def _block_scan(
    provider_factory: Any,
    dtype: Any,
    threads_per_block: Any = None,
    items_per_thread: int = 1,
    value_kind: str = "scalar",
    mode: str = "exclusive",
    scan_op: Any = None,
    initial_value: Any = None,
    block_aggregate: Any = None,
    algorithm: Any = "raking",
) -> Any:
    if threads_per_block is None:
        raise ValueError("threads_per_block must be provided")
    block_dim = normalize_dim_param(threads_per_block)
    items_per_thread = _positive_int(items_per_thread, name="items_per_thread")
    expected_kind = "array" if provider_factory is block_scan_array else "scalar"
    if value_kind != expected_kind:
        raise ValueError(
            f"{provider_factory.__name__} requires value_kind={expected_kind!r}"
        )
    if value_kind == "scalar" and items_per_thread != 1:
        raise ValueError("scalar block scan requires items_per_thread == 1")
    mode = _scan_mode(mode)

    dtype = normalize_dtype_param(dtype)
    dtype = validate_scan_operator_dtype(scan_op, dtype)
    initial_binding = _optional_binding(initial_value)
    if mode == "inclusive" and initial_binding.kind is not BindingKind.OMITTED:
        raise ValueError("inclusive scan does not accept initial_value")
    operation = normalize_scan_operation(scan_op)
    if (
        mode == "exclusive"
        and operation != "sum"
        and initial_binding.kind is BindingKind.OMITTED
    ):
        raise ValueError("non-sum exclusive scan requires initial_value")
    scan_operator = _scan_operator(
        scan_op,
        force_sum_operator=initial_binding.kind is not BindingKind.OMITTED,
    )
    core_spec = make_block_scan_spec(
        dtype=NumbaMlirCoreAdapter().core_dtype(dtype),
        block_dim=tuple(block_dim),
        items_per_thread=items_per_thread,
        mode=mode,
        algorithm=_block_scan_algorithm(algorithm),
        value_kind=value_kind,
        scan_operator=scan_operator,
        initial_value=_initial_value(initial_binding, dtype),
        block_aggregate=(block_aggregate is not None and block_aggregate is not False),
    )
    adapter = NumbaMlirCoreAdapter()
    specialization = adapter.materialize(
        core_spec.specialization,
        **_provider_metadata(provider_factory, namespace="block"),
        extra_type_definitions=(numba_type_to_wrapper(dtype),),
    )
    return make_invocable_from_specialization(specialization)


def block_scan_scalar(**kwargs: Any) -> Any:
    """Build a direct scalar CUB BlockScan invocable."""

    return _block_scan(block_scan_scalar, **kwargs)


def block_scan_array(**kwargs: Any) -> Any:
    """Build a direct array CUB BlockScan invocable."""

    return _block_scan(block_scan_array, **kwargs)


def warp_scan(
    dtype: Any,
    threads_in_warp: int = 32,
    threads_per_block: Any = None,
    mode: str = "exclusive",
    scan_op: Any = None,
    initial_value: Any = None,
    valid_items: Any = None,
    warp_aggregate: Any = None,
) -> Any:
    """Build a direct scalar CUB WarpScan invocable."""

    if threads_per_block is None:
        raise ValueError("threads_per_block must be provided")
    block_dim = normalize_dim_param(threads_per_block)
    threads_in_warp = _positive_int(threads_in_warp, name="threads_in_warp")
    mode = _scan_mode(mode)
    dtype = normalize_dtype_param(dtype)
    dtype = validate_scan_operator_dtype(scan_op, dtype)
    initial_binding = _optional_binding(initial_value)
    valid_items_binding = _optional_binding(valid_items)
    if mode == "inclusive" and initial_binding.kind is not BindingKind.OMITTED:
        raise ValueError("inclusive scan does not accept initial_value")
    operation = normalize_scan_operation(scan_op)
    if (
        mode == "exclusive"
        and operation != "sum"
        and initial_binding.kind is BindingKind.OMITTED
    ):
        raise ValueError("non-sum exclusive scan requires initial_value")

    # Let the core WarpScan canonicalizer inject an explicitly typed zero for
    # partial exclusive sums. An explicit initial still selects Scan rather
    # than Sum directly.
    force_sum_operator = initial_binding.kind is not BindingKind.OMITTED
    value_abis = {}
    if valid_items_binding.kind is BindingKind.RUNTIME:
        value_abis["valid_items"] = BoundedInteger(
            types.int32,
            minimum=1,
            maximum=threads_in_warp,
        )
    adapter = NumbaMlirCoreAdapter(value_abis=value_abis)
    core_spec = make_warp_scan_spec(
        dtype=adapter.core_dtype(dtype),
        threads_in_warp=threads_in_warp,
        mode=mode,
        scan_operator=_scan_operator(
            scan_op,
            force_sum_operator=force_sum_operator,
        ),
        initial_value=_initial_value(initial_binding, dtype),
        valid_items=valid_items_binding,
        warp_aggregate=(warp_aggregate is not None and warp_aggregate is not False),
    )
    specialization = adapter.materialize(
        core_spec.specialization,
        **_provider_metadata(warp_scan, namespace="warp"),
        extra_type_definitions=(numba_type_to_wrapper(dtype),),
    )
    return make_invocable_from_specialization(
        specialization,
        threads=threads_in_warp,
        block_threads=block_dim,
    )


for _factory, _operation, _namespace, _scope in (
    (
        block_scan_scalar,
        "block_scan_scalar",
        "block",
        SynchronizationScope.BLOCK,
    ),
    (
        block_scan_array,
        "block_scan_array",
        "block",
        SynchronizationScope.BLOCK,
    ),
    (warp_scan, "warp_scan", "warp", SynchronizationScope.WARP),
):
    register_factory(
        _factory,
        operation=_operation,
        namespace=_namespace,
        storage_abi=StorageABI.LEADING_POINTER,
        execution_scope=_scope,
        synchronization_scope=_scope,
    )
del _factory, _namespace, _operation, _scope


__all__: tuple[str, ...] = ()
