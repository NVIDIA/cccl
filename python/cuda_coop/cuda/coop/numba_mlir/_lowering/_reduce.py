# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Build CUB block and warp reduction wrappers.

Providers use shared-core algorithm descriptions with built-in operators,
compiled Python callbacks, and valid-prefix controls. Every call uses planned
scratch storage and returns a value defined only at group rank zero.
"""

from __future__ import annotations

import operator
from enum import Enum
from typing import Any

import numba_cuda_mlir.numba_cuda.types as numba_types
import numpy as np

from cuda.coop._core import (
    BindingKind,
    CxxOperator,
    Dependency,
    PythonOperator,
    SynchronizationScope,
    make_block_reduce_specialization,
    make_warp_reduce_specialization,
    normalize_block_reduce_algorithm,
)

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
from .._semantic import _normalize_numba_callable, _numba_semantic_token
from .._types import (
    BoundedInteger,
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


def normalize_reduce_operation(binary_op: Any) -> str:
    """Return the canonical token for a recognized built-in operator.

    Accept named aliases and the known operator/NumPy callable aliases;
    ``None`` means sum. Raise ``NotImplementedError`` for another callable so
    the qualified planner can consider the custom CUB callback path. Invalid
    strings or non-callable values are argument errors.
    """

    if binary_op is None:
        return "sum"
    if isinstance(binary_op, Enum):
        raise TypeError(
            "cuda.coop.numba_mlir.reduce binary_op must be a string or "
            "stateless device callable"
        )
    if isinstance(binary_op, str):
        token = binary_op.strip().lower().replace("-", "_")
        try:
            return _REDUCE_OPERATOR_ALIASES[token]
        except KeyError:
            choices = ", ".join(sorted(set(_REDUCE_OPERATOR_ALIASES.values())))
            raise ValueError(
                "cuda.coop.numba_mlir.reduce binary_op must be one of: "
                f"{choices}; or a stateless device callback on direct CUB forms"
            ) from None
    try:
        return _CALLABLE_REDUCE_OPERATOR_ALIASES[binary_op]
    except (KeyError, TypeError):
        pass
    if not callable(binary_op):
        raise TypeError(
            "cuda.coop.numba_mlir.reduce binary_op must be a string or "
            "stateless device callable"
        )
    raise NotImplementedError(
        "cuda.coop.numba_mlir.reduce supports sum, multiplies, min, max, "
        "bit_and, bit_or, and bit_xor built-ins, or a stateless device "
        "callback on direct CUB forms"
    )


def validate_reduce_operator_dtype(operation: str, dtype: Any) -> Any:
    """Require the numeric dtype supported by a built-in reduction.

    Normalize through the common numeric policy. Bitwise
    operators further require an integer type. Return the
    normalized compiler dtype for the provider builder.
    """

    dtype = _validate_common_numeric_dtype(
        dtype,
        operation="reduce",
        parameter="value",
    )
    if operation in _BITWISE_REDUCE_OPERATORS and not isinstance(
        dtype, numba_types.Integer
    ):
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
    """Require a factory registration in the expected block or warp namespace.

    Return its scratch ABI and execution/synchronization scopes for the core
    adapter. A missing or mismatched registration cannot define a valid
    provider call contract.
    """

    registered = factory_operation(factory)
    if registered is None:
        raise RuntimeError(f"unregistered cuda.coop provider {factory!r}")
    if registered.namespace != namespace:
        raise RuntimeError(
            f"invalid reduction provider registration {registered!r}"
        )
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
    """Build a CUB block reduction for the requested input and operator.

    Normalize exact block dimensions, dtype, item count, and
    scalar or array form. Valid-prefix controls apply only to
    scalar inputs. Select sum, a built-in C++ operator, or a
    dtype-dependent Python callback from the factory variant.

    Runtime prefix counts use a bounded int32 conversion between one and the
    block size. Adapt the shared-core algorithm with that scalar ABI,
    registered scratch metadata, and dtype support, then return an invocable.
    The callable returns the native scalar result with CUB's root visibility.
    """

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
                op_tokenizer=_numba_semantic_token,
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
            numba_types.int32,
            minimum=1,
            maximum=block_threads,
        )
    adapter = NumbaMlirCoreAdapter(value_abis=value_abis)
    core_specialization = make_block_reduce_specialization(
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
        core_specialization.specialization,
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
    items_per_thread: int = 1,
    value_kind: str | None = None,
    *,
    callback: bool = False,
) -> Any:
    """Build a scalar or array CUB reduction for a physical or logical warp.

    Select sum, a built-in C++ operator, or a Python callback. Retain the
    valid-prefix binding and use a bounded int32 ABI for runtime counts.
    Specialize the core algorithm at the logical width, then materialize it
    with the registered warp scratch and synchronization contract.

    Supply both logical width and enclosing block shape to invocable
    construction so each warp instance gets its own scratch. The native result
    is meaningful at the group's root.
    """

    if threads_per_block is None:
        raise ValueError("threads_per_block must be provided")
    block_dim = normalize_dim_param(threads_per_block)
    threads_in_warp = _positive_int(threads_in_warp, name="threads_in_warp")
    items_per_thread = _positive_int(items_per_thread, name="items_per_thread")
    if value_kind is None:
        value_kind = "scalar" if items_per_thread == 1 else "array"
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
                op_tokenizer=_numba_semantic_token,
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
            numba_types.int32,
            minimum=1,
            maximum=threads_in_warp,
        )
    adapter = NumbaMlirCoreAdapter(value_abis=value_abis)
    core_specialization = make_warp_reduce_specialization(
        dtype=adapter.core_dtype(dtype),
        threads_in_warp=threads_in_warp,
        operation=operation,
        items_per_thread=items_per_thread,
        value_kind=value_kind,
        reduce_operator=reduce_operator,
        valid_items=valid_items_binding,
        include_full_warp=False,
    )
    specialization = adapter.materialize(
        core_specialization.specialization,
        **_provider_metadata(provider_factory, namespace="warp"),
        extra_type_definitions=(numba_type_to_wrapper(dtype),),
    )
    return make_invocable_from_specialization(
        specialization,
        logical_warp_threads=threads_in_warp,
        block_threads=block_dim,
    )


def warp_sum(
    dtype: Any,
    threads_in_warp: int = 32,
    valid_items: Any = None,
    threads_per_block: Any = None,
    items_per_thread: int = 1,
    value_kind: str | None = None,
) -> Any:
    """Build a direct CUB WarpReduce Sum invocable."""

    return _warp_reduce(
        warp_sum,
        dtype,
        threads_in_warp=threads_in_warp,
        valid_items=valid_items,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        value_kind=value_kind,
    )


def warp_reduce_builtin(
    dtype: Any,
    binary_op: Any,
    threads_in_warp: int = 32,
    valid_items: Any = None,
    threads_per_block: Any = None,
    items_per_thread: int = 1,
    value_kind: str | None = None,
) -> Any:
    """Build a direct CUB WarpReduce invocable with a C++ operator."""

    return _warp_reduce(
        warp_reduce_builtin,
        dtype,
        binary_op,
        threads_in_warp,
        valid_items,
        threads_per_block,
        items_per_thread,
        value_kind,
    )


def warp_reduce(
    dtype: Any,
    binary_op: Any,
    threads_in_warp: int = 32,
    valid_items: Any = None,
    threads_per_block: Any = None,
    items_per_thread: int = 1,
    value_kind: str | None = None,
) -> Any:
    """Build a direct CUB WarpReduce invocable with a callback."""

    return _warp_reduce(
        warp_reduce,
        dtype,
        binary_op,
        threads_in_warp,
        valid_items,
        threads_per_block,
        items_per_thread,
        value_kind,
        callback=True,
    )


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
del _factory, _operation


__all__: tuple[str, ...] = ()
