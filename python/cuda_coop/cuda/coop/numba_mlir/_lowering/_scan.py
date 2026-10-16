# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Build Numba Scan providers from shared CUB primitive descriptions.

These factories normalize selectors, describe operators and optional seeds,
and adapt the shared block or Warp specialization to Numba's calling
convention. The resulting invocable supplies provider code and storage
metadata to kernel compilation. Registration defines its storage argument
and execution scope for the compiler rewrite.
"""

from __future__ import annotations

import operator
from enum import Enum
from typing import Any

import numba_cuda_mlir.numba_cuda.types as numba_types
import numpy as np

from cuda.coop._core import (
    BindingKind,
    CxxFunction,
    CxxOperator,
    Dependency,
    PythonOperator,
    Reference,
    StatefulOperator,
    SynchronizationScope,
    make_block_scan_specialization,
    make_warp_scan_specialization,
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
from .._semantic import _normalize_numba_callable, _numba_semantic_token
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
    """Return a built-in name or mark a callable for device compilation.

    Recognized Python and NumPy aliases use the same C++ operators as string
    selectors. Other callables return None so the caller can create a typed
    PythonOperator. An omitted selector is the built-in sum.
    """

    if scan_op is None:
        return "sum"
    if isinstance(scan_op, str) and not isinstance(scan_op, Enum):
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
        "cuda.coop.numba_mlir scan_op must be a string or stateless "
        "device callback"
    )


def validate_scan_operator_dtype(scan_op: Any, dtype: Any) -> Any:
    """Normalize the numeric dtype and require integers for bitwise scans."""

    dtype = _validate_common_numeric_dtype(
        dtype,
        operation="scan",
        parameter="value",
    )
    operation = normalize_scan_operation(scan_op)
    if operation in _BITWISE_SCAN_OPERATORS and not isinstance(
        dtype, numba_types.Integer
    ):
        raise TypeError(
            f"cuda.coop.numba_mlir scan {operation} requires an integer dtype"
        )
    return dtype


def _positive_int(value: Any, *, name: str) -> int:
    """Accept a positive index-like value, excluding booleans."""

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
    """Normalize a supported string selector to the shared CUB enum."""

    if not isinstance(algorithm, str) or isinstance(algorithm, Enum):
        raise TypeError("block scan algorithm must be a string")
    token = algorithm.strip().lower().replace("-", "_")
    if token not in {"raking", "raking_memoize", "warp_scans"}:
        raise ValueError(
            "block scan algorithm must be one of: raking, raking_memoize, "
            "warp_scans"
        )
    return normalize_block_scan_algorithm(token)


def _scan_mode(mode: Any) -> str:
    """Normalize the inclusive or exclusive mode used by both providers."""

    if not isinstance(mode, str) or isinstance(mode, Enum):
        raise TypeError("scan mode must be a string")
    token = mode.strip().lower().replace("-", "_")
    if token not in {"exclusive", "inclusive"}:
        raise ValueError("scan mode must be 'exclusive' or 'inclusive'")
    return token


def _provider_metadata(factory: Any, *, namespace: str) -> dict[str, Any]:
    """Read storage and synchronization rules from the factory registry.

    The rewrite and generated wrapper must agree on these rules. Use the
    registered values instead of maintaining a second copy in each factory.
    """

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
    """Choose CUB's sum overload or describe an explicit binary operator.

    A supplied seed forces sum through the general Scan overload with a plus
    functor. Other built-ins use C++ functors; custom callables carry their
    input/output type contract into backend specialization.
    """

    operation = normalize_scan_operation(scan_op)
    if operation == "sum" and not force_sum_operator:
        return None
    if operation is None:
        return PythonOperator(
            op_tokenizer=_numba_semantic_token,
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


def _prefix_operator(
    prefix_op: Any,
) -> PythonOperator | StatefulOperator | None:
    """Describe a prefix callback without binding its runtime state.

    A ``StatefulFunction`` contributes its callable and numeric state dtype;
    the callback argument and result still depend on the scan payload type.
    A plain callable becomes a stateless unary operator. Return ``None``
    when no prefix callback is requested, and reject non-callable inputs.

    The core adapter resolves these descriptors and compiles their callbacks
    during specialization. State contents never enter this description.
    """

    if prefix_op is None:
        return None

    from .._stateful_function import StatefulFunction

    if isinstance(prefix_op, StatefulFunction):
        state_dtype = _validate_common_numeric_dtype(
            normalize_dtype_param(prefix_op.dtype),
            operation="scan",
            parameter="prefix_state",
        )
        return StatefulOperator(
            op_tokenizer=_numba_semantic_token,
            op=prefix_op.op,
            state_dtype=NumbaMlirCoreAdapter().core_dtype(state_dtype),
            ret_dtype=Dependency("T"),
            arg_dtypes=(Dependency("T"),),
            name="prefix_op",
        )

    normalized = _normalize_numba_callable(prefix_op)
    if not callable(normalized):
        raise TypeError(
            "prefix_op must be a stateless device callable or StatefulFunction"
        )
    return PythonOperator(
        op_tokenizer=_numba_semantic_token,
        ret_dtype=Dependency("T"),
        arg_dtypes=(Dependency("T"),),
        op=normalized,
        name="prefix_op",
    )


def _initial_value(binding: Any, dtype: Any) -> Any:
    """Represent a seed as no argument, a runtime reference, or typed C++.

    Static values are embedded in the provider, while runtime values remain
    kernel operands. Both use the same payload type dependency.
    """

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
    prefix_op: Any = None,
    prefix_state: Any = None,
    block_aggregate: Any = None,
    algorithm: Any = "raking",
) -> Any:
    """Build the scalar or array BlockScan provider selected by the factory.

    Validate shape, mode, operator, seed, and prefix-callback rules before
    adapting the shared specialization to Numba. A stateful callback requires
    ``prefix_state`` to indicate that the device call has a state operand;
    the factory does not receive that array's contents. A prefix callback
    excludes an explicit seed and a separate aggregate output.

    The factory identity selects the payload calling convention and registry
    metadata. Materialization resolves operator types and compiles callback
    LTO. Invocable construction supplies the provider wrapper and storage
    metadata; a requested aggregate remains a separate output reference.
    """

    if threads_per_block is None:
        raise ValueError("threads_per_block must be provided")
    block_dim = normalize_dim_param(threads_per_block)
    items_per_thread = _positive_int(items_per_thread, name="items_per_thread")
    expected_kind = (
        "array" if provider_factory is block_scan_array else "scalar"
    )
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
    prefix_operator = _prefix_operator(prefix_op)
    from .._stateful_function import StatefulFunction

    stateful_prefix = isinstance(prefix_op, StatefulFunction)
    has_prefix_state = prefix_state is not None and prefix_state is not False
    if stateful_prefix and not has_prefix_state:
        raise ValueError(
            "StatefulFunction prefix callbacks require prefix_state"
        )
    if not stateful_prefix and has_prefix_state:
        raise ValueError(
            "stateless prefix callbacks do not accept prefix_state"
        )
    if (
        prefix_operator is not None
        and initial_binding.kind is not BindingKind.OMITTED
    ):
        raise ValueError(
            "initial_value and prefix callbacks are mutually exclusive"
        )
    if (
        prefix_operator is not None
        and block_aggregate is not None
        and block_aggregate is not False
    ):
        raise ValueError(
            "block_aggregate and prefix callbacks are mutually exclusive"
        )
    if (
        mode == "exclusive"
        and operation != "sum"
        and initial_binding.kind is BindingKind.OMITTED
        and prefix_operator is None
    ):
        raise ValueError("non-sum exclusive scan requires initial_value")
    scan_operator = _scan_operator(
        scan_op,
        force_sum_operator=initial_binding.kind is not BindingKind.OMITTED,
    )
    core_specialization = make_block_scan_specialization(
        dtype=NumbaMlirCoreAdapter().core_dtype(dtype),
        block_dim=tuple(block_dim),
        items_per_thread=items_per_thread,
        mode=mode,
        algorithm=_block_scan_algorithm(algorithm),
        value_kind=value_kind,
        scan_operator=scan_operator,
        initial_value=_initial_value(initial_binding, dtype),
        prefix_operator=prefix_operator,
        block_aggregate=(
            block_aggregate is not None and block_aggregate is not False
        ),
    )
    adapter = NumbaMlirCoreAdapter()
    specialization = adapter.materialize(
        core_specialization.specialization,
        **_provider_metadata(provider_factory, namespace="block"),
        extra_type_definitions=(numba_type_to_wrapper(dtype),),
    )
    return make_invocable_from_specialization(specialization)


def block_scan_scalar(**kwargs: Any) -> Any:
    """Build a BlockScan provider returning one scalar prefix per thread."""

    return _block_scan(block_scan_scalar, **kwargs)


def block_scan_array(**kwargs: Any) -> Any:
    """Build a BlockScan provider with separate input and output arrays."""

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
    """Build a scalar WarpScan provider for a physical or logical warp.

    Pass the lane-group width and enclosing block shape to the invocable so
    it can select the correct storage slice and synchronization mask. Runtime
    ``valid_items`` values use a bounded integer argument: the wrapper checks
    that the value is between 1 and the group width before narrowing the
    argument for CUB.
    """

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
            numba_types.int32,
            minimum=1,
            maximum=threads_in_warp,
        )
    adapter = NumbaMlirCoreAdapter(value_abis=value_abis)
    core_specialization = make_warp_scan_specialization(
        dtype=adapter.core_dtype(dtype),
        threads_in_warp=threads_in_warp,
        mode=mode,
        scan_operator=_scan_operator(
            scan_op,
            force_sum_operator=force_sum_operator,
        ),
        initial_value=_initial_value(initial_binding, dtype),
        valid_items=valid_items_binding,
        warp_aggregate=(
            warp_aggregate is not None and warp_aggregate is not False
        ),
    )
    specialization = adapter.materialize(
        core_specialization.specialization,
        **_provider_metadata(warp_scan, namespace="warp"),
        extra_type_definitions=(numba_type_to_wrapper(dtype),),
    )
    return make_invocable_from_specialization(
        specialization,
        logical_warp_threads=threads_in_warp,
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


__all__ = [
    "_block_scan_algorithm",
    "_scan_mode",
    "normalize_scan_operation",
    "validate_scan_operator_dtype",
]
