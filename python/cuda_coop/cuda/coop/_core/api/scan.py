# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Common cooperative scan entry points."""

from __future__ import annotations

from enum import Enum
from typing import Any

from ..dtype_policy import validate_common_integer_value_dtype_name
from ..scan import normalize_scan_operator_alias
from ..thread_group import ThreadGroup
from ._dispatch import (
    _backend_module_name,
    _common_group_operation,
    _common_selector,
    _group_primitive_marker,
    _validate_common_operation_group,
)
from ._payload import (
    _ReadableThreadDataLike,
    _validate_common_numeric_scalar,
    _validate_common_numeric_value,
    _validate_common_temp_storage,
)

_COMMON_SCAN_GROUP_KINDS = ("block", "warp", "threads_within_warp")
_COMMON_SCAN_MODES = frozenset({"exclusive", "inclusive"})
_COMMON_SCAN_ALGORITHMS = frozenset({"raking", "raking_memoize", "warp_scans"})
_BITWISE_OPERATORS = frozenset({"bit_and", "bit_or", "bit_xor"})
_WARP_GROUP_KINDS = frozenset({"warp", "threads_within_warp"})


def _common_scan_operator(operation: str, value: Any) -> Any:
    if _backend_module_name() is None or value is None:
        return value
    if not isinstance(value, str) or isinstance(value, Enum):
        raise TypeError(
            f"cuda.coop.{operation} scan_op must be a string; use a "
            "backend-qualified import for custom operators"
        )
    operator = normalize_scan_operator_alias(value)
    if operator is None:
        choices = "bit_and, bit_or, bit_xor, max, min, multiplies, sum"
        raise ValueError(
            f"cuda.coop.{operation} scan_op must be one of: "
            f"{choices}; use a backend-qualified import for custom operators"
        ) from None
    return operator


def _validate_common_scan_value(
    operation: str,
    value: Any,
    scan_op: Any,
) -> str:
    dtype_name = _validate_common_numeric_value(
        operation,
        "value",
        value,
        allow_readonly_thread_data=True,
    )
    assert dtype_name is not None
    if scan_op in _BITWISE_OPERATORS:
        validate_common_integer_value_dtype_name(
            dtype_name,
            operation=operation,
            parameter="value",
        )
    return dtype_name


def _validate_common_scan_options(
    operation: str,
    group: ThreadGroup,
    value: Any,
    *,
    mode: str,
    scan_op: Any,
    initial_value: Any,
    algorithm: Any,
    temp_storage: Any,
) -> None:
    if _backend_module_name() is None:
        return
    _validate_common_operation_group(operation, group)
    if mode == "inclusive" and initial_value is not None:
        raise ValueError(
            f"cuda.coop.{operation} initial_value is not supported for inclusive scans"
        )
    if mode == "exclusive" and scan_op not in {None, "sum"}:
        if initial_value is None:
            raise ValueError(
                f"cuda.coop.{operation} non-sum exclusive scans require initial_value"
            )
    if initial_value is not None:
        _validate_common_numeric_scalar(operation, "initial_value", initial_value)
    if group.kind in _WARP_GROUP_KINDS:
        if isinstance(value, _ReadableThreadDataLike):
            raise TypeError(
                f"cuda.coop.{operation} value must be a numeric scalar "
                "for warp scans in the common API"
            )
        if algorithm is not None:
            raise ValueError(
                f"cuda.coop.{operation} algorithm selection is supported only "
                "for blocks"
            )
        if temp_storage is not None:
            raise ValueError(
                f"cuda.coop.{operation} temp_storage is supported only for blocks"
            )
    elif temp_storage is not None:
        _validate_common_temp_storage(operation, temp_storage)


def _scan_call(
    operation: str,
    group: ThreadGroup,
    value: Any,
    *,
    mode: str,
    scan_op: Any,
    initial_value: Any,
    algorithm: Any,
    temp_storage: Any,
) -> Any:
    scan_op = _common_scan_operator(operation, scan_op)
    if _backend_module_name() is not None:
        _validate_common_scan_value(operation, value, scan_op)
    _validate_common_scan_options(
        operation,
        group,
        value,
        mode=mode,
        scan_op=scan_op,
        initial_value=initial_value,
        algorithm=algorithm,
        temp_storage=temp_storage,
    )
    kwargs = {
        "algorithm": algorithm,
        "temp_storage": temp_storage,
    }
    if operation in {"scan", "exclusive_scan", "inclusive_scan"}:
        kwargs["scan_op"] = scan_op
    if operation in {"scan", "exclusive_scan"}:
        kwargs["initial_value"] = initial_value
    if operation == "scan":
        kwargs["mode"] = mode
    return _group_primitive_marker(operation, group, value, **kwargs)


@_common_group_operation("scan", group_kinds=_COMMON_SCAN_GROUP_KINDS)
def scan(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    mode: str = "exclusive",
    scan_op: Any = None,
    initial_value: Any = None,
    algorithm: Any = None,
    temp_storage: Any = None,
) -> Any:
    """Compute a prefix for every input value in a block or warp group.

    Parameters
    ----------
    group : cuda.coop.ThreadGroup
        Participating threads; see :ref:`thread groups <coop-common-groups>`.
        Supports blocks and physical or logical warps. Warp scans require
        an enclosing block size divisible by 32. All group members must
        execute the primitive together.
    value : numeric scalar or cuda.coop.ThreadDataLike
        Each thread's input. Blocks accept a scalar or a readable
        :ref:`per-thread payload <coop-common-payloads>`; warps accept one scalar
        per lane. Payloads have the same item count and dtype in each thread.
        A block scans payloads in blocked order: all items from thread zero,
        then all items from thread one, and so on. Scalar inputs follow
        linear group rank. The input is preserved.
    mode : {"exclusive", "inclusive"}, optional
        Compile-time choice, default ``"exclusive"``. An exclusive prefix
        combines ``initial_value`` with the inputs preceding the current
        item. An inclusive prefix also includes the current item and has
        no initial value.
    scan_op : str, optional
        Compile-time operator: ``"sum"``, ``"multiplies"``, ``"min"``,
        ``"max"``, ``"bit_and"``, ``"bit_or"``, or ``"bit_xor"``.
        ``None`` selects sum. Bitwise operators require integer values.
        :func:`cuda.coop.numba_mlir.scan` also accepts custom operators.
        :func:`cuda.coop.cutlass.scan` accepts recognized Python and NumPy
        aliases for built-ins, but no custom operators or prefix callbacks.
    initial_value : numeric scalar, optional
        First output of an exclusive scan, combined with every subsequent
        prefix. Defaults to zero for sum; required for other exclusive
        operators. Must be ``None`` for inclusive scans. The value must be
        uniform across the group. Typed scalars must have exactly the input
        dtype. Python literals must be finite and in range; an integer
        input requires an integer literal.
    algorithm : str, optional
        Compile-time block algorithm. ``None`` selects ``"raking"``, which
        scans shared partial reductions. ``"raking_memoize"`` keeps those
        partials in registers to reduce shared-memory reads.
        ``"warp_scans"`` combines warp scans and requires a block size
        divisible by 32. Warp groups require ``None``.
    temp_storage : cuda.coop.TempStorageLike, optional
        :ref:`Scratch descriptor <coop-common-storage>` for a block scan.
        ``None`` uses automatic scratch and reuse synchronization. Warp
        scans require ``None`` and use automatic storage for each group.

    Returns
    -------
    numeric scalar or cuda.coop.ThreadDataLike
        This thread's prefixes, with the input dtype and item count. A scalar
        input produces a scalar. A payload input produces a new payload;
        every output item has a defined value on every participating thread.

    Notes
    -----
    Prefixes restart for each group; this call does not scan across blocks.
    Scan relies on associative operators and may regroup arithmetic.
    Floating-point prefixes can differ from sequential CPU results.
    Integer scans accumulate in the input dtype and can overflow.

    See Also
    --------
    :cpp:struct:`cub::BlockScan`, :cpp:struct:`cub::WarpScan`
        C++ primitive types providing ``ExclusiveScan``, ``InclusiveScan``,
        and their sum overloads.

    Examples
    --------
    Compute inclusive bitwise XOR prefixes across a block with
    Numba-CUDA-MLIR. The qualified import activates the backend, and the
    calls use the common ``cuda.coop`` API.

    .. literalinclude:: ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_scan_examples.py
        :language: python
        :start-after: # scan-example-begin
        :end-before: # scan-example-end
        :dedent: 4

    For a CuTe block scan with shared scratch, see
    :ref:`CUTLASS Scan <coop-cutlass-scan>`. Both qualified APIs add built-in
    operator aliases and aggregate outputs; only Numba-CUDA-MLIR supports
    custom operators and prefix callbacks.
    """

    mode = _common_selector(
        "scan",
        "mode",
        mode,
        _COMMON_SCAN_MODES,
    )
    algorithm = _common_selector(
        "scan",
        "algorithm",
        algorithm,
        _COMMON_SCAN_ALGORITHMS,
        allow_none=True,
    )
    return _scan_call(
        "scan",
        group,
        value,
        mode=mode,
        scan_op=scan_op,
        initial_value=initial_value,
        algorithm=algorithm,
        temp_storage=temp_storage,
    )


@_common_group_operation(
    "exclusive_sum",
    group_kinds=_COMMON_SCAN_GROUP_KINDS,
)
def exclusive_sum(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    algorithm: Any = None,
    temp_storage: Any = None,
) -> Any:
    """Sum the inputs preceding each item, starting with zero.

    Parameters
    ----------
    group : cuda.coop.ThreadGroup
        Block or physical/logical warp whose members execute the primitive
        together; see :ref:`thread groups <coop-common-groups>`. Warp scans
        require an enclosing block size divisible by 32.
    value : numeric scalar or cuda.coop.ThreadDataLike
        Each thread's input. Blocks accept a scalar or a readable
        :ref:`per-thread payload <coop-common-payloads>`; warps accept one scalar
        per lane. All threads use the same dtype and item count. Payload
        items follow blocked order, with all items from each thread placed
        consecutively in linear group-rank order. The input is preserved.
    algorithm : str, optional
        Compile-time block algorithm: ``"raking"``, ``"raking_memoize"``,
        or ``"warp_scans"``; ``None`` selects ``"raking"``. The
        ``"warp_scans"`` algorithm requires a block size divisible by 32.
        Warp groups require ``None``. See :func:`cuda.coop.scan` for the
        algorithm choices.
    temp_storage : cuda.coop.TempStorageLike, optional
        :ref:`Scratch descriptor <coop-common-storage>` for blocks. ``None``
        uses automatic scratch and reuse synchronization. Warp groups
        require ``None``.

    Returns
    -------
    numeric scalar or cuda.coop.ThreadDataLike
        This thread's exclusive prefixes, in the input dtype. The first
        item in each group is zero. A scalar input produces a scalar;
        a payload input produces a new payload with the same item count.

    Notes
    -----
    Prefixes restart for each group. Integer overflow and floating-point
    regrouping follow the rules in :func:`cuda.coop.scan`. Use
    :func:`cuda.coop.exclusive_scan` to supply a nonzero initial value or
    another operator.

    See Also
    --------
    :cpp:struct:`cub::BlockScan`, :cpp:struct:`cub::WarpScan`
        C++ primitive types providing ``ExclusiveSum``.

    Examples
    --------
    Turn per-thread item counts into offsets within one block using
    Numba-CUDA-MLIR. Each offset is the sum of earlier threads' counts.

    .. literalinclude:: ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_scan_examples.py
        :language: python
        :start-after: # exclusive-sum-example-begin
        :end-before: # exclusive-sum-example-end
        :dedent: 4

    The :ref:`CUTLASS Scan example <coop-cutlass-scan>` shows the CuTe
    block form. Its explicit seed can be omitted for an exclusive sum
    starting at zero.
    """

    algorithm = _common_selector(
        "exclusive_sum",
        "algorithm",
        algorithm,
        _COMMON_SCAN_ALGORITHMS,
        allow_none=True,
    )
    return _scan_call(
        "exclusive_sum",
        group,
        value,
        mode="exclusive",
        scan_op=None,
        initial_value=None,
        algorithm=algorithm,
        temp_storage=temp_storage,
    )


@_common_group_operation(
    "inclusive_sum",
    group_kinds=_COMMON_SCAN_GROUP_KINDS,
)
def inclusive_sum(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    algorithm: Any = None,
    temp_storage: Any = None,
) -> Any:
    """Sum the inputs up to and including each item.

    Parameters
    ----------
    group : cuda.coop.ThreadGroup
        Block or physical/logical warp whose members execute the primitive
        together; see :ref:`thread groups <coop-common-groups>`. Warp scans
        require an enclosing block size divisible by 32.
    value : numeric scalar or cuda.coop.ThreadDataLike
        Each thread's input. Blocks accept a scalar or a readable
        :ref:`per-thread payload <coop-common-payloads>`; warps accept one scalar
        per lane. All threads use the same dtype and item count. Payload
        items follow blocked order, with all items from each thread placed
        consecutively in linear group-rank order. The input is preserved.
    algorithm : str, optional
        Compile-time block algorithm: ``"raking"``, ``"raking_memoize"``,
        or ``"warp_scans"``; ``None`` selects ``"raking"``. The
        ``"warp_scans"`` algorithm requires a block size divisible by 32.
        Warp groups require ``None``. See :func:`cuda.coop.scan` for the
        algorithm choices.
    temp_storage : cuda.coop.TempStorageLike, optional
        :ref:`Scratch descriptor <coop-common-storage>` for blocks. ``None``
        uses automatic scratch and reuse synchronization. Warp groups
        require ``None``.

    Returns
    -------
    numeric scalar or cuda.coop.ThreadDataLike
        This thread's inclusive prefixes, in the input dtype. The first
        item in each group equals its input. A scalar input produces a
        scalar; a payload input produces a new payload with the same
        item count.

    Notes
    -----
    Prefixes restart for each group. Integer overflow and floating-point
    regrouping follow the rules in :func:`cuda.coop.scan`. Use
    :func:`cuda.coop.inclusive_scan` for another operator.

    See Also
    --------
    :cpp:struct:`cub::BlockScan`, :cpp:struct:`cub::WarpScan`
        C++ primitive types providing ``InclusiveSum``.

    Examples
    --------
    Scan two values per thread with Numba-CUDA-MLIR. The Load, Scan, and
    Store use the same blocked order, so the output is the prefix sum of
    the source array. The original per-thread values remain available.

    .. literalinclude:: ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_scan_examples.py
        :language: python
        :start-after: # inclusive-sum-example-begin
        :end-before: # inclusive-sum-example-end
        :dedent: 4

    The :ref:`CUTLASS Scan example <coop-cutlass-scan>` uses the same
    Load/Scan/Store composition in CuTe. Use ``inclusive_sum`` without an
    ``initial_value`` to include the current item.
    """

    algorithm = _common_selector(
        "inclusive_sum",
        "algorithm",
        algorithm,
        _COMMON_SCAN_ALGORITHMS,
        allow_none=True,
    )
    return _scan_call(
        "inclusive_sum",
        group,
        value,
        mode="inclusive",
        scan_op=None,
        initial_value=None,
        algorithm=algorithm,
        temp_storage=temp_storage,
    )


@_common_group_operation(
    "exclusive_scan",
    group_kinds=_COMMON_SCAN_GROUP_KINDS,
)
def exclusive_scan(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    scan_op: Any = None,
    initial_value: Any = None,
    algorithm: Any = None,
    temp_storage: Any = None,
) -> Any:
    """Combine an initial value with the inputs preceding each item.

    Parameters
    ----------
    group : cuda.coop.ThreadGroup
        Block or physical/logical warp whose members execute the primitive
        together; see :ref:`thread groups <coop-common-groups>`. Warp scans
        require an enclosing block size divisible by 32.
    value : numeric scalar or cuda.coop.ThreadDataLike
        Each thread's input. Blocks accept a scalar or a readable
        :ref:`per-thread payload <coop-common-payloads>`; warps accept one scalar
        per lane. All threads use the same dtype and item count. Payload
        items follow blocked order, with all items from each thread placed
        consecutively in linear group-rank order. The input is preserved.
    scan_op : str, optional
        Compile-time operator: ``"sum"``, ``"multiplies"``, ``"min"``,
        ``"max"``, ``"bit_and"``, ``"bit_or"``, or ``"bit_xor"``.
        ``None`` selects sum. Bitwise operators require integer values.
        :func:`cuda.coop.numba_mlir.scan` also accepts custom operators.
        :func:`cuda.coop.cutlass.scan` accepts recognized Python and NumPy
        aliases for built-ins, but no custom operators or prefix callbacks.
    initial_value : numeric scalar, optional
        First output of each group, combined with every subsequent prefix.
        Defaults to zero for sum; required for all other operators. Must
        have the same value across the group. Typed scalars must have
        exactly the input dtype. Python literals must be finite and in
        range; an integer input requires an integer literal.
    algorithm : str, optional
        Compile-time block algorithm: ``"raking"``, ``"raking_memoize"``,
        or ``"warp_scans"``; ``None`` selects ``"raking"``. The
        ``"warp_scans"`` algorithm requires a block size divisible by 32.
        Warp groups require ``None``. See :func:`cuda.coop.scan` for the
        algorithm choices.
    temp_storage : cuda.coop.TempStorageLike, optional
        :ref:`Scratch descriptor <coop-common-storage>` for blocks. ``None``
        uses automatic scratch and reuse synchronization. Warp groups
        require ``None``.

    Returns
    -------
    numeric scalar or cuda.coop.ThreadDataLike
        This thread's exclusive prefixes, in the input dtype. A scalar
        input produces a scalar; a payload input produces a new payload
        with the same item count. Every item has a defined output,
        including the first item, whose output is ``initial_value``.

    Notes
    -----
    Equivalent to :func:`cuda.coop.scan` with ``mode="exclusive"``.
    Prefixes restart for each group. See that function for arithmetic
    assumptions and algorithm descriptions.

    See Also
    --------
    :cpp:struct:`cub::BlockScan`, :cpp:struct:`cub::WarpScan`
        C++ primitive types providing ``ExclusiveScan``.

    Examples
    --------
    Compute exclusive minimum prefixes with Numba-CUDA-MLIR. The runtime
    initial value has the same dtype as the input and seeds the first
    output. Each later output is the minimum of that seed and earlier
    input values.

    .. literalinclude:: ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_scan_examples.py
        :language: python
        :start-after: # exclusive-scan-example-begin
        :end-before: # exclusive-scan-example-end
        :dedent: 4

    For an exclusive scan with an initial value in CuTe, see
    :ref:`CUTLASS Scan <coop-cutlass-scan>`. The qualified
    :func:`cuda.coop.cutlass.exclusive_scan` reference also shows a partial
    logical warp with an aggregate output.
    """

    algorithm = _common_selector(
        "exclusive_scan",
        "algorithm",
        algorithm,
        _COMMON_SCAN_ALGORITHMS,
        allow_none=True,
    )
    return _scan_call(
        "exclusive_scan",
        group,
        value,
        mode="exclusive",
        scan_op=scan_op,
        initial_value=initial_value,
        algorithm=algorithm,
        temp_storage=temp_storage,
    )


@_common_group_operation(
    "inclusive_scan",
    group_kinds=_COMMON_SCAN_GROUP_KINDS,
)
def inclusive_scan(
    group: ThreadGroup,
    value: Any,
    /,
    *,
    scan_op: Any = None,
    algorithm: Any = None,
    temp_storage: Any = None,
) -> Any:
    """Combine the inputs up to and including each item.

    Parameters
    ----------
    group : cuda.coop.ThreadGroup
        Block or physical/logical warp whose members execute the primitive
        together; see :ref:`thread groups <coop-common-groups>`. Warp scans
        require an enclosing block size divisible by 32.
    value : numeric scalar or cuda.coop.ThreadDataLike
        Each thread's input. Blocks accept a scalar or a readable
        :ref:`per-thread payload <coop-common-payloads>`; warps accept one scalar
        per lane. All threads use the same dtype and item count. Payload
        items follow blocked order, with all items from each thread placed
        consecutively in linear group-rank order. The input is preserved.
    scan_op : str, optional
        Compile-time operator: ``"sum"``, ``"multiplies"``, ``"min"``,
        ``"max"``, ``"bit_and"``, ``"bit_or"``, or ``"bit_xor"``.
        ``None`` selects sum. Bitwise operators require integer values.
        :func:`cuda.coop.numba_mlir.scan` also accepts custom operators.
        :func:`cuda.coop.cutlass.scan` accepts recognized Python and NumPy
        aliases for built-ins, but no custom operators or prefix callbacks.
    algorithm : str, optional
        Compile-time block algorithm: ``"raking"``, ``"raking_memoize"``,
        or ``"warp_scans"``; ``None`` selects ``"raking"``. The
        ``"warp_scans"`` algorithm requires a block size divisible by 32.
        Warp groups require ``None``. See :func:`cuda.coop.scan` for the
        algorithm choices.
    temp_storage : cuda.coop.TempStorageLike, optional
        :ref:`Scratch descriptor <coop-common-storage>` for blocks. ``None``
        uses automatic scratch and reuse synchronization. Warp groups
        require ``None``.

    Returns
    -------
    numeric scalar or cuda.coop.ThreadDataLike
        This thread's inclusive prefixes, in the input dtype. A scalar
        input produces a scalar; a payload input produces a new payload
        with the same item count. The first item in each group equals
        its input.

    Notes
    -----
    Equivalent to :func:`cuda.coop.scan` with ``mode="inclusive"``; there
    is no initial value. Prefixes restart for each group. See that
    function for arithmetic assumptions and algorithm descriptions.

    See Also
    --------
    :cpp:struct:`cub::BlockScan`, :cpp:struct:`cub::WarpScan`
        C++ primitive types providing ``InclusiveScan``.

    Examples
    --------
    Compute running maxima independently in each group of eight lanes
    with Numba-CUDA-MLIR. The 64-thread block contains eight logical warps.

    .. literalinclude:: ../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_scan_examples.py
        :language: python
        :start-after: # inclusive-scan-example-begin
        :end-before: # inclusive-scan-example-end
        :dedent: 4

    CuTe kernels use the same logical-warp descriptor. See
    :func:`cuda.coop.cutlass.inclusive_scan` for qualified controls and
    :ref:`CUTLASS Scan <coop-cutlass-scan>` for the executable block example.
    """

    algorithm = _common_selector(
        "inclusive_scan",
        "algorithm",
        algorithm,
        _COMMON_SCAN_ALGORITHMS,
        allow_none=True,
    )
    return _scan_call(
        "inclusive_scan",
        group,
        value,
        mode="inclusive",
        scan_op=scan_op,
        initial_value=None,
        algorithm=algorithm,
        temp_storage=temp_storage,
    )


__all__ = [
    "exclusive_scan",
    "exclusive_sum",
    "inclusive_scan",
    "inclusive_sum",
    "scan",
]
