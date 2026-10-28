# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Trace built-in BlockScan and WarpScan calls from CuTe kernels.

Normalize mode and algorithm strings and adapt register tensors and TensorSSA
values to ThreadData before planning. The provider reconciles dtypes, owns
scratch, and emits the C++ call. The public functions support built-in
operators, per-warp valid prefixes, and aggregate outputs.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any, Literal, TypeVar

from cuda.coop.cutlass._planning.thread_group import ThreadGroup
from cuda.coop.cutlass._typing import (
    CommonNumericScalar,
    CommonThreadDataLike,
    ScanAlgorithm,
    ScanOperator,
    TempStorageLike,
    ThreadDataLike,
    ValidItems,
)
from cuda.coop.cutlass._validation import _validate_temp_storage

from ._thread_data import (
    CutlassTensorSample,
    CutlassTensorSSASample,
    ThreadData,
    _coerce_thread_payload,
)
from ._thread_group import _require_complete_warp_partition
from ._typing import BlockGroup, WarpGroup

_ItemT = TypeVar("_ItemT", bound=CommonNumericScalar)


_SCOPE = "cuda.coop.cutlass"
_ALGORITHMS = frozenset({"raking", "raking_memoize", "warp_scans"})


def _selector(value, *, name, choices):
    if not isinstance(value, str) or isinstance(value, Enum):
        raise TypeError(f"{_SCOPE}.scan {name} must be a string")
    value = value.strip().lower().replace("-", "_")
    if value not in choices:
        raise ValueError(
            f"{_SCOPE}.scan {name} must be one of: {', '.join(sorted(choices))}"
        )
    return value


def scan(
    group: BlockGroup | WarpGroup,
    value: CommonThreadDataLike[_ItemT]
    | CutlassTensorSample
    | CutlassTensorSSASample
    | _ItemT,
    /,
    *,
    mode: Literal["exclusive", "inclusive"] = "exclusive",
    scan_op: ScanOperator | Callable[[object, object], object] | None = None,
    initial_value: CommonNumericScalar | None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
    valid_items: ValidItems | None = None,
    aggregate_output: ThreadDataLike[Any] | None = None,
) -> ThreadData | _ItemT:
    """Scan register values with optional valid-prefix and aggregate controls.

    Prefixes follow linear group rank. Block payloads use blocked order: all
    items from thread zero, then all items from thread one, and so on. Prefixes
    restart for each group; this call does not scan across blocks.

    Parameters
    ----------
    group : ThreadGroup
        Complete block, physical warp, or supported logical warp. Every group
        member executes the same call. Warp scans require an enclosing block
        size divisible by 32.
    value : numeric scalar, ThreadData, CuTe register tensor, or TensorSSA
        Blocks accept scalars or fixed-size per-thread payloads in blocked
        order. Register tensors and ``TensorSSA`` values are converted with
        :meth:`cuda.coop.cutlass.ThreadData.from_payload`. All members must use
        the same dtype and extent. Physical and logical warps accept scalars
        only, including when each thread would otherwise hold one item.
    mode : {"exclusive", "inclusive"}, optional
        Compile-time choice, default ``"exclusive"``. Exclusive prefixes
        combine ``initial_value`` with preceding inputs. Inclusive prefixes
        also include the current item and require no initial value.
    scan_op : str or built-in alias, optional
        Compile-time operator, default sum. Supports ``"sum"``,
        ``"multiplies"``, ``"min"``, ``"max"``, ``"bit_and"``, ``"bit_or"``,
        and ``"bit_xor"``, and corresponding ``operator`` or NumPy aliases
        such as ``operator.add`` and ``numpy.maximum``. Bitwise operators
        require an integer dtype. Custom device functions are not supported.
    initial_value : numeric scalar, optional
        Starting value for exclusive mode. Sum defaults to zero; other
        operators require an explicit value. A typed CuTe or NumPy scalar
        must match the input dtype. Python literals must be finite and
        representable in that dtype. Inclusive mode requires ``None``. The
        initial value must be uniform across the group.
    valid_items : int or CuTe integer scalar, optional
        Warp-only count of contributing lanes, from one through the group
        size, uniform within the group. ``None`` includes every lane. All
        lanes participate, but only ranks below this count have defined scan
        results. The enclosing block must contain complete physical warps.
        Blocks require ``None``.
    aggregate_output : ThreadData, optional
        Writable one-item payload receiving the input aggregate on every
        member. Its dtype must match the input, or may be omitted for
        inference. The aggregate excludes ``initial_value`` and lanes beyond
        ``valid_items``. CuTe register tensors and ``TensorSSA`` values are
        not accepted as this output argument. ``None`` omits the aggregate.

    algorithm : str, optional
        Compile-time block algorithm: ``"raking"`` (the default),
        ``"raking_memoize"``, or ``"warp_scans"``. The last requires a block
        size divisible by 32. Warp groups require ``None``.
    temp_storage : TempStorage, optional
        Explicit block scratch descriptor. Omission enables automatic scratch
        and reuse synchronization. Warp groups require ``None`` and use
        automatic storage for each group.

    Returns
    -------
    CuTe numeric scalar or cuda.coop.cutlass.ThreadData
        This thread's prefixes with the input dtype. Scalar input returns a
        CuTe scalar; payload input returns a new writable ``ThreadData`` with
        the input extent and alignment. The input remains unchanged.
        With ``valid_items``, read only valid lanes. ``aggregate_output`` is
        written separately, including on lanes outside the valid prefix.

    Notes
    -----
    Only blocks accept ``algorithm`` and explicit ``temp_storage``. The
    compiler synchronizes automatically when it manages scratch. An explicit
    :class:`cuda.coop.cutlass.TempStorage` defaults to ``auto_sync=False``.
    It requires block barriers before reuse; request ``auto_sync=True`` for
    automatic reuse synchronization.
    Prefix callbacks and callback state are not supported. Scan requires an
    associative operator and may regroup arithmetic. Floating-point prefixes
    can differ from a sequential fold; integer arithmetic uses the input dtype.

    See Also
    --------
    cuda.coop.cutlass.exclusive_scan
        Executable example of a partial logical warp and aggregate output.
    :cpp:struct:`cub::BlockScan`, :cpp:struct:`cub::WarpScan`
        C++ scan, sum, and aggregate overloads.

    Examples
    --------
    Compare seeded exclusive sums, inclusive maxima, and the sum helpers.
    One automatically synchronized scratch descriptor serves all four calls.

    The launcher accepts device pointers and a compile-time
    ``items_per_thread`` value.

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_qualified_scan_examples.py
        :language: python
        :start-after: # qualified-scan-example-begin
        :end-before: # qualified-scan-example-end
        :dedent: 4
    """
    from ._compiler._launch import current_kernel_launch_facts
    from ._operators import normalize_operator

    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_SCOPE}.scan group must be a ThreadGroup")
    if group.kind not in {"block", "warp", "threads_within_warp"}:
        raise NotImplementedError(
            f"{_SCOPE}.scan requires a block or warp group"
        )
    mode = _selector(mode, name="mode", choices={"inclusive", "exclusive"})
    if algorithm is not None:
        algorithm = _selector(algorithm, name="algorithm", choices=_ALGORITHMS)
    op = normalize_operator(scan_op, primitive="scan")
    if mode == "inclusive" and initial_value is not None:
        raise ValueError(
            f"{_SCOPE}.scan inclusive mode does not accept initial_value"
        )
    if mode == "exclusive" and op != "sum" and initial_value is None:
        raise ValueError(f"{_SCOPE}.scan exclusive {op} requires initial_value")
    if group.kind != "block":
        if algorithm is not None:
            raise ValueError(
                f"{_SCOPE}.scan algorithm selection requires a block"
            )
        if temp_storage is not None:
            raise ValueError(
                f"{_SCOPE}.scan temp_storage is supported only for blocks"
            )
    elif temp_storage is not None:
        _validate_temp_storage("scan", temp_storage)
    value = _coerce_thread_payload(
        value,
        scope=_SCOPE,
        primitive_name="scan",
        arg_name="value",
    )
    launch = current_kernel_launch_facts()
    _require_complete_warp_partition(
        group, feature="scan", exact_block_dim=launch.exact_block_dim
    )
    from ._lowering._scan import provider_scan

    return provider_scan(
        group=group,
        launch=launch,
        value=value,
        mode=mode,
        op=op,
        initial_value=initial_value,
        algorithm=algorithm,
        temp_storage=temp_storage,
        valid_items=valid_items,
        aggregate_output=aggregate_output,
    )


def exclusive_scan(
    group: BlockGroup | WarpGroup,
    value: CommonThreadDataLike[_ItemT]
    | CutlassTensorSample
    | CutlassTensorSSASample
    | _ItemT,
    /,
    *,
    scan_op: ScanOperator | Callable[[object, object], object] | None = None,
    initial_value: CommonNumericScalar | None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
    valid_items: ValidItems | None = None,
    aggregate_output: ThreadDataLike[Any] | None = None,
) -> ThreadData | _ItemT:
    """Return exclusive prefixes with an optional initial value and aggregate.

    Uses the parameters and return behavior of
    :func:`cuda.coop.cutlass.scan`, with exclusive mode fixed.
    Sum starts from zero unless ``initial_value`` is supplied; other built-in
    operators require it. ``aggregate_output`` receives the input aggregate
    without that initial value.

    Returns
    -------
    CuTe numeric scalar or cuda.coop.cutlass.ThreadData
        Prefix before each input item, with the input dtype. Payload input
        produces a fresh payload. For a partial warp, only ranks below
        ``valid_items`` have defined prefixes.

    Examples
    --------
    Scan five lanes of each eight-lane logical warp, starting from seven.
    Every lane participates and receives the aggregate. Only valid lanes
    write their prefixes. ``operator.add`` selects the built-in sum.

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_qualified_scan_examples.py
        :language: python
        :start-after: # qualified-exclusive-scan-example-begin
        :end-before: # qualified-exclusive-scan-example-end
        :dedent: 4
    """
    return scan(
        group,
        value,
        mode="exclusive",
        scan_op=scan_op,
        initial_value=initial_value,
        algorithm=algorithm,
        temp_storage=temp_storage,
        valid_items=valid_items,
        aggregate_output=aggregate_output,
    )


def inclusive_scan(
    group: BlockGroup | WarpGroup,
    value: CommonThreadDataLike[_ItemT]
    | CutlassTensorSample
    | CutlassTensorSSASample
    | _ItemT,
    /,
    *,
    scan_op: ScanOperator | Callable[[object, object], object] | None = None,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
    valid_items: ValidItems | None = None,
    aggregate_output: ThreadDataLike[Any] | None = None,
) -> ThreadData | _ItemT:
    """Return inclusive prefixes with a built-in operator.

    Uses the parameters and return behavior of
    :func:`cuda.coop.cutlass.scan`, with inclusive mode fixed.
    Each prefix includes its current item. This function has no
    ``initial_value`` parameter. The ``valid_items`` and
    ``aggregate_output`` controls follow :func:`cuda.coop.cutlass.scan`.

    Returns
    -------
    CuTe numeric scalar or cuda.coop.cutlass.ThreadData
        Inclusive prefixes with the input dtype. Payload input produces a
        fresh payload and remains unchanged. Read only valid lanes when
        ``valid_items`` is supplied.

    See Also
    --------
    cuda.coop.cutlass.exclusive_scan
        Partial-warp example. To include each current item, replace the call
        with ``inclusive_scan`` and remove ``initial_value``.

    Examples
    --------
    Compute running maxima alongside seeded and unseeded prefix sums.

    The launcher accepts device pointers and a compile-time
    ``items_per_thread`` value.

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_qualified_scan_examples.py
        :language: python
        :start-after: # qualified-scan-example-begin
        :end-before: # qualified-scan-example-end
        :dedent: 4
    """
    return scan(
        group,
        value,
        mode="inclusive",
        scan_op=scan_op,
        algorithm=algorithm,
        temp_storage=temp_storage,
        valid_items=valid_items,
        aggregate_output=aggregate_output,
    )


def exclusive_sum(
    group: BlockGroup | WarpGroup,
    value: CommonThreadDataLike[_ItemT]
    | CutlassTensorSample
    | CutlassTensorSSASample
    | _ItemT,
    /,
    *,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
    valid_items: ValidItems | None = None,
    aggregate_output: ThreadDataLike[Any] | None = None,
) -> ThreadData | _ItemT:
    """Return exclusive sums starting from zero.

    Uses the parameters and return behavior of
    :func:`cuda.coop.cutlass.scan`, with exclusive mode and sum
    fixed. Use :func:`cuda.coop.cutlass.exclusive_scan` for an explicit
    ``initial_value`` or a different built-in operator. The
    ``valid_items`` and ``aggregate_output`` controls are available here too.

    Returns
    -------
    CuTe numeric scalar or cuda.coop.cutlass.ThreadData
        Sum of the items preceding each input item, with the input dtype.
        Payload input produces a fresh payload and remains unchanged. Read
        only valid lanes when ``valid_items`` is supplied.

    See Also
    --------
    cuda.coop.cutlass.exclusive_scan
        Partial-warp example. For an exclusive sum starting at zero, replace
        the call with ``exclusive_sum`` and omit ``scan_op`` and
        ``initial_value``.

    Examples
    --------
    Compute prefixes before each item and compare them with inclusive sums
    and an exclusive scan starting from seven.

    The launcher accepts device pointers and a compile-time
    ``items_per_thread`` value.

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_qualified_scan_examples.py
        :language: python
        :start-after: # qualified-scan-example-begin
        :end-before: # qualified-scan-example-end
        :dedent: 4
    """
    return scan(
        group,
        value,
        mode="exclusive",
        algorithm=algorithm,
        temp_storage=temp_storage,
        valid_items=valid_items,
        aggregate_output=aggregate_output,
    )


def inclusive_sum(
    group: BlockGroup | WarpGroup,
    value: CommonThreadDataLike[_ItemT]
    | CutlassTensorSample
    | CutlassTensorSSASample
    | _ItemT,
    /,
    *,
    algorithm: ScanAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
    valid_items: ValidItems | None = None,
    aggregate_output: ThreadDataLike[Any] | None = None,
) -> ThreadData | _ItemT:
    """Return inclusive sums, including each current item.

    Uses the parameters and return behavior of
    :func:`cuda.coop.cutlass.scan`, with inclusive mode and sum
    fixed. The ``valid_items`` and ``aggregate_output`` controls
    follow that function.

    Returns
    -------
    CuTe numeric scalar or cuda.coop.cutlass.ThreadData
        Sum through each input item, with the input dtype. Payload input
        produces a fresh payload and remains unchanged. Read only valid
        lanes when ``valid_items`` is supplied.

    See Also
    --------
    cuda.coop.cutlass.inclusive_scan
        Inclusive prefixes with other built-in operators.
    cuda.coop.cutlass.exclusive_scan
        Partial-warp example. For an inclusive sum, replace the call with
        ``inclusive_sum`` and omit ``scan_op`` and ``initial_value``.

    Examples
    --------
    Compute prefixes through each item alongside exclusive sums and
    running maxima.

    The launcher accepts device pointers and a compile-time
    ``items_per_thread`` value.

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_qualified_scan_examples.py
        :language: python
        :start-after: # qualified-scan-example-begin
        :end-before: # qualified-scan-example-end
        :dedent: 4
    """
    return scan(
        group,
        value,
        mode="inclusive",
        algorithm=algorithm,
        temp_storage=temp_storage,
        valid_items=valid_items,
        aggregate_output=aggregate_output,
    )


__all__ = [
    "exclusive_scan",
    "exclusive_sum",
    "inclusive_scan",
    "inclusive_sum",
    "scan",
]
