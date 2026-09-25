# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Built-in group Scan and Sum entry points for CuTe kernels."""

from enum import Enum

from cuda.coop._core.api._payload import _validate_common_temp_storage
from cuda.coop._core.thread_group import ThreadGroup

from ._thread_data import _coerce_thread_payload
from ._thread_group import _require_complete_warp_partition

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
    group,
    value,
    /,
    *,
    mode="exclusive",
    scan_op=None,
    initial_value=None,
    algorithm=None,
    temp_storage=None,
    valid_items=None,
    aggregate_output=None,
):
    """Scan register values with optional valid-prefix and aggregate controls.

    Extends :func:`cuda.coop.scan` with the operand forms and controls below.
    Group requirements, scan order, modes, algorithms, and temporary storage
    follow the common function.

    Parameters
    ----------
    value : numeric scalar, ThreadData, CuTe register tensor, or TensorSSA
        Blocks accept scalars or fixed-size per-thread payloads in blocked
        order. Register tensors and ``TensorSSA`` values are converted with
        :meth:`cuda.coop.cutlass.ThreadData.from_payload`. All members must use
        the same dtype and extent. Physical and logical warps accept scalars
        only, including when each thread would otherwise hold one item.
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
        representable in that dtype. Inclusive mode requires ``None``.
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
    Only blocks accept ``algorithm`` and explicit ``temp_storage``. Automatic
    trailing synchronization protects scratch reuse; disabling it on a
    :class:`cuda.coop.TempStorage` requires explicit block barriers.
    Prefix callbacks and callback state are not supported.

    See Also
    --------
    cuda.coop.scan
        Shared scan order, algorithms, and storage contract.
    cuda.coop.cutlass.exclusive_scan
        Executable example of a partial logical warp and aggregate output.
    :cpp:struct:`cub::BlockScan`, :cpp:struct:`cub::WarpScan`
        C++ scan, sum, and aggregate overloads.
    """
    from ._compiler._launch import current_kernel_launch_facts
    from ._operators import normalize_operator

    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_SCOPE}.scan group must be a ThreadGroup")
    if group.kind not in {"block", "warp", "threads_within_warp"}:
        raise NotImplementedError(f"{_SCOPE}.scan requires a block or warp group")
    mode = _selector(mode, name="mode", choices={"inclusive", "exclusive"})
    if algorithm is not None:
        algorithm = _selector(algorithm, name="algorithm", choices=_ALGORITHMS)
    op = normalize_operator(scan_op, primitive="scan")
    if mode == "inclusive" and initial_value is not None:
        raise ValueError(f"{_SCOPE}.scan inclusive mode does not accept initial_value")
    if mode == "exclusive" and op != "sum" and initial_value is None:
        raise ValueError(f"{_SCOPE}.scan exclusive {op} requires initial_value")
    if group.kind != "block":
        if algorithm is not None:
            raise ValueError(f"{_SCOPE}.scan algorithm selection requires a block")
        if temp_storage is not None:
            raise ValueError(f"{_SCOPE}.scan temp_storage is supported only for blocks")
    elif temp_storage is not None:
        _validate_common_temp_storage("scan", temp_storage)
    value = _coerce_thread_payload(
        value,
        scope=_SCOPE,
        primitive_name="scan",
        arg_name="value",
        common_root_payload_kind="scalar_or_thread_data",
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
    group,
    value,
    /,
    *,
    scan_op=None,
    initial_value=None,
    algorithm=None,
    temp_storage=None,
    valid_items=None,
    aggregate_output=None,
):
    """Return exclusive prefixes with an optional initial value and aggregate.

    Extends :func:`cuda.coop.exclusive_scan` with the parameters and return
    behavior of :func:`cuda.coop.cutlass.scan`, with exclusive mode fixed.
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

    .. literalinclude:: ../../python/cuda_coop/tests/backends/cutlass/runtime/test_qualified_collective_examples.py
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
    group,
    value,
    /,
    *,
    scan_op=None,
    algorithm=None,
    temp_storage=None,
    valid_items=None,
    aggregate_output=None,
):
    """Return inclusive prefixes with a built-in operator.

    Extends :func:`cuda.coop.inclusive_scan` with the parameters and return
    behavior of :func:`cuda.coop.cutlass.scan`, with inclusive mode fixed.
    Each prefix includes its current item. This function has no
    ``initial_value`` parameter. The qualified ``valid_items`` and
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
    group,
    value,
    /,
    *,
    algorithm=None,
    temp_storage=None,
    valid_items=None,
    aggregate_output=None,
):
    """Return exclusive sums starting from zero.

    Extends :func:`cuda.coop.exclusive_sum` with the parameters and return
    behavior of :func:`cuda.coop.cutlass.scan`, with exclusive mode and sum
    fixed. Use :func:`cuda.coop.cutlass.exclusive_scan` for an explicit
    ``initial_value`` or a different built-in operator. The qualified
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
    group,
    value,
    /,
    *,
    algorithm=None,
    temp_storage=None,
    valid_items=None,
    aggregate_output=None,
):
    """Return inclusive sums, including each current item.

    Extends :func:`cuda.coop.inclusive_sum` with the parameters and return
    behavior of :func:`cuda.coop.cutlass.scan`, with inclusive mode and sum
    fixed. The qualified ``valid_items`` and ``aggregate_output`` controls
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


__all__ = ["scan", "exclusive_scan", "inclusive_scan", "exclusive_sum", "inclusive_sum"]
