# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Group-first Exchange entry point for CuTe kernels."""

from enum import Enum

from cuda.coop._core import GroupExchangeMode
from cuda.coop._core.thread_group import ThreadGroup

from ._thread_data import ThreadData, _coerce_thread_payload
from ._thread_group import (
    _require_complete_warp_partition,
    _resolve_primitive_group_from_launch,
)

_SCOPE = "cuda.coop.cutlass"
_BLOCK_MODES = frozenset(mode.value for mode in GroupExchangeMode)
_WARP_MODES = frozenset({"striped_to_blocked", "blocked_to_striped"})


def _normalize_exchange_mode(mode, *, group_kind):
    if not isinstance(mode, str) or isinstance(mode, Enum):
        raise TypeError(f"{_SCOPE}.exchange mode must be a compile-time string")
    token = mode.strip().lower().replace("-", "_")
    allowed = _BLOCK_MODES if group_kind == "block" else _WARP_MODES
    if token not in allowed:
        raise ValueError(
            f"{_SCOPE}.exchange mode for {group_kind} groups must be one of: "
            + ", ".join(sorted(allowed))
        )
    return GroupExchangeMode(token)


def _payload(value, *, name):
    value = _coerce_thread_payload(
        value,
        scope=_SCOPE,
        primitive_name="exchange",
        arg_name=name,
        common_root_payload_kind="thread_data",
    )
    if not isinstance(value, ThreadData):
        raise TypeError(f"{_SCOPE}.exchange {name} must be a fixed-size ThreadData")
    return value


def exchange(
    group,
    value,
    /,
    *,
    mode="striped_to_blocked",
    ranks=None,
    valid_flags=None,
    warp_time_slicing=False,
):
    """Exchange register payloads, including ranked block scatters.

    See :func:`cuda.coop.exchange` for the shared group requirements, layout
    definitions, and dtypes. The qualified API adds CuTe register payloads,
    block scatter modes, and the controls below. Every group member must call,
    including members whose scatter items are all invalid.

    Parameters
    ----------
    value : ThreadData, CuTe register tensor, or TensorSSA
        Fixed-size per-thread payload. Register tensors and ``TensorSSA``
        values are converted with :meth:`cuda.coop.cutlass.ThreadData.from_payload`.
        Every member must supply the same dtype and extent.
    mode : str, optional
        Compile-time conversion, default ``"striped_to_blocked"``. Blocks
        also support ``"blocked_to_warp_striped"`` and
        ``"warp_striped_to_blocked"``, which stripe within each physical warp
        and require a block size divisible by 32. Block scatter modes are
        ``"scatter_to_blocked"``, ``"scatter_to_striped"``,
        ``"scatter_to_striped_guarded"``, and ``"scatter_to_striped_flagged"``.
        Physical and logical warps support only ``"striped_to_blocked"`` and
        ``"blocked_to_striped"``; their enclosing block must contain complete
        physical warps.
    ranks : ThreadData, CuTe register tensor, or TensorSSA, optional
        Destination ranks, required for every scatter mode and rejected for
        layout conversions. Must have a signed integer dtype and the same
        per-thread extent as ``value``. Valid ranks must be distinct across
        the block and lie in ``[0, block_tile_size)``. Guarded scatter skips
        negative ranks. Other scatter modes require each participating item's
        rank to be in range. ``None`` is the default.
    valid_flags : ThreadData, CuTe register tensor, or TensorSSA, optional
        Per-item flags, required for ``"scatter_to_striped_flagged"`` and
        rejected for other modes. Must have an integer, non-boolean dtype and
        the same extent as ``value``. Only items with nonzero flags scatter;
        their ranks must be distinct and in range. ``None`` is the default.
    warp_time_slicing : bool, optional
        Compile-time option, default ``False``. Reuse block exchange scratch
        across warps to reduce shared memory usage at the cost of additional
        synchronization. Requires a block; guarded and flagged scatter do
        not support it.

    Returns
    -------
    cuda.coop.cutlass.ThreadData
        New writable payload with the input dtype, extent, and requested
        alignment in the output layout. All inputs, ranks, and flags remain
        unchanged. A scatter destination with no valid input is undefined;
        initialize it before reading it.

    Notes
    -----
    Scratch allocation and trailing synchronization are automatic. See
    :ref:`coop-temp-storage` for storage reuse and :ref:`coop-thread-groups`
    for participation requirements.

    See Also
    --------
    cuda.coop.exchange
        Shared blocked and striped layout conversions.
    :cpp:struct:`cub::BlockExchange`, :cpp:struct:`cub::WarpExchange`
        C++ exchange primitives.

    Examples
    --------
    Reverse a 128-item block tile by assigning each input its destination
    rank. Each of the 64 threads holds two items.

    .. literalinclude:: ../../python/cuda_coop/tests/backends/cutlass/runtime/test_qualified_collective_examples.py
        :language: python
        :start-after: # qualified-scatter-example-begin
        :end-before: # qualified-scatter-example-end
        :dedent: 4
    """
    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_SCOPE}.exchange group must be a ThreadGroup")
    if group.kind not in {"block", "warp", "threads_within_warp"}:
        raise NotImplementedError(f"{_SCOPE}.exchange requires a block or warp group")
    mode = _normalize_exchange_mode(mode, group_kind=group.kind)
    if not isinstance(warp_time_slicing, bool):
        raise TypeError(
            f"{_SCOPE}.exchange warp_time_slicing must be a compile-time bool"
        )
    if warp_time_slicing and group.kind != "block":
        raise ValueError(f"{_SCOPE}.exchange warp_time_slicing applies only to blocks")
    if mode.uses_ranks != (ranks is not None):
        requirement = "requires" if mode.uses_ranks else "does not accept"
        raise ValueError(f"{_SCOPE}.exchange {mode.value} {requirement} ranks")
    if mode.uses_valid_flags != (valid_flags is not None):
        requirement = "requires" if mode.uses_valid_flags else "does not accept"
        raise ValueError(f"{_SCOPE}.exchange {mode.value} {requirement} valid_flags")
    value = _payload(value, name="value")
    ranks = None if ranks is None else _payload(ranks, name="ranks")
    valid_flags = (
        None if valid_flags is None else _payload(valid_flags, name="valid_flags")
    )

    from ._compiler._launch import current_kernel_launch_facts
    from ._lowering import _exchange

    launch = current_kernel_launch_facts()
    group = _resolve_primitive_group_from_launch(group, launch, feature="exchange")
    _require_complete_warp_partition(
        group, feature="exchange", exact_block_dim=launch.exact_block_dim
    )
    value_type, rank_type, flag_type = _exchange._resolve_exchange_operand_types(
        value=value, ranks=ranks, valid_flags=valid_flags
    )
    plan = _exchange._make_group_exchange_plan(
        group=group,
        launch=launch,
        dtype=value_type,
        items_per_thread=value.items_per_thread,
        mode=mode.value,
        rank_dtype=rank_type,
        valid_flag_dtype=flag_type,
        warp_time_slicing=warp_time_slicing,
    ).require_supported()
    return _exchange.provider_exchange(
        plan=plan, value=value, ranks=ranks, valid_flags=valid_flags
    )


__all__ = ["exchange"]
