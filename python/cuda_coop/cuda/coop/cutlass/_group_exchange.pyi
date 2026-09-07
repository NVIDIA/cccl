# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing declarations for qualified CUTLASS exchanges."""

from __future__ import annotations

from typing import Any, Literal, overload

from ._types import ThreadData
from ._types import _BlockExchangeMode as _BlockExchangeMode
from ._types import _BlockGroup as _BlockGroup
from ._types import _CutlassNumericT as _CutlassNumericT
from ._types import _CutlassTensorSample as _CutlassTensorSample
from ._types import _CutlassTensorSSASample as _CutlassTensorSSASample
from ._types import _WarpExchangeMode as _WarpExchangeMode
from ._types import _WarpGroup as _WarpGroup

@overload
def exchange(
    group: _BlockGroup,
    value: ThreadData[_CutlassNumericT],
    /,
    *,
    mode: _BlockExchangeMode = "striped_to_blocked",
    ranks: ThreadData | None = None,
    valid_flags: ThreadData | None = None,
    warp_time_slicing: bool = False,
) -> ThreadData[_CutlassNumericT]:
    """Return a layout-rearranged ``ThreadData`` payload without mutation.

    The overload set accepts complete blocks, physical warps, and logical warps.
    The portable modes are ``"striped_to_blocked"`` and
    ``"blocked_to_striped"``. Blocks additionally support warp-striped and
    scatter modes; warp groups support scatter-to-striped. Scatter modes consume
    ``ranks``, flagged block scatter also consumes ``valid_flags``, and
    ``warp_time_slicing`` is block-only.
    ``value`` must own one through five items per participant; scalar inputs are
    not supported.
    The result preserves the input payload's shape and item type.
    """

@overload
def exchange(
    group: _WarpGroup,
    value: ThreadData[_CutlassNumericT],
    /,
    *,
    mode: _WarpExchangeMode = "striped_to_blocked",
    ranks: ThreadData | None = None,
    valid_flags: None = None,
    warp_time_slicing: Literal[False] = False,
) -> ThreadData[_CutlassNumericT]:
    """Exchange a ``ThreadData`` payload across a physical or logical warp.

    Warp groups support blocked-to-striped, striped-to-blocked, and
    scatter-to-striped layouts. Scatter-to-striped consumes ``ranks``. Warp
    exchange does not accept ``valid_flags`` or ``warp_time_slicing``.
    """

@overload
def exchange(
    group: _BlockGroup,
    value: _CutlassTensorSample | _CutlassTensorSSASample,
    /,
    *,
    mode: _BlockExchangeMode = "striped_to_blocked",
    ranks: ThreadData | None = None,
    valid_flags: ThreadData | None = None,
    warp_time_slicing: bool = False,
) -> ThreadData[Any]:
    """Exchange one CUTLASS register tensor across ``group``.

    ``value`` is an rmem ``Tensor`` or ``TensorSSA`` with one through five static
    items per member. ``mode`` selects the qualified layout conversion, ``ranks``
    supplies scatter destinations, ``valid_flags`` guards flagged scatter, and
    ``warp_time_slicing`` selects the time-sliced specialization. The result is
    flattened ``ThreadData``; its external element type is statically ``Any``.
    """

@overload
def exchange(
    group: _WarpGroup,
    value: _CutlassTensorSample | _CutlassTensorSSASample,
    /,
    *,
    mode: _WarpExchangeMode = "striped_to_blocked",
    ranks: ThreadData | None = None,
    valid_flags: None = None,
    warp_time_slicing: Literal[False] = False,
) -> ThreadData[Any]:
    """Exchange one CUTLASS register tensor across physical or logical ``group``.

    ``value`` is an rmem ``Tensor`` or ``TensorSSA`` with one through five
    static items per member. ``mode`` selects the warp layout and
    scatter-to-striped consumes ``ranks``. Warp exchange does not accept
    ``valid_flags`` or ``warp_time_slicing``.
    """

__all__ = [
    "exchange",
]
