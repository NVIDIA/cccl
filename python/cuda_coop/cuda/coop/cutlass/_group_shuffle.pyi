# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing declarations for qualified CUTLASS shuffles."""

from __future__ import annotations

from typing import Any, Literal, overload

from ._types import ThreadData
from ._types import _BlockGroup as _BlockGroup
from ._types import _CutlassNumericT as _CutlassNumericT
from ._types import _CutlassTensorSample as _CutlassTensorSample
from ._types import _CutlassTensorSSASample as _CutlassTensorSSASample
from ._types import _ScalarT as _ScalarT

@overload
def shuffle(
    group: _BlockGroup,
    value: ThreadData[_CutlassNumericT],
    /,
    *,
    mode: Literal["down"] = "down",
    distance: Literal[1] = 1,
    block_prefix: ThreadData[_CutlassNumericT] | None = None,
    block_suffix: None = None,
) -> ThreadData[_CutlassNumericT]:
    """Return a downward unit shift with an optional CUTLASS prefix output.

    ``group`` must be a complete physical block and ``value`` a fixed-size
    register payload. ``mode`` is ``"down"``: the final flattened result item
    is undefined, and the first input item may be written to ``block_prefix``.
    A supplied ``block_prefix`` must contain exactly one item.
    ``block_suffix`` must remain ``None`` and ``distance`` must be ``1``.
    """

@overload
def shuffle(
    group: _BlockGroup,
    value: ThreadData[_CutlassNumericT],
    /,
    *,
    mode: Literal["up"],
    distance: Literal[1] = 1,
    block_prefix: None = None,
    block_suffix: ThreadData[_CutlassNumericT] | None = None,
) -> ThreadData[_CutlassNumericT]:
    """Return an upward unit shift with an optional CUTLASS suffix output.

    ``group`` must be a complete physical block and ``value`` a fixed-size
    register payload. ``mode`` is ``"up"``: the first flattened result item is
    undefined, and the final input item may be written to ``block_suffix``.
    A supplied ``block_suffix`` must contain exactly one item.
    ``block_prefix`` must remain ``None`` and ``distance`` must be ``1``.
    """

@overload
def shuffle(
    group: _BlockGroup,
    value: _CutlassTensorSample | _CutlassTensorSSASample,
    /,
    *,
    mode: Literal["down"] = "down",
    distance: Literal[1] = 1,
    block_prefix: ThreadData[Any] | None = None,
    block_suffix: None = None,
) -> ThreadData[Any]:
    """Return a downward unit shift of a CUTLASS register tensor.

    ``group`` is a complete block and ``value`` supplies a static rmem or SSA
    payload. ``mode`` is ``"down"``, ``distance`` is one, ``block_prefix`` may
    receive the first input item, and ``block_suffix`` remains ``None``. The
    flattened result element type is statically ``Any``.
    """

@overload
def shuffle(
    group: _BlockGroup,
    value: _CutlassTensorSample | _CutlassTensorSSASample,
    /,
    *,
    mode: Literal["up"],
    distance: Literal[1] = 1,
    block_prefix: None = None,
    block_suffix: ThreadData[Any] | None = None,
) -> ThreadData[Any]:
    """Return an upward unit shift of a CUTLASS register tensor.

    ``group`` is a complete block and ``value`` supplies a static rmem or SSA
    payload. ``mode`` is ``"up"``, ``distance`` is one, ``block_suffix`` may
    receive the final input item, and ``block_prefix`` remains ``None``. The
    flattened result element type is statically ``Any``.
    """

@overload
def shuffle(
    group: _BlockGroup,
    value: _ScalarT,
    /,
    *,
    mode: Literal["offset", "rotate"],
    distance: int = 1,
    block_prefix: None = None,
    block_suffix: None = None,
) -> _ScalarT:
    """Return one CUTLASS scalar selected by Offset or Rotate.

    ``group`` must be a complete physical block and ``value`` one scalar.
    ``mode`` set to ``"offset"`` accepts a signed ``distance``;
    ``mode="rotate"`` accepts a nonnegative distance. Scalar calls do not
    accept ``block_prefix`` or ``block_suffix``.
    """

__all__ = [
    "shuffle",
]
