# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import Any

from cuda.tile._bytecode.version import BytecodeVersion
from cuda.tile._ir.ops import tile_impl_registry

from . import _api
from ._lowerings import (
    adjacent_difference_subtract_left_impl,
    adjacent_difference_subtract_right_impl,
    discontinuity_flag_heads_and_tails_impl,
    discontinuity_flag_heads_impl,
    discontinuity_flag_tails_impl,
    exchange_blocked_to_striped_impl,
    exchange_striped_to_blocked_impl,
    exclusive_scan_impl,
    exclusive_sum_impl,
    histogram_256_impl,
    histogram_impl,
    inclusive_scan_impl,
    inclusive_sum_impl,
    reduce_impl,
    run_length_decode_impl,
    run_length_decode_with_offsets_impl,
    scan_impl,
    shuffle_down_impl,
    shuffle_offset_impl,
    shuffle_rotate_impl,
    shuffle_up_impl,
    sum_impl,
)

_registered = False
_register_lock = Lock()
impl = tile_impl_registry.impl
_MIN_TILEIR_BYTECODE_VERSION = BytecodeVersion.V_13_5


def _register_tilelib_impl(
    stub: Callable[..., Any], lowering: Callable[..., Any]
) -> None:
    impl(stub, min_version=_MIN_TILEIR_BYTECODE_VERSION)(lowering)


def register() -> None:
    """Install public-marker lowerings into cuTile's process-local registry.

    The lock prevents concurrent first access from registering the same set
    twice. Every entry requires TileIR bytecode 13.5 or newer. Mark the set
    registered only after all entries have been installed. Registration
    installs handlers without compiling a TileLib or launching a kernel.
    """

    global _registered
    if _registered:
        return
    with _register_lock:
        if _registered:
            return
        _register_tilelib_impl(_api.exclusive_sum, exclusive_sum_impl)
        _register_tilelib_impl(_api.inclusive_sum, inclusive_sum_impl)
        _register_tilelib_impl(_api.scan, scan_impl)
        _register_tilelib_impl(_api.exclusive_scan, exclusive_scan_impl)
        _register_tilelib_impl(_api.inclusive_scan, inclusive_scan_impl)
        _register_tilelib_impl(_api.reduce, reduce_impl)
        _register_tilelib_impl(_api.sum, sum_impl)
        _register_tilelib_impl(_api.histogram_256, histogram_256_impl)
        _register_tilelib_impl(_api.histogram, histogram_impl)
        _register_tilelib_impl(_api.run_length_decode, run_length_decode_impl)
        _register_tilelib_impl(
            _api.run_length_decode_with_offsets,
            run_length_decode_with_offsets_impl,
        )
        _register_tilelib_impl(
            _api.discontinuity_flag_heads, discontinuity_flag_heads_impl
        )
        _register_tilelib_impl(
            _api.discontinuity_flag_tails, discontinuity_flag_tails_impl
        )
        _register_tilelib_impl(
            _api.discontinuity_flag_heads_and_tails,
            discontinuity_flag_heads_and_tails_impl,
        )
        _register_tilelib_impl(
            _api.adjacent_difference_subtract_left,
            adjacent_difference_subtract_left_impl,
        )
        _register_tilelib_impl(
            _api.adjacent_difference_subtract_right,
            adjacent_difference_subtract_right_impl,
        )
        _register_tilelib_impl(
            _api.exchange_striped_to_blocked, exchange_striped_to_blocked_impl
        )
        _register_tilelib_impl(
            _api.exchange_blocked_to_striped, exchange_blocked_to_striped_impl
        )
        _register_tilelib_impl(_api.shuffle_up, shuffle_up_impl)
        _register_tilelib_impl(_api.shuffle_down, shuffle_down_impl)
        _register_tilelib_impl(_api.shuffle_offset, shuffle_offset_impl)
        _register_tilelib_impl(_api.shuffle_rotate, shuffle_rotate_impl)
        _registered = True
