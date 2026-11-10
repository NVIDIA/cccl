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
    exclusive_scan_impl,
    exclusive_sum_impl,
    inclusive_scan_impl,
    inclusive_sum_impl,
    reduce_impl,
    scan_impl,
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
        _registered = True
