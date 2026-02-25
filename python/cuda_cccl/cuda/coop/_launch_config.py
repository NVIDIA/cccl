# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import contextlib
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from typing import Protocol

    class LaunchConfig(Protocol):
        args: Sequence[Any]
        dispatcher: Any
        blockdim: Any
        sharedmem: int
        pre_launch_callbacks: list[Any]

        def mark_kernel_as_launch_config_sensitive(self) -> None: ...
else:
    LaunchConfig = Any


def _get_env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, None)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


try:
    from numba.cuda.launchconfig import (
        current_launch_config as _current_launch_config_impl,
    )
    from numba.cuda.launchconfig import (
        ensure_current_launch_config as _ensure_current_launch_config_impl,
    )
except ModuleNotFoundError:
    _current_launch_config_impl = None
    _ensure_current_launch_config_impl = None


LAUNCH_CONFIG_AVAILABLE = (
    _current_launch_config_impl is not None
    and _ensure_current_launch_config_impl is not None
)
LAUNCH_CONFIG_ENABLED = _get_env_bool(
    "NUMBA_CCCL_COOP_ENABLE_LAUNCH_CONFIG",
    default=True,
)


def is_launch_config_available() -> bool:
    return LAUNCH_CONFIG_AVAILABLE


def is_launch_config_enabled() -> bool:
    return LAUNCH_CONFIG_ENABLED


def is_launch_config_active() -> bool:
    return LAUNCH_CONFIG_AVAILABLE and LAUNCH_CONFIG_ENABLED


def set_launch_config_enabled(enabled: bool) -> None:
    global LAUNCH_CONFIG_ENABLED
    LAUNCH_CONFIG_ENABLED = bool(enabled)


def reset_launch_config_enabled() -> None:
    global LAUNCH_CONFIG_ENABLED
    LAUNCH_CONFIG_ENABLED = _get_env_bool(
        "NUMBA_CCCL_COOP_ENABLE_LAUNCH_CONFIG",
        default=True,
    )


@contextlib.contextmanager
def temporary_launch_config_enabled(enabled: bool) -> "Iterator[None]":
    old = LAUNCH_CONFIG_ENABLED
    set_launch_config_enabled(enabled)
    try:
        yield
    finally:
        set_launch_config_enabled(old)


def current_launch_config() -> "LaunchConfig | None":
    if not is_launch_config_active():
        return None
    return _current_launch_config_impl()


def ensure_current_launch_config() -> "LaunchConfig | None":
    if not is_launch_config_active():
        return None
    return _ensure_current_launch_config_impl()
