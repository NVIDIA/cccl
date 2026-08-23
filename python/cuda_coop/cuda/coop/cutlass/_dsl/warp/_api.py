# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from typing import Any, Callable

from ._dispatch import register_primitive_impl

PrimitiveImpl = Callable[..., Any]


def register_provider_impl(primitive_name: str, impl: PrimitiveImpl) -> None:
    """Register a provider-backed implementation for a warp primitive."""
    register_primitive_impl(primitive_name, impl=impl)
