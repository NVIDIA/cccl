# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compatibility imports for CUTLASS ThreadData helpers."""

from .._internal._thread_data import (
    _UNSET,
    ThreadData,
    _coerce_thread_payload,
    _is_register_fragment,
    _is_thread_payload_candidate,
    _normalize_index_int,
    _validate_items_per_thread,
)

__all__ = [
    "ThreadData",
    "_UNSET",
    "_coerce_thread_payload",
    "_is_register_fragment",
    "_is_thread_payload_candidate",
    "_normalize_index_int",
    "_validate_items_per_thread",
]
