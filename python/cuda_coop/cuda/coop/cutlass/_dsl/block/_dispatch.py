# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from typing import Any, Mapping

from .._dispatch import (
    PrimitiveDispatcher,
    PrimitiveImpl,
    dispatch_single_phase_primitive,
)
from .._scope import BLOCK_SCOPE as _SCOPE
from ._single_phase import (
    coerce_thread_payloads_to_thread_data,
    dispatch_thread_data_aware,
    extract_single_phase_context,
    register_temp_storage_use,
)

_DISPATCHER = PrimitiveDispatcher(scope=_SCOPE, package=__package__)
_IMPLS = _DISPATCHER.impls


def register_primitive_impl(
    primitive_name: str,
    *,
    impl: PrimitiveImpl,
) -> None:
    _DISPATCHER.register_primitive_impl(primitive_name, impl=impl)


def dispatch_primitive(
    primitive_name: str,
    *,
    kwargs: Mapping[str, Any] | None = None,
) -> Any:
    return dispatch_single_phase_primitive(
        _DISPATCHER,
        primitive_name,
        kwargs=kwargs,
        extract_single_phase_context=extract_single_phase_context,
        coerce_thread_payloads_to_thread_data=coerce_thread_payloads_to_thread_data,
        register_temp_storage_use=register_temp_storage_use,
        dispatch_thread_data_aware=dispatch_thread_data_aware,
    )
