# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from typing import Any

from ..._temp_storage import TempStorage
from .._scope import BLOCK_SCOPE as _SCOPE
from .._single_phase import (
    SinglePhaseContext,
)
from .._single_phase import (
    coerce_thread_payloads_to_thread_data as _coerce_thread_payloads_to_thread_data,
)
from .._single_phase import dispatch_thread_data_aware as _dispatch_thread_data_aware
from .._single_phase import (
    extract_single_phase_context as _extract_single_phase_context,
)
from .._temp_storage import (
    register_block_temp_storage_use as _register_block_temp_storage_use,
)


def extract_single_phase_context(
    primitive_name: str,
    kwargs: dict[str, Any],
    *,
    reserve_context_fields: bool = False,
) -> SinglePhaseContext:
    return _extract_single_phase_context(
        primitive_name,
        kwargs,
        scope=_SCOPE,
        temp_storage_type=TempStorage,
        reserve_context_fields=reserve_context_fields,
    )


def register_temp_storage_use(
    primitive_name: str,
    context: SinglePhaseContext,
    kwargs: dict[str, Any],
) -> None:
    _register_block_temp_storage_use(
        primitive_name,
        context,
        kwargs,
        scope=_SCOPE,
    )


def coerce_thread_payloads_to_thread_data(
    primitive_name: str,
    kwargs: dict[str, Any],
) -> None:
    _coerce_thread_payloads_to_thread_data(
        primitive_name,
        kwargs,
        scope=_SCOPE,
    )


coerce_register_fragments_to_thread_data = coerce_thread_payloads_to_thread_data


def dispatch_thread_data_aware(
    primitive_name: str,
    impl,
    kwargs: dict[str, Any],
    *,
    strip_launch_metadata: bool = False,
) -> Any:
    return _dispatch_thread_data_aware(
        primitive_name,
        impl,
        kwargs,
        scope=_SCOPE,
        strip_launch_metadata=strip_launch_metadata,
    )
