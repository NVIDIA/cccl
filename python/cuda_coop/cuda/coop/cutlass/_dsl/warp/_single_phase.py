# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from typing import Any

from .. import _single_phase
from .._scope import WARP_SCOPE as _SCOPE
from .._temp_storage import TempStorageBase


class TempStorage(TempStorageBase):
    """Identity-shared scratch planner for warp-scoped CUTLASS collectives."""

    scope = "cuda.coop.cutlass._warp"


def extract_single_phase_context(
    primitive_name: str,
    kwargs: dict[str, Any],
    *,
    reserve_context_fields: bool = False,
) -> _single_phase.SinglePhaseContext:
    return _single_phase.extract_single_phase_context(
        primitive_name,
        kwargs,
        scope=_SCOPE,
        temp_storage_type=TempStorage,
        reserve_context_fields=reserve_context_fields,
    )


def register_temp_storage_use(
    primitive_name: str,
    context: _single_phase.SinglePhaseContext,
    _kwargs: dict[str, Any],
) -> None:
    if context.temp_storage is not None:
        raise NotImplementedError(
            f"{_SCOPE}.{primitive_name} does not support explicit "
            "TempStorage planning yet"
        )


def coerce_thread_payloads_to_thread_data(
    primitive_name: str,
    kwargs: dict[str, Any],
) -> None:
    _single_phase.coerce_thread_payloads_to_thread_data(
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
    return _single_phase.dispatch_thread_data_aware(
        primitive_name,
        impl,
        kwargs,
        scope=_SCOPE,
        strip_launch_metadata=strip_launch_metadata,
    )
