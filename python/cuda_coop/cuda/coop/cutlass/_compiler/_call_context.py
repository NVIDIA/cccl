# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Transactional per-call state shared by CUTLASS family lowerers."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from .._temp_storage import TempStorageBase
from .._thread_data import ThreadData


@dataclass(frozen=True)
class SinglePhaseContext:
    thread_data: ThreadData | None
    temp_storage: TempStorageBase | None


_ACTIVE_SINGLE_PHASE_CONTEXT: ContextVar[SinglePhaseContext | None] = ContextVar(
    "cuda_coop_cutlass_active_single_phase_context",
    default=None,
)


@contextmanager
def activate_single_phase_context(context: SinglePhaseContext):
    token = _ACTIVE_SINGLE_PHASE_CONTEXT.set(context)
    try:
        yield
    finally:
        _ACTIVE_SINGLE_PHASE_CONTEXT.reset(token)


def get_active_single_phase_context() -> SinglePhaseContext | None:
    return _ACTIVE_SINGLE_PHASE_CONTEXT.get()


@contextmanager
def single_phase_transaction(
    single_phase_context: SinglePhaseContext,
    *,
    snapshot_provider_session: Callable[[], Any],
    restore_provider_session: Callable[[Any], None],
) -> Iterator[None]:
    """Roll back scratch planning and provider registration on failure."""

    temp_storage = single_phase_context.temp_storage
    temp_storage_snapshot = (
        temp_storage._snapshot_uses() if temp_storage is not None else None
    )
    provider_session_snapshot = snapshot_provider_session()
    try:
        yield
    except Exception:
        if temp_storage is not None and temp_storage_snapshot is not None:
            temp_storage._restore_uses(temp_storage_snapshot)
        restore_provider_session(provider_session_snapshot)
        raise
