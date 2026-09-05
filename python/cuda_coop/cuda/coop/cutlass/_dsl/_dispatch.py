# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Primitive dispatch registry for CuTe cooperative scopes."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Mapping

from ._single_phase import SinglePhaseContext, activate_single_phase_context

PrimitiveImpl = Callable[..., Any]
ExtractSinglePhaseContext = Callable[..., SinglePhaseContext]
CoerceThreadPayloads = Callable[[str, dict[str, Any]], None]
RegisterTempStorageUse = Callable[[str, SinglePhaseContext, dict[str, Any]], None]
DispatchThreadDataAware = Callable[..., Any]

_DEBUG_DISPATCH_ENV = "CUDA_COOP_CUTLASS_DEBUG_DISPATCH"
_DEBUG_ENABLED = os.environ.get(_DEBUG_DISPATCH_ENV, "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


class PrimitiveDispatcher:
    def __init__(self, *, scope: str, package: str):
        self.scope = scope
        self.package = package
        self.impls: dict[str, PrimitiveImpl] = {}

    def debug_log(self, message: str) -> None:
        if _DEBUG_ENABLED:
            print(f"[{self.scope}] {message}", file=sys.stderr)

    def register_primitive_impl(
        self,
        primitive_name: str,
        *,
        impl: PrimitiveImpl,
    ) -> None:
        if not isinstance(primitive_name, str) or not primitive_name:
            raise ValueError("primitive_name must be a non-empty string")
        if not callable(impl):
            raise TypeError("impl must be callable")
        self.impls[primitive_name] = impl

    def get_impl(self, primitive_name: str) -> PrimitiveImpl:
        impl = self.impls.get(primitive_name)
        if impl is None:
            raise NotImplementedError(
                f"{self.scope}.{primitive_name} has no provider implementation "
                "registered yet."
            )
        return impl

    def _provider_module_names(self) -> tuple[str, ...]:
        shared_package, _, _ = self.package.rpartition(".")
        return tuple(
            dict.fromkeys(
                (
                    f"{self.package}._provider",
                    f"{shared_package}._provider",
                )
            )
        )

    def snapshot_provider_session(self):
        for module_name in self._provider_module_names():
            provider = sys.modules.get(module_name)
            if provider is None:
                continue
            snapshot = getattr(
                provider,
                "_snapshot_active_session_state",
                None,
            ) or getattr(
                provider,
                "snapshot_active_session_state",
                None,
            )
            if snapshot is None:
                continue
            try:
                return ((module_name, snapshot()),)
            except (AttributeError, ImportError):
                continue
        return ((self._provider_module_names()[-1], None),)

    def restore_provider_session(self, snapshots) -> None:
        for module_name, snapshot in snapshots:
            provider = sys.modules.get(module_name)
            if provider is None:
                continue
            restore = getattr(
                provider, "_restore_active_session_state", None
            ) or getattr(
                provider,
                "restore_active_session_state",
                None,
            )
            if restore is not None:
                try:
                    restore(snapshot)
                except (AttributeError, ImportError):
                    continue

    def dispatch_provider(
        self,
        primitive_name: str,
        impl: PrimitiveImpl,
        payload: Mapping[str, Any],
    ) -> Any:
        self.debug_log(f"dispatch primitive={primitive_name} backend=provider")
        return impl(**dict(payload))


def dispatch_provider_primitive(
    dispatcher: PrimitiveDispatcher,
    primitive_name: str,
    *,
    kwargs: Mapping[str, Any] | None = None,
) -> Any:
    impl = dispatcher.get_impl(primitive_name)
    payload = dict(kwargs) if kwargs is not None else {}
    return dispatcher.dispatch_provider(primitive_name, impl, payload)


def _requires_single_phase_validation(
    impl: PrimitiveImpl,
    single_phase_context: SinglePhaseContext,
) -> bool:
    if bool(getattr(impl, "_supports_native_thread_data", False)):
        return True
    if single_phase_context.temp_storage is not None:
        return True
    return False


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


def dispatch_single_phase_primitive(
    dispatcher: PrimitiveDispatcher,
    primitive_name: str,
    *,
    kwargs: Mapping[str, Any] | None = None,
    extract_single_phase_context: ExtractSinglePhaseContext,
    coerce_thread_payloads_to_thread_data: CoerceThreadPayloads,
    register_temp_storage_use: RegisterTempStorageUse,
    dispatch_thread_data_aware: DispatchThreadDataAware,
) -> Any:
    impl = dispatcher.get_impl(primitive_name)
    payload = dict(kwargs) if kwargs is not None else {}
    supports_native_thread_data = bool(
        getattr(impl, "_supports_native_thread_data", False)
    )
    uses_planned_temp_storage = bool(getattr(impl, "_uses_planned_temp_storage", False))
    strip_launch_metadata = supports_native_thread_data and not bool(
        getattr(impl, "_preserves_launch_metadata", False)
    )
    single_phase_context = extract_single_phase_context(
        primitive_name,
        payload,
        reserve_context_fields=supports_native_thread_data,
    )
    coerce_thread_payloads_to_thread_data(primitive_name, payload)
    temp_storage_payload = dict(payload)
    if not _requires_single_phase_validation(impl, single_phase_context):
        dispatcher.debug_log(f"dispatch primitive={primitive_name} backend=provider")
        return dispatch_thread_data_aware(
            primitive_name,
            impl,
            payload,
            strip_launch_metadata=strip_launch_metadata,
        )

    temp_storage = single_phase_context.temp_storage
    if (
        temp_storage is not None
        and getattr(temp_storage, "is_deferred", False)
        and not bool(getattr(impl, "_supports_deferred_temp_storage", False))
    ):
        raise NotImplementedError(
            f"{dispatcher.scope}.{primitive_name} does not yet support inferred "
            "TempStorage; deferred planning is currently limited to "
            "cuda.coop.cutlass block Load, Store, Exchange, Scan, "
            "AdjacentDifference, Discontinuity, RadixSort, and MergeSort"
        )
    with single_phase_transaction(
        single_phase_context,
        snapshot_provider_session=dispatcher.snapshot_provider_session,
        restore_provider_session=dispatcher.restore_provider_session,
    ):
        if not uses_planned_temp_storage:
            register_temp_storage_use(
                primitive_name,
                single_phase_context,
                temp_storage_payload,
            )
        dispatcher.debug_log(f"dispatch primitive={primitive_name} backend=provider")
        with activate_single_phase_context(single_phase_context):
            return dispatch_thread_data_aware(
                primitive_name,
                impl,
                payload,
                strip_launch_metadata=strip_launch_metadata,
            )
