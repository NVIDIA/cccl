# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Finalize traced CUTLASS provider requests into one linkable bundle."""

from __future__ import annotations

import os
import shutil
from typing import Any

from cutlass.base_dsl.common import DSLRuntimeError

from . import _bundle as _provider_bundle_support
from . import _cache as _provider_cache_support
from . import _rendering as _provider_rendering
from . import _state as _provider_state
from . import _storage as _provider_storage
from . import _target as _provider_target_support
from . import _types as _provider_types
from ._types import ROOT_SCOPE as _ROOT_SCOPE

_PROVIDER_DIR = os.path.dirname(os.path.dirname(__file__))


def _get_cute_dsl():
    return _provider_state._get_cute_dsl()


def _remove_managed_bundle_link_options(dsl: Any) -> None:
    """Keep persistent CUTLASS compile options from relinking prior bundles."""

    managed_paths = _provider_cache_support.managed_bundle_paths()
    if not managed_paths:
        return
    try:
        from cutlass.base_dsl.compiler import LinkLibraries

        options = dsl.compile_options.options
        try:
            option = options[LinkLibraries]
        except KeyError:
            return
        paths = [path for path in str(option.value).split(",") if path]
    except (AttributeError, ImportError, TypeError) as exc:
        raise DSLRuntimeError(
            f"{_ROOT_SCOPE} provider requires a CUTLASS DSL with mutable "
            "link-library compile options.",
            cause=exc,
        ) from exc

    filtered_paths = [
        path for path in paths if os.path.realpath(path) not in managed_paths
    ]
    if filtered_paths != paths:
        options[LinkLibraries] = LinkLibraries(",".join(filtered_paths))


def _configured_gpu_arch() -> str:
    return _provider_target_support.configured_gpu_arch(_get_cute_dsl)


def _resolve_nvrtc_arch() -> str:
    return _provider_target_support.resolve_nvrtc_arch(
        _ROOT_SCOPE,
        _configured_gpu_arch,
    )


def _resolve_nvrtc_sm_arch() -> str:
    return _provider_target_support.resolve_nvrtc_sm_arch(
        _ROOT_SCOPE,
        _configured_gpu_arch,
    )


def _select_bundle_format() -> str:
    return _provider_target_support.select_bundle_format(_ROOT_SCOPE)


def _compile_bundle_source(source: str, symbols: tuple[str, ...]) -> str:
    return _provider_bundle_support.compile_bundle_source(
        source,
        scope=_ROOT_SCOPE,
        provider_dir=_PROVIDER_DIR,
        registered_headers=_provider_rendering.registered_bundle_headers,
        select_bundle_format=_select_bundle_format,
        resolve_nvrtc_sm_arch=_resolve_nvrtc_sm_arch,
        resolve_nvrtc_arch=_resolve_nvrtc_arch,
        symbols=symbols,
        which=shutil.which,
    )


def _compile_bundle_source_with_layouts(
    source: str,
    probes: dict[object, _provider_types.ScratchLayoutProbe],
    symbols: tuple[str, ...],
) -> _provider_bundle_support.BundleCompilation:
    return _provider_bundle_support.compile_bundle_source_with_layouts(
        source,
        layout_probes=(
            _provider_bundle_support.LayoutProbe(
                key=key,
                size_expression=probe.size_expression,
                alignment_expression=probe.alignment_expression,
            )
            for key, probe in probes.items()
        ),
        scope=_ROOT_SCOPE,
        provider_dir=_PROVIDER_DIR,
        registered_headers=_provider_rendering.registered_bundle_headers,
        select_bundle_format=_select_bundle_format,
        resolve_nvrtc_sm_arch=_resolve_nvrtc_sm_arch,
        resolve_nvrtc_arch=_resolve_nvrtc_arch,
        symbols=symbols,
        which=shutil.which,
    )


def _reject_unregistered_request(request: Any) -> list[str]:
    kind = getattr(request, "kind", type(request).__name__)
    raise ValueError(f"CUTLASS provider request {kind!r} has no renderer")


def _render_bundle_source(requests: list[Any]) -> str:
    return _provider_rendering.render_bundle_source(
        requests,
        scope=_ROOT_SCOPE,
        render_local_request=_reject_unregistered_request,
    )


def _trace_finalize_hook(dsl: Any, module: Any, function_name: str) -> None:
    session = _provider_state.pop_bundle_session(dsl.compile_options)
    _remove_managed_bundle_link_options(dsl)
    del dsl, function_name
    if (
        session is None
        or not session.belongs_to_trace_module(getattr(module, "operation", module))
        or session.is_empty()
    ):
        return

    requests = session.request_list()
    events = session.deferred_temp_storage_event_list()
    probes = _provider_rendering.bundle_scratch_layout_probes(requests)
    missing_probe_keys = {
        event.requirement_key for event in events if event.requirement_key not in probes
    }
    if missing_probe_keys:
        raise DSLRuntimeError(
            "Deferred TempStorage events have no registered exact-layout probe: "
            f"{sorted(map(repr, missing_probe_keys))}"
        )

    source = _render_bundle_source(requests)
    symbols = tuple(sorted(request.symbol_name for request in requests))
    if probes:
        compilation = _compile_bundle_source_with_layouts(source, probes, symbols)
        bundle_path = compilation.path
        layouts = {
            key: _provider_types.ScratchLayout(
                size_in_bytes=layout.size_in_bytes,
                alignment=layout.alignment,
            )
            for key, layout in compilation.layouts.items()
        }
    else:
        bundle_path = _compile_bundle_source(source, symbols)
        layouts = {}

    plans = _provider_storage.plan_deferred_temp_storage_events(events, layouts)
    _provider_storage.materialize_deferred_temp_storage_plans(plans)
    _provider_bundle_support.append_link_library_attr(module, bundle_path)


_provider_state.register_bundle_finalizer(
    _trace_finalize_hook,
    scope=_ROOT_SCOPE,
)

__all__: tuple[str, ...] = ()
