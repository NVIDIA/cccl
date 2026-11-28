# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile a trace's C++ requests and record its device-link inputs."""

from __future__ import annotations

import os
import sys
from typing import Any

from cutlass.base_dsl.common import DSLRuntimeError, DSLUserCodeError

from . import (
    _bundle,
    _cache,
    _convergence,
    _lifetime,
    _rendering,
    _state,
    _storage,
    _target,
)

ROOT_SCOPE = "cuda.coop.cutlass"


def _remove_managed_bundle_link_options(dsl: Any) -> None:
    """Remove prior provider bundles from persistent CUTLASS link options.

    CuTe merges GPU-module link attributes into options that can outlive one
    trace. Keep user-supplied libraries, but remove provider-owned paths
    before attaching the current module's bundle.
    """

    managed_paths = _cache.managed_bundle_paths()
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
            f"{ROOT_SCOPE} provider requires a CUTLASS DSL with mutable "
            "link-library compile options.",
            cause=exc,
        ) from exc

    filtered_paths = [
        path for path in paths if os.path.realpath(path) not in managed_paths
    ]
    if filtered_paths != paths:
        options[LinkLibraries] = LinkLibraries(",".join(filtered_paths))


def _attach_link_inputs(dsl, module, arch, paths):
    """Select the device-link implementation for this trace."""
    if sys.platform == "win32":
        from ._windows_pipeline import configure_device_link

        configure_device_link(dsl, module, arch, paths)
    else:
        for link_path in paths:
            _bundle.append_link_library_attr(module, link_path)


def _trace_finalize_hook(dsl, module, function_name):
    """Finish one trace's provider bundle and deferred scratch allocations.

    CuTe calls this hook after tracing and before hashing the module. All
    cooperative calls are known, so their wrappers can share one NVRTC
    compilation.

    The matching session supplies deduplicated wrapper requests and every
    recorded scratch use. Compile first to learn exact C++ layouts, then plan
    and insert allocations. Mark the provider calls convergent, then attach
    the provider and device-operator paths to the GPU modules on Linux.
    On Windows, record them in the traced module's device-link plan.
    Other trace sessions remain separate, including nested compilations.

    Parameters
    ----------
    dsl : object
        Active CuTe DSL instance, which owns the compile options.
    module : ir.Module
        Completed trace module that owns the cooperative calls.
    function_name : str
        Function name supplied by CuTe. Unused because sessions are selected
        by module identity, not by function name.
    """

    del function_name
    options = dsl.compile_options
    _remove_managed_bundle_link_options(dsl)
    session = _state.lookup_bundle_session(options, trace_module_op=module)
    if session is None or not session.belongs_to_trace_module(module):
        return
    session = _state.pop_bundle_session(options, trace_module_op=module)
    if session is None or session.is_empty():
        return
    _lifetime.validate_temp_storage_lifetimes(session)
    requests = session.request_list()
    if not requests:
        plans = _storage.plan_deferred_temp_storage_events(
            session.deferred_temp_storage_event_list(), {}
        )
        _storage.materialize_deferred_temp_storage_plans(plans)
        _lifetime.erase_temp_storage_lifetime_markers(session)
        return
    source = _rendering.render_bundle_source(requests)
    arch = _target.resolve_nvrtc_arch(
        ROOT_SCOPE, lambda: _target.configured_gpu_arch(lambda: dsl)
    )
    operators = _rendering.bundle_device_operators(requests)
    for operator in operators:
        if operator.arch != arch:
            raise DSLUserCodeError(
                f"{ROOT_SCOPE} device operator targets {operator.arch}, "
                f"but this kernel targets {arch}; compile the operator "
                "for the kernel's architecture."
            )
    headers = tuple(_rendering.registered_bundle_headers().values())
    probes = _rendering.bundle_scratch_layout_probes(requests)
    if probes:
        compilation = _bundle.compile_bundle_source_with_layouts(
            source,
            arch=arch,
            required_headers=headers,
            layout_probes=tuple(probes.values()),
        )
        layouts = compilation.layouts
        path = compilation.path
    else:
        layouts = {}
        path = _bundle.compile_bundle_source(
            source, arch=arch, required_headers=headers
        )
    plans = _storage.plan_deferred_temp_storage_events(
        session.deferred_temp_storage_event_list(), layouts
    )
    _storage.materialize_deferred_temp_storage_plans(plans)
    _lifetime.erase_temp_storage_lifetime_markers(session)
    _convergence.lower_convergent_provider_calls(module, requests)
    paths = (path, *(operator._materialize() for operator in operators))
    _attach_link_inputs(dsl, module, arch, paths)


_state.register_bundle_finalizer(_trace_finalize_hook)
