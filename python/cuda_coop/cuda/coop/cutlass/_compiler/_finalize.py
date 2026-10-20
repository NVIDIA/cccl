# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile a trace's C++ requests and attach the result for CuTe linking."""

from __future__ import annotations

import os
from typing import Any

from cutlass.base_dsl.common import DSLRuntimeError

from . import _bundle, _cache, _rendering, _state, _storage, _target

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


def _trace_finalize_hook(dsl, module, function_name):
    """Finish one trace's provider bundle and deferred scratch allocations.

    CuTe calls this registered hook after it has traced a module and before
    it links device code. At that point all cooperative calls are known, so
    their wrappers can share one NVRTC compilation. The hook changes the
    module and compiler link options; it does not launch the kernel.

    The matching session supplies deduplicated wrapper requests and every
    recorded scratch use. Compile first to learn exact C++ layouts, then plan
    and insert allocations before attaching the LTO-IR file for CuTe linking.
    Other trace sessions remain separate, including nested compilations.

    Parameters
    ----------
    dsl : object
        Active CuTe DSL instance, which owns the compile options.
    module : ir.Module
        Completed trace module whose GPU modules receive the link attribute.
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
    requests = session.request_list()
    if not requests:
        plans = _storage.plan_deferred_temp_storage_events(
            session.deferred_temp_storage_event_list(), {}
        )
        _storage.materialize_deferred_temp_storage_plans(plans)
        return
    source = _rendering.render_bundle_source(requests)
    arch = _target.resolve_nvrtc_arch(
        ROOT_SCOPE, lambda: _target.configured_gpu_arch(lambda: dsl)
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
    _bundle.append_link_library_attr(module, path)


_state.register_bundle_finalizer(_trace_finalize_hook)
