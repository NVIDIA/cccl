# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Finalize each trace's provider requests before final CuTe linking."""

from __future__ import annotations

import os
from typing import Any

from cutlass.base_dsl.common import DSLRuntimeError

from . import _bundle, _cache, _rendering, _state, _target

ROOT_SCOPE = "cuda.coop.cutlass"


def _remove_managed_bundle_link_options(dsl: Any) -> None:
    """Keep persistent CUTLASS compile options from relinking prior bundles."""

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
    source = _rendering.render_bundle_source(requests)
    arch = _target.resolve_nvrtc_arch(
        ROOT_SCOPE, lambda: _target.configured_gpu_arch(lambda: dsl)
    )
    path = _bundle.compile_bundle_source(
        source,
        arch=arch,
        required_headers=tuple(_rendering.registered_bundle_headers().values()),
    )
    _bundle.append_link_library_attr(module, path)


_state.register_bundle_finalizer(_trace_finalize_hook)
