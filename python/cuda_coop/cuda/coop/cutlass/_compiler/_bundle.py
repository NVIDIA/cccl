# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile and cache generated C++ as LTO-IR for CuTe linking."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from dataclasses import asdict

from cutlass._mlir import ir

from cuda.coop._core._source_dump import dump_source

from . import _cache, _nvrtc
from ._layout import BundleCompilation, _prepare_layout_probes
from ._types import ScratchLayoutProbe


def append_link_library_attr(module: ir.Module, path: str) -> None:
    """Attach a bundle path while retaining other GPU-module libraries.

    CuTe merges these attributes into its link options later. Sorted unique
    paths keep repeated registration deterministic.
    """

    for op in module.body.operations:
        if op.name != "gpu.module":
            continue
        paths = set()
        if "link-libraries" in op.attributes:
            paths.update(attr.value for attr in op.attributes["link-libraries"])
        paths.add(path)
        op.attributes["link-libraries"] = ir.ArrayAttr.get(
            [ir.StringAttr.get(item) for item in sorted(paths)]
        )


def compile_bundle_source(
    source: str, *, arch: str, required_headers: tuple[str, ...]
) -> str:
    """Return the cached LTO-IR path for a bundle with no layout queries.

    The shared compiler helper validates cached bytes or compiles under the
    artifact lock. CuTe links these device functions into the final kernel.
    """

    return _compile_bundle_source(
        source, arch=arch, required_headers=required_headers, layout_probes=()
    ).path


def compile_bundle_source_with_layouts(
    source: str,
    *,
    arch: str,
    required_headers: tuple[str, ...],
    layout_probes: Iterable[ScratchLayoutProbe],
) -> BundleCompilation:
    """Compile a provider bundle and recover its C++ scratch layouts.

    Parameters
    ----------
    source : str
        Generated C++ wrappers before the layout probes are added.
    arch : str
        NVRTC target architecture.
    required_headers : tuple of str
        Header paths that must be available in the selected include roots.
    layout_probes : iterable of ScratchLayoutProbe
        C++ size/alignment expressions indexed by caller requirement keys.

    Returns
    -------
    BundleCompilation
        Cached LTO-IR path and layouts indexed by the supplied keys. The image
        and layouts come from one NVRTC program. Equivalent query sets can
        reuse a compilation even when their caller keys differ.
    """

    return _compile_bundle_source(
        source,
        arch=arch,
        required_headers=required_headers,
        layout_probes=layout_probes,
    )


def _compile_bundle_source(
    source: str,
    *,
    arch: str,
    required_headers: tuple[str, ...],
    layout_probes: Iterable[ScratchLayoutProbe],
) -> BundleCompilation:
    """Cache the compiled image together with its complete set of layouts.

    Generated probe source, query expressions, compiler context, and options
    identify the artifact. Caller requirement keys only map results back to
    uses; they do not change compiled code. A cache entry must contain exactly
    the requested expression set before any of its layouts can be reused.
    """

    prepared = _prepare_layout_probes(source, layout_probes)
    context = _nvrtc.resolve_compile_context(required_headers)
    options = _nvrtc.compiler_options(context, arch)
    identity = ("cutlass-ltoir-v1", prepared.source, asdict(context), options)
    if prepared.expressions:
        identity += (prepared.expressions,)
    key = hashlib.sha256(repr(identity).encode("utf-8")).hexdigest()
    dump_source(prepared.source, backend="cutlass", identity=(key,))

    def complete(cached):
        if cached is None or set(cached.layouts_by_expression) != set(
            prepared.expressions
        ):
            return None
        return cached

    cached = complete(_cache.memory_cached_bundle(key))
    if cached is None:
        path = os.path.join(
            _cache.ensure_cache_dir("cuda.coop.cutlass"), f"{key}.ltoir"
        )
        with _cache.artifact_lock(path, scope="cuda.coop.cutlass"):
            cached = complete(_cache.memory_cached_bundle(key)) or complete(
                _cache.load_bundle(path, key)
            )
            if cached is None:
                if prepared.expressions:
                    blob, layouts = _nvrtc.compile_ltoir_with_layouts(
                        prepared, options
                    )
                else:
                    blob, layouts = _nvrtc.compile_ltoir(source, options), {}
                cached = _cache.publish_bundle(
                    path, key, blob, layouts_by_expression=layouts
                )
            _cache.store_memory_bundle(key, cached)
    _cache.add_managed_bundle_path(cached.path)
    return BundleCompilation(
        cached.path,
        {
            key: cached.layouts_by_expression[expression]
            for key, expression in prepared.key_to_expression.items()
        },
    )
