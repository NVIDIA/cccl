# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Resolve one trace's provider source to a validated LTO-IR artifact."""

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
    """Compile one provider bundle and recover its exact C++ storage layouts."""

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
                    blob, layouts = _nvrtc.compile_ltoir_with_layouts(prepared, options)
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
