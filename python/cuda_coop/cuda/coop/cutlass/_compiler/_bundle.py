# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Resolve one trace's provider source to a validated LTO-IR artifact."""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict

from cutlass._mlir import ir

from cuda.coop._core._source_dump import dump_source

from . import _cache, _nvrtc


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
    context = _nvrtc.resolve_compile_context(required_headers)
    options = _nvrtc.compiler_options(context, arch)
    identity = ("cutlass-ltoir-v1", source, asdict(context), options)
    key = hashlib.sha256(repr(identity).encode("utf-8")).hexdigest()
    dump_source(source, backend="cutlass", identity=(key,))
    cached = _cache.memory_cached_bundle(key)
    if cached is None:
        path = os.path.join(
            _cache.ensure_cache_dir("cuda.coop.cutlass"), f"{key}.ltoir"
        )
        with _cache.artifact_lock(path, scope="cuda.coop.cutlass"):
            cached = _cache.memory_cached_bundle(key) or _cache.load_bundle(path, key)
            if cached is None:
                blob = _nvrtc.compile_ltoir(source, options)
                cached = _cache.publish_bundle(path, key, blob)
            _cache.store_memory_bundle(key, cached)
    _cache.add_managed_bundle_path(cached.path)
    return cached.path
