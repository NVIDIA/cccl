# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile and cache generated C++ as LTO-IR for CuTe linking."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from cuda.coop.cutlass import _aot_pack
from cuda.coop.cutlass._planning._source_dump import dump_source
from cutlass._mlir import ir

from . import _cache, _nvrtc
from ._layout import (
    BundleCompilation,
    _prepare_layout_probes,
    _PreparedLayoutProbes,
)
from ._types import ScratchLayout, ScratchLayoutProbe


def append_link_library_attr(module: ir.Module, path: str) -> None:
    """Attach a bundle path while retaining other GPU-module libraries.

    CuTe merges these attributes into its link options later. Sorted unique
    paths keep repeated registration deterministic.
    """

    # CuTe generally emits one gpu.module containing all kernels in a trace.
    # Visit every GPU module, as CuTe's link-library collection does.
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
    path, layouts = _resolve_bundle(prepared, arch, required_headers)
    _cache.add_managed_bundle_path(path)
    return BundleCompilation(
        path,
        {
            key: layouts[expression]
            for key, expression in prepared.key_to_expression.items()
        },
    )


def _resolve_bundle(
    prepared: _PreparedLayoutProbes,
    arch: str,
    required_headers: tuple[str, ...],
) -> tuple[str, dict[str, ScratchLayout]]:
    # A selected pack supplies its own compiled CUB implementation and layout
    # answers. An exact hit only needs the consumer linker.
    pack_key = None
    if _aot_pack.active():
        identity = (
            "cutlass-provider-abi-v1",
            sys.platform,  # NVRTC uses LLP64 on Windows and LP64 on Linux.
            prepared.source,
            prepared.expressions,
            tuple(flag.decode("ascii") for flag in _nvrtc.bundle_options(arch)),
        )
        pack_key = hashlib.sha256(
            json.dumps(identity, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        dump_source(prepared.source, backend="cutlass", identity=(pack_key,))
        hit = _aot_pack.lookup(pack_key, arch, prepared.expressions)
        if hit is not None:
            blob, layouts, producer_version = hit
            path = _cache.materialize_pack_artifact(blob)
            if _aot_pack.capturing():
                _aot_pack.record(
                    pack_key, arch, blob, layouts, producer_version
                )
            return path, layouts

    cached, producer_version = _jit_bundle(
        prepared, arch, required_headers, dump_jit_source=pack_key is None
    )
    if pack_key is not None and _aot_pack.capturing():
        blob = Path(cached.path).read_bytes()
        if hashlib.sha256(blob).hexdigest() != cached.artifact_sha256:
            raise RuntimeError("Provider artifact changed during AOT capture")
        _aot_pack.record(
            pack_key, arch, blob, cached.layouts_by_expression, producer_version
        )
    return cached.path, cached.layouts_by_expression


def _jit_bundle(
    prepared: _PreparedLayoutProbes,
    arch: str,
    required_headers: tuple[str, ...],
    *,
    dump_jit_source: bool,
) -> tuple[_cache._CachedBundle, tuple[int, int]]:
    context = _nvrtc.resolve_compile_context(required_headers)
    options = _nvrtc.compiler_options(context, arch)
    identity: tuple[object, ...] = (
        "cutlass-ltoir-v1",
        prepared.source,
        asdict(context),
        options,
    )
    if prepared.expressions:
        identity += (prepared.expressions,)
    key = hashlib.sha256(repr(identity).encode("utf-8")).hexdigest()
    if dump_jit_source:
        dump_source(prepared.source, backend="cutlass", identity=(key,))

    def complete(cached):
        if cached is None or set(cached.layouts_by_expression) != set(
            prepared.expressions
        ):
            return None
        return cached

    # Scope memory hits to the selected directory, including after an override
    # changes in a long-lived process.
    path = os.path.join(
        _cache.ensure_cache_dir("cuda.coop.cutlass"), f"{key}.ltoir"
    )
    cached = complete(_cache.memory_cached_bundle(path))
    if cached is None:
        with _cache.artifact_lock(path, scope="cuda.coop.cutlass"):
            cached = complete(_cache.memory_cached_bundle(path)) or complete(
                _cache.load_bundle(path, key)
            )
            if cached is None:
                options += _nvrtc.pch_options(context)
                if prepared.expressions:
                    blob, layouts = _nvrtc.compile_ltoir_with_layouts(
                        prepared, options
                    )
                else:
                    blob, layouts = (
                        _nvrtc.compile_ltoir(prepared.source, options),
                        {},
                    )
                cached = _cache.publish_bundle(
                    path, key, blob, layouts_by_expression=layouts
                )
            _cache.store_memory_bundle(path, cached)
    return cached, context.nvrtc_version
