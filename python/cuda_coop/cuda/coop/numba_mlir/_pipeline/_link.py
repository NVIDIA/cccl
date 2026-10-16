# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile a kernel's providers once, and link them from memory.

Storage analysis compiles every provider specialization of a kernel in one
NVRTC translation unit, a *bundle*. Bundles are memoized in process by what
determines their bytes: the providers' private symbol digests, the target
and compiler options, the header identity and the NVRTC version. A second
kernel, or a recompile of the same one, with the same providers reuses the
bundle without compiling.

Each bundle links its provider image and any supporting callback/type images
through in-memory ``LTOIR`` items. The memo is a least-recently-used cache
bounded both by entry count and by the total size of those images.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from io import StringIO
from typing import Any

from . import _bridges as nb

MAX_ENTRIES = 64
MAX_BYTES = 256 * 1024 * 1024


@dataclass
class Bundle:
    """One compiled bundle.

    ``layouts`` maps each provider's private symbol digest to its scratch
    ``(size, alignment)``, or to ``None`` when it takes no scratch.
    """

    key: tuple
    image: bytes
    layouts: dict[str, tuple[int, int] | None]
    specializations: int
    link_item: Any = field(default=None, repr=False)
    support_images: tuple[bytes, ...] = field(default=(), repr=False)
    support_link_items: tuple = field(default=(), repr=False)

    @property
    def link_items(self) -> tuple:
        return (self.link_item, *self.support_link_items)

    @property
    def byte_size(self) -> int:
        return len(self.image) + sum(map(len, self.support_images))


@dataclass
class _Statistics:
    compiles: int = 0
    hits: int = 0
    evictions: int = 0


STATISTICS = _Statistics()
_MEMO: OrderedDict[tuple, Bundle] = OrderedDict()
_memo_bytes = 0


def bundle_key(algorithms: Sequence[Any], identity: Any, context: Any) -> tuple:
    """Return what determines the bytes of the bundle of ``algorithms``."""

    # Validate mutable providers even on a cache hit. The existing provider
    # key includes callback code, target, signature and captured semantics.
    for algorithm in algorithms:
        algorithm._bind_private_symbol_namespace(compile_identity=identity)
    return (
        tuple(sorted({a._private_symbol_digest for a in algorithms})),
        identity.cc,
        identity.arch_suffix,
        identity.code,
        identity.compiler_options,
        context.header_identity,
        str(context.nvrtc_version),
    )


def _compile(algorithms: Sequence[Any], identity: Any, context: Any):
    """Compile one translation unit for every distinct provider."""

    from .._compiler import _nvrtc
    from .._types import _dedupe_ltoirs, _strip_source_preamble

    representatives: OrderedDict[Any, Any] = OrderedDict()
    for algorithm in algorithms:
        representatives.setdefault(algorithm._private_symbol_key, algorithm)
    sources, storage_types, udfs = {}, {}, {}
    includes: OrderedDict[str, None] = OrderedDict()
    type_definitions: OrderedDict[str, None] = OrderedDict()
    udf_declarations: OrderedDict[str, str] = OrderedDict()
    support_images = []
    for rep in representatives.values():
        source, support, rep_storage_types, udf = rep._source_code(
            compile_identity=identity
        )
        support_images.extend(support)
        sources[id(rep)] = source
        storage_types[id(rep)] = rep_storage_types
        udfs[id(rep)] = udf
        for include in rep.includes or []:
            includes.setdefault(include, None)
        for definition in rep.type_definitions or []:
            if getattr(definition, "code", ""):
                type_definitions.setdefault(definition.code, None)
        for name, declaration in udf.items():
            previous = udf_declarations.setdefault(name, declaration)
            if previous != declaration:
                raise RuntimeError(
                    "Python operators produced conflicting device symbols"
                )

    text = StringIO()
    text.write("#include <cuda/std/cstdint>\n")
    for include in includes:
        text.write(f"#include <{include}>\n")
    for code in type_definitions:
        text.write(f"{code}\n")
    text.write("\n")
    for declaration in udf_declarations.values():
        text.write(f"{declaration}\n")
    bodies = [
        _strip_source_preamble(sources[id(rep)], rep, udfs[id(rep)]).rstrip()
        + "\n"
        for rep in representatives.values()
    ]
    layout_types = tuple(
        t for rep in representatives.values() for t in storage_types[id(rep)]
    )
    _, (image, layouts) = _nvrtc.compile_with_layouts(
        cpp=text.getvalue() + "\n".join(bodies),
        layout_types=layout_types,
        rdc=True,
        context=context,
        **identity.target.nvrtc_kwargs(),
    )
    by_type = dict(zip(layout_types, layouts, strict=True))
    by_digest = {}
    for algorithm in algorithms:
        rep = representatives[algorithm._private_symbol_key]
        types_ = storage_types[id(rep)]
        by_digest[algorithm._private_symbol_digest] = (
            tuple(by_type[types_[0]]) if types_ else None
        )
    support = tuple(
        blob.encode() if isinstance(blob, str) else bytes(blob)
        for blob in _dedupe_ltoirs(support_images)
    )
    return bytes(image), by_digest, len(representatives), support


def _evict() -> None:
    global _memo_bytes
    while _MEMO and (len(_MEMO) > MAX_ENTRIES or _memo_bytes > MAX_BYTES):
        _, old = _MEMO.popitem(last=False)
        _memo_bytes -= old.byte_size
        STATISTICS.evictions += 1


def compile_bundle(
    algorithms: Sequence[Any], identity: Any, context: Any
) -> Bundle:
    """Return the bundle of ``algorithms``, compiling it on a memo miss."""

    global _memo_bytes
    key = bundle_key(algorithms, identity, context)
    bundle = _MEMO.get(key)
    if bundle is not None:
        _MEMO.move_to_end(key)
        STATISTICS.hits += 1
        return bundle
    image, layouts, specializations, support = _compile(
        algorithms, identity, context
    )
    bundle = Bundle(
        key, image, layouts, specializations, support_images=support
    )
    digest = hashlib.sha1(repr(key).encode()).hexdigest()[:16]
    bundle.link_item = nb.LTOIR(
        lambda: bundle.image, name=f"cuda_coop_bundle_{digest}.ltoir"
    )
    bundle.support_link_items = tuple(
        nb.LTOIR(
            lambda image=image: image,
            name=f"cuda_coop_support_{hashlib.sha256(image).hexdigest()}.ltoir",
        )
        for image in support
    )
    STATISTICS.compiles += 1
    _MEMO[key] = bundle
    _memo_bytes += bundle.byte_size
    _evict()
    return bundle


def clear() -> None:
    """Forget every memoized bundle."""

    global _memo_bytes
    _MEMO.clear()
    _memo_bytes = 0


def memo_size() -> tuple[int, int]:
    """Return the memo's entry count and total image size in bytes."""

    return len(_MEMO), _memo_bytes


__all__ = [
    "MAX_BYTES",
    "MAX_ENTRIES",
    "STATISTICS",
    "Bundle",
    "bundle_key",
    "clear",
    "compile_bundle",
    "memo_size",
]
