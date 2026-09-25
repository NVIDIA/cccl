# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Canonical request registration and generated C++ rendering."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from ._types import BundleRenderer

_FEATURE_DEFINE_RE = re.compile(r"^#define\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s|\(|$)")
_BUNDLE_RENDERERS: dict[str, BundleRenderer] = {}


def register_bundle_renderer(kind, *, render, include_lines=(), cccl_headers=()):
    if kind in _BUNDLE_RENDERERS:
        raise ValueError(f"bundle renderer {kind!r} is already registered")
    _BUNDLE_RENDERERS[kind] = BundleRenderer(
        tuple(include_lines), tuple(cccl_headers), render
    )


def bundle_renderer_for(request: Any) -> BundleRenderer | None:
    return _BUNDLE_RENDERERS.get(getattr(request, "kind", ""))


def canonical_bundle_requests(requests: Iterable[Any]) -> tuple[Any, ...]:
    """Return one deterministic request per provider symbol."""

    requests_by_symbol: dict[str, Any] = {}
    for request in requests:
        symbol = getattr(request, "symbol_name", None)
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("provider bundle requests require a non-empty symbol_name")
        existing = requests_by_symbol.get(symbol)
        if existing is not None and existing != request:
            raise ValueError(
                f"provider symbol {symbol!r} maps to conflicting bundle requests"
            )
        requests_by_symbol[symbol] = request
    return tuple(requests_by_symbol[symbol] for symbol in sorted(requests_by_symbol))


def canonical_bundle_preamble_lines(lines: Iterable[str]) -> tuple[str, ...]:
    """Canonicalize feature definitions before all other preamble lines."""

    feature_definitions: dict[str, str] = {}
    other_lines: set[str] = set()
    for line in lines:
        if not line:
            continue
        if line.startswith("#define "):
            match = _FEATURE_DEFINE_RE.match(line)
            if match is None:
                raise ValueError(f"invalid provider feature definition: {line!r}")
            name = match.group(1)
            existing = feature_definitions.get(name)
            if existing is not None and existing != line:
                raise ValueError(
                    f"provider feature {name!r} has conflicting definitions"
                )
            feature_definitions[name] = line
        else:
            other_lines.add(line)
    return (
        *(feature_definitions[name] for name in sorted(feature_definitions)),
        *sorted(other_lines),
    )


def bundle_include_lines(requests: Iterable[Any]) -> list[str]:
    include_lines: list[str] = []
    for request in canonical_bundle_requests(requests):
        renderer = bundle_renderer_for(request)
        if renderer is not None:
            include_lines.extend(renderer.include_lines)
    return list(canonical_bundle_preamble_lines(include_lines))


def registered_bundle_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    for kind in sorted(_BUNDLE_RENDERERS):
        renderer = _BUNDLE_RENDERERS[kind]
        for include, relative_path in sorted(renderer.cccl_headers):
            existing = headers.get(include)
            if existing is not None and existing != relative_path:
                raise ValueError(
                    f"provider include {include!r} maps to conflicting CCCL headers"
                )
            headers[include] = relative_path
    return {include: headers[include] for include in sorted(headers)}


def render_bundle_source(requests):
    lines = [*bundle_include_lines(requests), 'extern "C" {']
    for request in canonical_bundle_requests(requests):
        renderer = bundle_renderer_for(request)
        if renderer is None:
            raise ValueError(f"No CUTLASS provider renderer for {request.kind!r}")
        lines.extend(renderer.render(request))
    return "\n".join([*lines, "}", ""])
