# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


"""Canonical request registration and generated C++ rendering."""

from __future__ import annotations

import re
from collections.abc import Callable, Hashable, Iterable
from typing import Any

from ._types import BundleRenderer, ScratchLayoutProbe

_FEATURE_DEFINE_RE = re.compile(r"^#define\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s|\(|$)")

_BUNDLE_RENDERERS: dict[str, BundleRenderer] = {}


def register_bundle_renderer(
    kind: str,
    *,
    render: Callable[[Any], list[str]],
    include_lines: tuple[str, ...] = (),
    cccl_headers: tuple[tuple[str, str], ...] = (),
    scratch_layout_probe: Callable[[Any], ScratchLayoutProbe | None] | None = None,
) -> None:
    """Register an internal non-block request renderer with the shared bundle JIT."""
    if not kind:
        raise ValueError("kind must be a non-empty string")
    if not callable(render):
        raise TypeError("render must be callable")
    if scratch_layout_probe is not None and not callable(scratch_layout_probe):
        raise TypeError("scratch_layout_probe must be callable or None")
    if kind in _BUNDLE_RENDERERS:
        raise ValueError(f"bundle renderer {kind!r} is already registered")
    _BUNDLE_RENDERERS[kind] = BundleRenderer(
        include_lines=tuple(include_lines),
        cccl_headers=tuple(cccl_headers),
        render=render,
        scratch_layout_probe=scratch_layout_probe,
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


def bundle_scratch_layout_probes(
    requests: list[Any],
) -> dict[Hashable, ScratchLayoutProbe]:
    """Collect exact-layout probes registered by provider request renderers."""

    probes: dict[Hashable, ScratchLayoutProbe] = {}
    for request in canonical_bundle_requests(requests):
        renderer = bundle_renderer_for(request)
        if renderer is None or renderer.scratch_layout_probe is None:
            continue
        probe = renderer.scratch_layout_probe(request)
        if probe is None:
            continue
        try:
            hash(probe.requirement_key)
        except TypeError as exc:
            raise TypeError("scratch requirement keys must be hashable") from exc
        existing = probes.get(probe.requirement_key)
        if existing is not None and existing != probe:
            raise ValueError(
                "scratch requirement key maps to conflicting NVRTC layout probes"
            )
        probes[probe.requirement_key] = probe
    return probes


def make_scratch_layout_probe(
    requirement_key: Hashable,
    cpp_type: str,
) -> ScratchLayoutProbe:
    """Describe a name expression for the exact layout of ``cpp_type``."""

    if not cpp_type:
        raise ValueError("scratch layout probe requires a non-empty C++ type")
    return ScratchLayoutProbe(
        requirement_key=requirement_key,
        size_expression=f"sizeof({cpp_type})",
        alignment_expression=f"alignof({cpp_type})",
    )


def bundle_source_preamble_lines(
    include_lines: list[str] | tuple[str, ...] = (),
) -> list[str]:
    return [
        *[line for line in include_lines if line],
        'extern "C" {',
        "static inline unsigned int cuda_coop_cutlass_logical_warp_mask(",
        "    unsigned int logical_width) {",
        "  if (logical_width >= 32u) {",
        "    return 0xffffffffu;",
        "  }",
        "  unsigned int lane;",
        '  asm("mov.u32 %0, %%laneid;" : "=r"(lane));',
        "  unsigned int first_lane = (lane / logical_width) * logical_width;",
        "  return ((1u << logical_width) - 1u) << first_lane;",
        "}",
        "static inline void cuda_coop_cutlass_warp_sync(unsigned int logical_width) {",
        "  unsigned int mask =",
        "      cuda_coop_cutlass_logical_warp_mask(logical_width);",
        '  asm volatile("bar.warp.sync %0;" :: "r"(mask) : "memory");',
        "}",
        "static inline void cuda_coop_cutlass_block_sync() {",
        '  asm volatile("bar.sync 0;" ::: "memory");',
        "}",
        "static inline unsigned int cuda_coop_cutlass_tid_x() {",
        "  unsigned int tid;",
        '  asm("mov.u32 %0, %%tid.x;" : "=r"(tid));',
        "  return tid;",
        "}",
        "static inline unsigned int cuda_coop_cutlass_tid_y() {",
        "  unsigned int tid;",
        '  asm("mov.u32 %0, %%tid.y;" : "=r"(tid));',
        "  return tid;",
        "}",
        "static inline unsigned int cuda_coop_cutlass_tid_z() {",
        "  unsigned int tid;",
        '  asm("mov.u32 %0, %%tid.z;" : "=r"(tid));',
        "  return tid;",
        "}",
        "static inline unsigned int cuda_coop_cutlass_ntid_x() {",
        "  unsigned int ntid;",
        '  asm("mov.u32 %0, %%ntid.x;" : "=r"(ntid));',
        "  return ntid;",
        "}",
        "static inline unsigned int cuda_coop_cutlass_ntid_y() {",
        "  unsigned int ntid;",
        '  asm("mov.u32 %0, %%ntid.y;" : "=r"(ntid));',
        "  return ntid;",
        "}",
        "static inline unsigned int cuda_coop_cutlass_ntid_z() {",
        "  unsigned int ntid;",
        '  asm("mov.u32 %0, %%ntid.z;" : "=r"(ntid));',
        "  return ntid;",
        "}",
        "static inline unsigned int cuda_coop_cutlass_linear_tid() {",
        "  unsigned int tidx = cuda_coop_cutlass_tid_x();",
        "  unsigned int tidy = cuda_coop_cutlass_tid_y();",
        "  unsigned int tidz = cuda_coop_cutlass_tid_z();",
        "  unsigned int ntx = cuda_coop_cutlass_ntid_x();",
        "  unsigned int nty = cuda_coop_cutlass_ntid_y();",
        "  return tidx + ntx * (tidy + nty * tidz);",
        "}",
        "static inline int cuda_coop_cutlass_group_threads() {",
        (
            "  int group_threads = "
            "(int)(cuda_coop_cutlass_ntid_x() * cuda_coop_cutlass_ntid_y() * "
            "cuda_coop_cutlass_ntid_z());"
        ),
        "  if (group_threads <= 0) {",
        "    group_threads = 1;",
        "  }",
        "  return group_threads;",
        "}",
        "static inline void* cuda_coop_cutlass_shared_ptr(",
        "    unsigned int shared_addr) {",
        "  unsigned long long shared_addr_u64 =",
        "      static_cast<unsigned long long>(shared_addr);",
        "  unsigned long long generic_addr;",
        '  asm("cvta.shared.u64 %0, %1;" : "=l"(generic_addr) : "l"(shared_addr_u64));',
        "  return reinterpret_cast<void*>(generic_addr);",
        "}",
    ]


def render_bundle_source(
    requests: list[Any],
    *,
    scope: str,
    include_lines: list[str] | tuple[str, ...] = (),
    multi_item_kinds: frozenset[str] = frozenset(),
    render_local_request: Callable[[Any], list[str]],
) -> str:
    """Render a CuTe provider bundle with provider-specific local requests.

    Providers pass their extra include lines, the request kinds that support
    multi-item ``ThreadData``, and a ``render_local_request`` callback for
    requests that are not handled by registered shared renderers. The callback
    must return C++ source lines for supported local request kinds and raise for
    unsupported kinds.
    """
    requests = list(canonical_bundle_requests(requests))
    preamble_lines = canonical_bundle_preamble_lines(
        [
            *include_lines,
            *bundle_include_lines(requests),
        ]
    )
    lines = bundle_source_preamble_lines(preamble_lines)

    for request in requests:
        if bundle_renderer_for(request) is not None:
            continue
        if request.items_per_thread != 1 and request.kind not in multi_item_kinds:
            raise NotImplementedError(
                f"{scope} provider does not yet support multi-item "
                "ThreadData shims; use items_per_thread=1"
            )

    for request in requests:
        external_renderer = bundle_renderer_for(request)
        if external_renderer is not None:
            lines.extend(external_renderer.render(request))
            continue
        lines.extend(render_local_request(request))

    lines.append("}")
    return "\n".join(lines) + "\n"
