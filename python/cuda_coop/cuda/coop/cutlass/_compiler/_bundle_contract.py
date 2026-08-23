# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


"""Stable bundle identities, schemas, and layout contracts."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from cutlass.base_dsl.common import DSLRuntimeError

from cuda.coop._headers._identity import (
    HeaderIdentityError,
    IncludeDirsIdentity,
)
from cuda.coop._headers._identity import (
    include_dirs_identity as resolve_include_dirs_identity,
)

if os.name == "nt":
    pass
else:
    pass

NVVM_VERSION_OPT = b"-nvvm-version=nvvm-latest"

BUNDLE_CACHE_SCHEMA_VERSION = 2

PROVIDER_BUNDLE_ABI_VERSION = 1

BUNDLE_METADATA_VERSION = 2

LAYOUT_METADATA_VERSION = BUNDLE_METADATA_VERSION

RESOLUTION_ROUTE_PRECOMPILED = "precompiled"

RESOLUTION_ROUTE_MEMORY = "memory"

RESOLUTION_ROUTE_DISK = "disk"

RESOLUTION_ROUTE_CLANG = "clang"

RESOLUTION_ROUTE_NVRTC = "nvrtc"


@dataclass(frozen=True)
class LayoutProbe:
    """C++ constant expressions whose values describe one storage layout."""

    key: Hashable
    size_expression: str
    alignment_expression: str


@dataclass(frozen=True)
class StorageLayout:
    size_in_bytes: int
    alignment: int


@dataclass(frozen=True)
class BundleCompilation:
    path: str
    layouts: dict[Hashable, StorageLayout]


@dataclass(frozen=True)
class BundleIdentity:
    """Portable identity for one provider source and compilation contract."""

    provider_abi_version: int
    source_hash: str
    bundle_format: str
    bundle_arch: str
    bundle_sm_arch: str
    compiler_options: tuple[str, ...]
    layout_expressions: tuple[str, ...]


@dataclass(frozen=True)
class BundleCacheIdentity:
    """Mutable-header identity used only by the existing JIT cache."""

    schema_version: int
    bundle: BundleIdentity
    include_key: str
    producer_compiler_version: str

    @property
    def cache_key(self) -> str:
        return (
            f"v{self.schema_version}:{self.bundle.source_hash}:"
            f"{self.bundle.bundle_format}:"
            f"{self.bundle.bundle_arch}:{self.bundle.bundle_sm_arch}:"
            f"{self.producer_compiler_version}:{self.include_key}"
        )

    @property
    def artifact_stem(self) -> str:
        compiler_version = re.sub(
            r"[^A-Za-z0-9_-]+",
            "_",
            self.producer_compiler_version,
        )
        return (
            f"bundle_v{self.schema_version}_{self.bundle.source_hash}_"
            f"{self.bundle.bundle_arch}_{self.bundle.bundle_sm_arch}_"
            f"{compiler_version}_{self.include_key}"
        )


@dataclass(frozen=True)
class BundleResolutionRequest:
    """Canonical source and layout metadata presented to bundle resolvers."""

    identity: BundleIdentity
    source: str


@dataclass(frozen=True)
class BundleResolution:
    """One resolved provider artifact and its exact layout metadata."""

    request: BundleResolutionRequest
    path: str
    layouts_by_expression: Mapping[str, StorageLayout]
    route: str
    producer_compiler: str | None
    producer_compiler_version: str | None
    producer_toolkit_version: str | None
    phase_timings_ns: Mapping[str, int]


@dataclass(frozen=True)
class BundleTelemetry:
    """Internal snapshot of provider resolution counts and phase timings."""

    route_counts: Mapping[str, int]
    phase_counts: Mapping[str, int]
    phase_timings_ns: Mapping[str, int]


@dataclass(frozen=True)
class _PreparedLayoutProbes:
    source: str
    expressions: tuple[str, ...]
    key_to_expression: dict[Hashable, str]
    symbol: str


def include_dirs_identity(include_dirs: list[str]) -> IncludeDirsIdentity:
    try:
        return resolve_include_dirs_identity(include_dirs)
    except HeaderIdentityError as exc:
        raise DSLRuntimeError(
            "Failed fingerprinting provider include paths.",
            cause=exc,
        ) from exc


def include_dirs_cache_key(include_dirs: list[str]) -> str:
    return include_dirs_identity(include_dirs).digest


def bundle_compiler_options(
    bundle_format: str,
    bundle_arch: str,
) -> tuple[str, ...]:
    if bundle_format == "bc":
        return (
            "--target=nvptx64-nvidia-cuda",
            "-std=c++17",
            "-O3",
            "-emit-llvm",
            "-c",
        )
    return (
        "--std=c++17",
        "--relocatable-device-code=true",
        "-default-device",
        NVVM_VERSION_OPT.decode("ascii"),
        f"--gpu-architecture={bundle_arch}",
        "-dlto",
    )


def make_bundle_identity(
    *,
    source_hash: str,
    bundle_format: str,
    bundle_arch: str,
    bundle_sm_arch: str,
    compiler_options: tuple[str, ...],
    layout_expressions: tuple[str, ...],
) -> BundleIdentity:
    return BundleIdentity(
        provider_abi_version=PROVIDER_BUNDLE_ABI_VERSION,
        source_hash=source_hash,
        bundle_format=bundle_format,
        bundle_arch=bundle_arch,
        bundle_sm_arch=bundle_sm_arch,
        compiler_options=compiler_options,
        layout_expressions=layout_expressions,
    )


def make_bundle_cache_identity(
    bundle: BundleIdentity,
    *,
    include_key: str,
    producer_compiler_version: str,
) -> BundleCacheIdentity:
    return BundleCacheIdentity(
        schema_version=BUNDLE_CACHE_SCHEMA_VERSION,
        bundle=bundle,
        include_key=include_key,
        producer_compiler_version=producer_compiler_version,
    )


def _prepare_layout_probes(
    source: str,
    layout_probes: Iterable[LayoutProbe],
) -> _PreparedLayoutProbes:
    probes_by_key: dict[Hashable, tuple[str, str]] = {}
    for probe in layout_probes:
        if not isinstance(probe, LayoutProbe):
            raise TypeError("layout_probes must contain LayoutProbe values")
        try:
            hash(probe.key)
        except TypeError as exc:
            raise TypeError("layout probe keys must be hashable") from exc
        size_expression = probe.size_expression.strip()
        alignment_expression = probe.alignment_expression.strip()
        if not size_expression or not alignment_expression:
            raise ValueError("layout probe expressions must be non-empty")
        expressions = (size_expression, alignment_expression)
        existing = probes_by_key.get(probe.key)
        if existing is not None and existing != expressions:
            raise ValueError(f"conflicting layout probes for key {probe.key!r}")
        probes_by_key[probe.key] = expressions

    if not probes_by_key:
        return _PreparedLayoutProbes(
            source=source,
            expressions=(),
            key_to_expression={},
            symbol="",
        )

    unique_probes = sorted(set(probes_by_key.values()))
    probe_digest = hashlib.sha256(
        json.dumps(
            {
                "version": LAYOUT_METADATA_VERSION,
                "probes": unique_probes,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    symbol = f"cuda_coop_layout_probe_{probe_digest}"
    source = (
        f"{source.rstrip()}\n\n"
        "template <unsigned long long Size, unsigned long long Alignment>\n"
        f"__device__ unsigned char {symbol} = 0;\n"
    )
    expression_by_probe = {
        probe: f"&{symbol}<({probe[0]}), ({probe[1]})>" for probe in unique_probes
    }
    key_to_expression = {
        key: expression_by_probe[probe] for key, probe in probes_by_key.items()
    }
    return _PreparedLayoutProbes(
        source=source,
        expressions=tuple(sorted(expression_by_probe.values())),
        key_to_expression=key_to_expression,
        symbol=symbol,
    )


def _validate_storage_layout(
    size_in_bytes: Any,
    alignment: Any,
    *,
    description: str,
) -> StorageLayout:
    if (
        not isinstance(size_in_bytes, int)
        or isinstance(size_in_bytes, bool)
        or not isinstance(alignment, int)
        or isinstance(alignment, bool)
        or size_in_bytes <= 0
        or alignment <= 0
        or alignment & (alignment - 1)
        or size_in_bytes % alignment != 0
    ):
        raise ValueError(
            f"Invalid storage layout for {description}: "
            f"size={size_in_bytes!r}, alignment={alignment!r}."
        )
    return StorageLayout(size_in_bytes=size_in_bytes, alignment=alignment)


def _decode_layout_probe_name(
    lowered_name: bytes | str,
    *,
    symbol: str,
    expression: str,
) -> StorageLayout:
    if isinstance(lowered_name, bytes):
        lowered_name = lowered_name.decode("utf-8", errors="strict")
    lowered_name = lowered_name.rstrip("\0")
    match = re.fullmatch(
        rf"_Z{len(symbol)}{re.escape(symbol)}ILy([0-9]+)ELy([0-9]+)EE",
        lowered_name,
    )
    if match is None:
        raise ValueError(
            "NVRTC returned an unexpected lowered layout-probe name for "
            f"{expression!r}: {lowered_name!r}."
        )
    return _validate_storage_layout(
        int(match.group(1)),
        int(match.group(2)),
        description=expression,
    )
