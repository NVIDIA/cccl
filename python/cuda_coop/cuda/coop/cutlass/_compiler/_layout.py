# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Recover exact C++ storage layouts from NVRTC template-name expressions."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from typing import Any

from ._types import ScratchLayout, ScratchLayoutProbe


@dataclass(frozen=True)
class BundleCompilation:
    path: str
    layouts: dict[Hashable, ScratchLayout]


@dataclass(frozen=True)
class _PreparedLayoutProbes:
    source: str
    expressions: tuple[str, ...]
    key_to_expression: dict[Hashable, str]
    symbol: str


def _prepare_layout_probes(
    source: str,
    layout_probes: Iterable[ScratchLayoutProbe],
) -> _PreparedLayoutProbes:
    probes_by_key: dict[Hashable, tuple[str, str]] = {}
    for probe in layout_probes:
        if not isinstance(probe, ScratchLayoutProbe):
            raise TypeError("layout_probes must contain ScratchLayoutProbe values")
        try:
            hash(probe.requirement_key)
        except TypeError as exc:
            raise TypeError("layout probe keys must be hashable") from exc
        size_expression = probe.size_expression.strip()
        alignment_expression = probe.alignment_expression.strip()
        if not size_expression or not alignment_expression:
            raise ValueError("layout probe expressions must be non-empty")
        expressions = (size_expression, alignment_expression)
        existing = probes_by_key.get(probe.requirement_key)
        if existing is not None and existing != expressions:
            raise ValueError(
                f"conflicting layout probes for key {probe.requirement_key!r}"
            )
        probes_by_key[probe.requirement_key] = expressions

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
                "version": 1,
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
) -> ScratchLayout:
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
    return ScratchLayout(size_in_bytes=size_in_bytes, alignment=alignment)


def _decode_layout_probe_name(
    lowered_name: bytes | str,
    *,
    symbol: str,
    expression: str,
) -> ScratchLayout:
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
