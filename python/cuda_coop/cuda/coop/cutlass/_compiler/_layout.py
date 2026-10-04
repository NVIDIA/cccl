# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Recover C++ scratch sizes and alignments from NVRTC template names.

CuTe must allocate scratch that matches the compiled C++ primitive. Each
probe places its size and alignment expressions in a variable template.
NVRTC evaluates these expressions as template arguments while compiling the
provider bundle. The NVRTC compile step registers each instantiation with
``nvrtcAddNameExpression`` and reads its mangled C++ symbol through
``nvrtcGetLoweredName``. This lowered name contains both values as decimal
integers. This module decodes that name, so the layout needs no GPU execution
or separate metadata link.
"""

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
    """Keep the compiled provider and its resolved scratch layouts together.

    ``path`` names the cached LTO-IR file for CuTe linking. ``layouts`` maps
    caller keys to byte sizes and alignments from that compilation. These
    keys match traced calls to their storage requirements at finalization.
    """

    path: str
    layouts: dict[Hashable, ScratchLayout]


@dataclass(frozen=True)
class _PreparedLayoutProbes:
    """Keep generated queries separate from their caller-facing identities.

    ``source`` includes the probe template when any probe exists.
    ``expressions`` contains each unique NVRTC name expression once.
    ``key_to_expression`` maps every caller key to its query, so equivalent
    layouts can share one compiler query. ``symbol`` identifies the template
    accepted by the name decoder.
    """

    source: str
    expressions: tuple[str, ...]
    key_to_expression: dict[Hashable, str]
    symbol: str


def _prepare_layout_probes(
    source: str,
    layout_probes: Iterable[ScratchLayoutProbe],
) -> _PreparedLayoutProbes:
    """Add deterministic layout queries to a provider translation unit.

    Parameters
    ----------
    source : str
        C++ provider source in which the probe expressions are valid.
    layout_probes : iterable of ScratchLayoutProbe
        Caller keys and C++ size/alignment expressions. A repeated key must
        have the same expressions. Distinct keys may share an expression pair.

    Returns
    -------
    _PreparedLayoutProbes
        Source with a variable template and sorted, deduplicated queries.
        With no probes, the source is unchanged and query fields are empty.
        This function prepares source; NVRTC evaluates it during compilation.

    Raises
    ------
    TypeError
        A probe has the wrong type or its requirement key is not hashable.
    ValueError
        An expression is empty or one key has conflicting expressions.
    """

    probes_by_key: dict[Hashable, tuple[str, str]] = {}
    for probe in layout_probes:
        if not isinstance(probe, ScratchLayoutProbe):
            raise TypeError(
                "layout_probes must contain ScratchLayoutProbe values"
            )
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
        probe: f"&{symbol}<({probe[0]}), ({probe[1]})>"
        for probe in unique_probes
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
    """Reject invalid byte layouts before the planner uses them.

    ``size_in_bytes`` and ``alignment`` must be positive integers, excluding
    booleans. Alignment must be a power of two and divide the size. The helper
    returns a ``ScratchLayout`` or raises ``ValueError`` with ``description``
    to identify the compiler query or cached entry that failed validation.
    """

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
    """Read evaluated size and alignment from one registered probe name.

    ``lowered_name`` is the probe's mangled C++ symbol, as a string or UTF-8
    bytes; it can end in a NUL. Only the expected unsigned-integer template
    encoding for ``symbol`` is accepted. ``expression`` identifies the query
    in errors. Unexpected names or invalid byte layouts raise ``ValueError``;
    malformed UTF-8 bytes raise ``UnicodeDecodeError``.
    """

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


__all__ = [
    "BundleCompilation",
    "_PreparedLayoutProbes",
    "_decode_layout_probe_name",
    "_prepare_layout_probes",
    "_validate_storage_layout",
]
