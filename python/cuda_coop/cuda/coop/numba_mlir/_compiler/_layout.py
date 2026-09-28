# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Recover C++ storage layouts from NVRTC template-name expressions."""

import hashlib
import re


def prepare_layout_queries(cpp: str, layout_types: tuple[str, ...]):
    if not layout_types:
        return cpp, "", ()
    if any(not storage_type.strip() for storage_type in layout_types):
        raise ValueError("layout types must be non-empty")
    digest = hashlib.sha256(repr(layout_types).encode("utf-8")).hexdigest()
    symbol = f"cuda_coop_layout_probe_{digest}"
    source = (
        f"{cpp.rstrip()}\n\n"
        "template <unsigned long long Size, unsigned long long Alignment>\n"
        f"__device__ unsigned char {symbol} = 0;\n"
    )
    queries = tuple(
        f"&{symbol}<(sizeof({storage_type})), (alignof({storage_type}))>"
        for storage_type in layout_types
    )
    return source, symbol, queries


def decode_layout_name(
    lowered_name: bytes | str, *, symbol: str, expression: str
) -> tuple[int, int]:
    if isinstance(lowered_name, bytes):
        lowered_name = lowered_name.decode("ascii")
    match = re.fullmatch(
        rf"_Z{len(symbol)}{re.escape(symbol)}ILy([0-9]+)ELy([0-9]+)EE",
        lowered_name.rstrip("\0"),
    )
    if match is None:
        raise ValueError(
            "NVRTC returned an unexpected lowered layout-probe name for "
            f"{expression!r}: {lowered_name!r}"
        )
    size, alignment = (int(value) for value in match.groups())
    if (
        size <= 0
        or alignment <= 0
        or alignment & (alignment - 1)
        or size % alignment
    ):
        raise ValueError(
            f"Invalid storage layout for {expression!r}: "
            f"size={size}, alignment={alignment}"
        )
    return size, alignment
