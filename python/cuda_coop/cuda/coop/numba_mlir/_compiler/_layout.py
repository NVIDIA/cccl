# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Read C++ size and alignment from names produced by NVRTC.

The compiler knows a provider's scratch layout after template instantiation.
A small variable template puts ``sizeof`` and ``alignof`` values into its
symbol name. Its address is registered as an NVRTC name expression. After
compilation, ``nvrtcGetLoweredName`` returns the mangled name containing both
values. The same compilation builds the provider image, so no separate
metadata program is compiled or linked. The layout stays tied to the
compiled source and target.
"""

import hashlib
import re


def prepare_layout_queries(
    cpp: str, layout_types: tuple[str, ...]
) -> tuple[str, str, tuple[str, ...]]:
    """Append a probe that encodes each requested layout in a template name.

    The probe's two ``unsigned long long`` template arguments are the byte
    size and alignment. Its name includes a digest of the ordered type names
    so the same request produces the same source and cache identity. This
    function only prepares source; NVRTC later evaluates the expressions.

    Parameters
    ----------
    cpp : str
        Provider translation unit containing the type definitions or aliases.
    layout_types : tuple of str
        Non-empty C++ type expressions visible at the end of ``cpp``. The
        returned queries keep their order and duplicates, so each layout
        matches its requested type.

    Returns
    -------
    tuple
        ``(source, symbol, queries)`` with the probe appended to the source,
        its symbol name, and one address expression per requested type. An
        empty tuple returns ``(cpp, "", ())`` and preserves the source.

    Raises
    ------
    ValueError
        A type expression is empty or contains only whitespace. Other C++
        syntax and type errors are left to NVRTC compilation.
    """

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
    """Decode one probe's compiler-generated name into a byte layout.

    Accept only the expected symbol followed by two unsigned-long-long
    template values in NVRTC's mangled-name encoding. This is a decoder for
    ``prepare_layout_queries`` probes, not a general C++ demangler. The
    function validates size and alignment before returning them for use in
    scratch allocation.

    Parameters
    ----------
    lowered_name : bytes or str
        Name returned by ``nvrtcGetLoweredName`` after compilation. Bytes
        must be ASCII. Trailing NUL characters are ignored for both bytes
        and str inputs.
    symbol : str
        Exact probe symbol returned by ``prepare_layout_queries``.
    expression : str
        Original NVRTC name expression, included in failure diagnostics.

    Returns
    -------
    tuple of int
        ``(size, alignment)`` in bytes. Both values are positive. Alignment
        is a power of two, and size is a multiple of alignment.

    Raises
    ------
    ValueError
        The name has an unexpected symbol or encoding, or the decoded layout
        fails these checks. Non-ASCII bytes also fail decoding.
    """

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
