# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
CCLB — compact binary container for pre-compiled CCCL algorithm objects.

File layout
-----------
Offset  Size  Content
──────  ────  ───────────────────────────────────────────────
0       4     Magic: b"CCLB"
4       4     Version: little-endian uint32 (currently 1)
8       4     algo_len: little-endian uint32
12      *     Algorithm name UTF-8 bytes ("reduce" | "merge_sort")
*       4     meta_len: little-endian uint32
*       *     JSON-encoded metadata (all scalar/string fields; no cubin)
*       8     cubin_len: little-endian uint64
*       *     Raw cubin bytes

The cubin is kept out of the JSON so it remains a compact binary blob
rather than a base64-encoded string.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

MAGIC = b"CCLB"
VERSION = 1


def write_cclb(path: Path, algorithm: str, meta: dict, cubin: bytes) -> None:
    """Write *meta* + *cubin* to *path* in CCLB format.

    *meta* must not contain the ``"cubin"`` key; pass that separately as
    *cubin*.
    """
    name_bytes = algorithm.encode()
    meta_bytes = json.dumps(meta, separators=(",", ":")).encode()
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", len(name_bytes)))
        f.write(name_bytes)
        f.write(struct.pack("<I", len(meta_bytes)))
        f.write(meta_bytes)
        f.write(struct.pack("<Q", len(cubin)))
        f.write(cubin)


def read_cclb(path: Path) -> tuple[str, dict]:
    """Read a CCLB file and return ``(algorithm_name, data_dict)``.

    The returned dict has the same shape as ``_serialize()`` — i.e. the
    ``"cubin"`` key is present as ``bytes`` — so it can be passed directly
    to ``_from_serialized()``.

    Raises ``ValueError`` if the file does not start with the CCLB magic.
    """
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(
                f"{path}: not a CCLB file (magic={magic!r}, expected {MAGIC!r})"
            )
        (version,) = struct.unpack("<I", f.read(4))
        if version != VERSION:
            raise ValueError(
                f"{path}: unsupported CCLB version {version} (expected {VERSION})"
            )
        (name_len,) = struct.unpack("<I", f.read(4))
        algorithm = f.read(name_len).decode()
        (meta_len,) = struct.unpack("<I", f.read(4))
        meta = json.loads(f.read(meta_len))
        (cubin_len,) = struct.unpack("<Q", f.read(8))
        cubin = f.read(cubin_len)

    meta["cubin"] = cubin
    return algorithm, meta
