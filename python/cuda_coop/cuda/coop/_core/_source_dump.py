# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Optional CUDA source diagnostics shared by compiler backends."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path


def dump_source(
    source: str,
    *,
    backend: str,
    identity: tuple[object, ...] = (),
) -> Path | None:
    """Write a backend-tagged, content-addressed CUDA translation unit.

    CUDA_COOP_SOURCE_DUMP_DIR selects the directory. Unset or empty disables
    dumping.
    Call before compiler/cache lookup to capture sources on cache hits too.
    """

    dump_dir = os.environ.get("CUDA_COOP_SOURCE_DUMP_DIR")
    if not dump_dir:
        return None

    contents = source.encode("utf-8")
    digest = hashlib.sha256()
    digest.update(repr(identity).encode("utf-8", errors="surrogateescape"))
    digest.update(b"\0")
    digest.update(contents)
    root = Path(dump_dir).expanduser().resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination = root / f"cuda_coop_{backend}_{digest.hexdigest()}.cu"
    if destination.is_file():
        return destination

    fd, temporary = tempfile.mkstemp(dir=root, prefix=f".{destination.name}.")
    try:
        with os.fdopen(fd, "wb") as source_file:
            source_file.write(contents)
            source_file.flush()
            os.fsync(source_file.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination
