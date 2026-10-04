# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Save generated CUDA source when source inspection is enabled.

Backends call this before cache lookup so cached compilations remain
inspectable. The backend, source, and compile identity determine the
filename. An atomic write keeps concurrent calls from leaving a partial
source file.
"""

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
    """Write generated CUDA source to a file named by its content and backend.

    ``CUDA_COOP_SOURCE_DUMP_DIR`` selects the directory and is read on every
    call; unset or empty disables dumping. Other values are directory names,
    including strings such as ``"0"`` or ``"false"``. Expand ``~`` and resolve
    relative paths against the current working directory when the call runs.
    Environment changes affect subsequent dumps without reimporting a backend.

    Call before compiler or cache lookup to capture sources on cache hits too.
    The filename combines the backend tag with a hash of the identity
    representation and source bytes, allowing different targets and generated
    variants to coexist. An existing file is reused without rewriting it;
    changing or unsetting the directory does not remove previous dumps.

    Publish through a flushed and synced temporary file in the same directory
    followed by atomic replacement, so concurrent dumps leave complete files.
    Filesystem failures propagate to the caller.

    Parameters
    ----------
    source : str
        Complete generated CUDA source, written as UTF-8 without modification.
    backend : str
        Backend label embedded directly in the filename, such as
        ``"numba_mlir"``. Callers supply a filename-safe label.
    identity : tuple of object, optional
        Extra compile identity, such as target architecture and output kind.
        Its ``repr`` contributes to the digest. Use reproducible
        representations so repeated compilations select the same path.

    Returns
    -------
    pathlib.Path or None
        Absolute path of the existing or newly written source file, or
        ``None`` when dumping is disabled.

    Raises
    ------
    OSError
        The destination directory or source file cannot be created or written.
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
