# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Link CuTe caller LTO-IR and device libraries into an executable cubin."""

from __future__ import annotations

import shlex
from pathlib import Path


def link_lto(
    caller: bytes, paths: tuple[str, ...], arch: str, *, ptxas_options: str = ""
) -> bytes:
    """Link the caller and the ordered device libraries for the target SM."""

    from cuda.pathfinder import load_nvidia_dynamic_lib

    # Provider compilation has selected the toolkit matching the CUDA headers.
    # Let the bindings use that same DLL, including in wheel installations.
    load_nvidia_dynamic_lib("nvJitLink")
    from cuda.bindings import nvjitlink

    # The caller LTO-IR emitted by stock CuTe DSL 4.9 needs the CUDA 13.4
    # reader even when NVRTC compiled its provider using older headers.
    major, minor = nvjitlink.version()
    if (major, minor) < (13, 4):
        raise RuntimeError(
            "CuTe caller LTO-IR on Windows requires nvJitLink 13.4 or "
            f"newer, but the loaded library is {major}.{minor}"
        )

    # CuTe emits current NVVM IR even for targets whose default reader is
    # the legacy NVVM implementation (for example, sm_80).
    options = [f"-arch={arch}", "-lto", "-nvvm-version=nvvm-latest"]
    if ptxas_options:
        flags = shlex.shlex(ptxas_options, posix=True)
        flags.whitespace_split = True
        flags.commenters = ""
        flags.escape = ""  # Backslashes in Windows paths are literal.
        options.extend(f"-Xptxas={flag}" for flag in flags)
    try:
        handle = nvjitlink.create(len(options), options)
    except nvjitlink.nvJitLinkError as exc:
        raise RuntimeError(
            f"nvJitLink could not create a linker for {arch} "
            f"with options {options!r}: {exc}"
        ) from exc
    failed = True
    operation = "add the CuTe caller"
    try:
        nvjitlink.add_data(
            handle,
            nvjitlink.InputType.LTOIR,
            caller,
            len(caller),
            "caller.ltoir",
        )
        # Python opens Unicode paths with the Windows wide-character API.
        # Keep the input buffers alive through nvJitLinkComplete.
        libraries = [(path, path.read_bytes()) for path in map(Path, paths)]
        for library, data in libraries:
            operation = f"add {str(library)!r}"
            nvjitlink.add_data(
                handle, nvjitlink.InputType.ANY, data, len(data), library.name
            )

        operation = f"link device code for {arch}"
        nvjitlink.complete(handle)
        operation = "retrieve the linked cubin"
        cubin = bytearray(nvjitlink.get_linked_cubin_size(handle))
        nvjitlink.get_linked_cubin(handle, cubin)
        failed = False
        return bytes(cubin)
    except nvjitlink.nvJitLinkError as exc:
        try:
            log = bytearray(nvjitlink.get_error_log_size(handle))
            nvjitlink.get_error_log(handle, log)
            detail = log.rstrip(b"\0").decode("utf-8", errors="replace")
        except nvjitlink.nvJitLinkError as log_error:
            detail = f"Could not read the linker error log: {log_error}"
        raise RuntimeError(
            f"nvJitLink could not {operation}: {exc}\n{detail}".rstrip()
        ) from exc
    finally:
        try:
            nvjitlink.destroy(handle)
        except nvjitlink.nvJitLinkError:
            if not failed:
                raise
