# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Link CuTe caller LTO-IR and device libraries into an executable cubin."""

from __future__ import annotations

import shlex


def link_lto(
    caller: bytes, paths: tuple[str, ...], arch: str, *, ptxas_options: str = ""
) -> bytes:
    """Link the caller and the ordered device libraries for the target SM."""

    from cuda.pathfinder import load_nvidia_dynamic_lib

    # Provider compilation has selected the toolkit matching the CUDA headers.
    # Let the bindings use that same DLL, including in wheel installations.
    load_nvidia_dynamic_lib("nvJitLink")
    from cuda.bindings import nvjitlink

    # CuTe emits current NVVM IR even for targets whose default reader is
    # the legacy NVVM implementation (for example, sm_80).
    options = [f"-arch={arch}", "-lto", "-nvvm-version=nvvm-latest"]
    if ptxas_options:
        flags = shlex.shlex(ptxas_options, posix=True)
        flags.whitespace_split = True
        flags.commenters = ""
        flags.escape = ""  # Backslashes in Windows paths are literal.
        options.extend(f"-Xptxas={flag}" for flag in flags)
    handle = nvjitlink.create(len(options), options)
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
        for path in paths:
            operation = f"add {path!r}"
            nvjitlink.add_file(handle, nvjitlink.InputType.ANY, path)

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
