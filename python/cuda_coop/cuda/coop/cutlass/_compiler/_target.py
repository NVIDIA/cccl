# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Select the target and artifact format for CUTLASS provider compilation.

This module translates CUTLASS launch configuration into the exact NVRTC or
Clang target used by bundle orchestration. It does not compile or cache an
artifact.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from cutlass.base_dsl.common import DSLRuntimeError
from cutlass.base_dsl.compiler import GPUArch

BUNDLE_FORMAT_ENV = "CUDA_COOP_CUTLASS_PROVIDER_BUNDLE_FORMAT"


def configured_gpu_arch(get_cute_dsl: Callable[[], Any]) -> str:
    dsl = get_cute_dsl()
    compile_options = getattr(dsl, "compile_options", None)
    options = getattr(compile_options, "options", None)
    option = None
    if options is not None:
        if hasattr(options, "get"):
            option = options.get(GPUArch)
        else:
            try:
                option = options[GPUArch]
            except (KeyError, TypeError):
                option = None
    option_arch = getattr(option, "value", None)
    if option_arch is not None:
        arch = str(option_arch).strip()
        if arch:
            return arch
    environment_arch = getattr(getattr(dsl, "envar", None), "arch", None)
    return "" if environment_arch is None else str(environment_arch).strip()


def _is_numeric_arch(arch: str) -> bool:
    if arch.isdigit():
        return True
    return bool(arch and arch[-1] in ("a", "f") and arch[:-1].isdigit())


def _configured_arch_suffix(scope: str, arch: str) -> str:
    original = arch
    for prefix in ("compute_", "compute", "sm_", "sm"):
        if arch.startswith(prefix):
            arch = arch[len(prefix) :]
            break
    if _is_numeric_arch(arch):
        return arch
    raise DSLRuntimeError(
        f"Invalid configured CUDA arch {original!r} for {scope} provider; "
        "expected digits with an optional 'a' or 'f' suffix, optionally "
        "prefixed by 'sm' or 'compute'."
    )


def resolve_nvrtc_arch(
    scope: str,
    configured_arch: Callable[[], str],
) -> str:
    """Return the ``compute_*`` target for NVRTC LTO-IR compilation."""

    arch = configured_arch()
    if arch:
        return f"compute_{_configured_arch_suffix(scope, arch)}"

    from cutlass.base_dsl.runtime import cuda as cuda_runtime

    major, minor = cuda_runtime.get_compute_capability_major_minor()
    if major is None or minor is None:
        raise DSLRuntimeError(
            f"Unable to resolve CUDA arch for {scope} provider NVRTC bundle "
            "compilation."
        )
    return f"compute_{major}{minor}"


def resolve_nvrtc_sm_arch(
    scope: str,
    configured_arch: Callable[[], str],
) -> str:
    """Return the ``sm_*`` target used for final linking."""

    arch = configured_arch()
    if arch:
        return f"sm_{_configured_arch_suffix(scope, arch)}"

    from cutlass.base_dsl.runtime import cuda as cuda_runtime

    major, minor = cuda_runtime.get_compute_capability_major_minor()
    if major is None or minor is None:
        raise DSLRuntimeError(
            f"Unable to resolve CUDA arch for {scope} provider NVRTC bundle "
            "compilation."
        )
    return f"sm_{major}{minor}"


def select_bundle_format(scope: str) -> str:
    """Return the configured provider artifact format."""

    bundle_format = os.environ.get(BUNDLE_FORMAT_ENV, "auto").strip().lower()
    if bundle_format == "bc":
        return "bc"
    if bundle_format in ("lto", "ltoir"):
        return "ltoir"
    if bundle_format == "cubin":
        raise DSLRuntimeError(
            f"Unsupported {BUNDLE_FORMAT_ENV} value: {bundle_format!r}. "
            f"{scope} external provider shims require LTO-IR or LLVM bitcode."
        )
    if bundle_format != "auto":
        raise DSLRuntimeError(
            f"Unsupported {BUNDLE_FORMAT_ENV} value: {bundle_format!r}. "
            "Use auto/bc/ltoir."
        )
    return "ltoir"
