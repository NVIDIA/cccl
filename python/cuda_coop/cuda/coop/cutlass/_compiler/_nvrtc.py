# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile CUTLASS provider bundles with NVRTC."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from typing import Any

from cutlass.base_dsl.common import DSLRuntimeError

import cuda.bindings.nvrtc as cuda_nvrtc
from cuda.coop._headers._toolkit import (
    preload_toolkit_compiler_libraries,
    validate_nvrtc_version,
)

from . import _cache as _cache_support
from ._bundle_contract import (
    BundleCacheIdentity,
    _decode_layout_probe_name,
    _PreparedLayoutProbes,
)
from ._cache import (
    _CachedBundle,
    _write_bundle_metadata,
    write_binary_atomic,
)

_COMPILE_PROGRAM_COUNTER = 0


def get_program_log(program: Any) -> str:
    err, log_size = cuda_nvrtc.nvrtcGetProgramLogSize(program)
    if err != cuda_nvrtc.nvrtcResult.NVRTC_SUCCESS or log_size <= 0:
        return ""
    log = bytearray(log_size)
    err = cuda_nvrtc.nvrtcGetProgramLog(program, log)[0]
    if err != cuda_nvrtc.nvrtcResult.NVRTC_SUCCESS:
        return ""
    return bytes(log).decode("utf-8", errors="replace").strip("\x00").strip()


def version_tuple(nvrtc: Any) -> tuple[int, int] | None:
    """Return the loaded NVRTC major/minor version when queryable."""

    version = getattr(nvrtc, "nvrtcVersion", None)
    if not callable(version):
        return None
    try:
        err, major, minor = version()
        parsed_version = int(major), int(minor)
    except (TypeError, ValueError):
        return None
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        return None
    return parsed_version


def get_version_tuple() -> tuple[int, int] | None:
    return version_tuple(cuda_nvrtc)


def get_version() -> str | None:
    loaded_version = get_version_tuple()
    if loaded_version is None:
        return None
    return f"{loaded_version[0]}.{loaded_version[1]}"


def preload_toolkit_nvrtc(include_dirs: Iterable[str]) -> None:
    """Load the toolkit NVRTC selected by the resolved CUDA headers."""

    try:
        libraries = preload_toolkit_compiler_libraries(include_dirs)
        actual_version = get_version_tuple()
        if actual_version is None:
            raise RuntimeError("loaded NVRTC did not report its version")
        validate_nvrtc_version(libraries, actual_version)
    except (OSError, RuntimeError) as exc:
        raise DSLRuntimeError(
            "Failed aligning provider NVRTC with the resolved CUDA headers.",
            cause=exc,
        ) from exc


def include_options(include_dirs: list[str]) -> list[bytes]:
    """Encode NVRTC include options with the platform filesystem codec."""

    return [os.fsencode(f"-I{path}") for path in include_dirs]


def compile_bundle(
    source: str,
    *,
    output_path: str,
    cache_identity: BundleCacheIdentity,
    compiler_options: tuple[str, ...],
    include_dirs: list[str],
    prepared_probes: _PreparedLayoutProbes,
    nvrtc_version: str | None,
    scope: str,
) -> _CachedBundle:
    """Compile one provider source to cached NVRTC LTO-IR."""

    global _COMPILE_PROGRAM_COUNTER

    source_bytes = source.encode("utf-8")
    options = [
        *(option.encode("ascii") for option in compiler_options),
        *include_options(include_dirs),
    ]
    err, program = cuda_nvrtc.nvrtcCreateProgram(
        source_bytes,
        b"cuda_coop_cutlass_bundle.cu",
        0,
        [],
        [],
    )
    if err != cuda_nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise DSLRuntimeError(
            f"Failed creating NVRTC program for {scope} provider bundle."
        )

    try:
        encoded_expressions = tuple(
            expression.encode("utf-8") for expression in prepared_probes.expressions
        )
        for expression in encoded_expressions:
            result = cuda_nvrtc.nvrtcAddNameExpression(program, expression)
            err = result[0] if isinstance(result, tuple) else result
            if err != cuda_nvrtc.nvrtcResult.NVRTC_SUCCESS:
                raise DSLRuntimeError(
                    f"Failed registering an NVRTC layout probe for {scope} "
                    "provider bundle."
                )

        err = cuda_nvrtc.nvrtcCompileProgram(program, len(options), options)[0]
        if err != cuda_nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise DSLRuntimeError(
                f"Failed compiling {scope} provider shim to LTO-IR.",
                cause=RuntimeError(get_program_log(program)),
            )
        with _cache_support._STATE_LOCK:
            _COMPILE_PROGRAM_COUNTER += 1

        layouts_by_expression = {}
        for expression, encoded_expression in zip(
            prepared_probes.expressions,
            encoded_expressions,
        ):
            err, lowered_name = cuda_nvrtc.nvrtcGetLoweredName(
                program,
                encoded_expression,
            )
            if err != cuda_nvrtc.nvrtcResult.NVRTC_SUCCESS:
                raise DSLRuntimeError(
                    f"Failed retrieving an NVRTC layout probe for {scope} "
                    "provider bundle."
                )
            layouts_by_expression[expression] = _decode_layout_probe_name(
                lowered_name,
                symbol=prepared_probes.symbol,
                expression=expression,
            )

        err, blob_size = cuda_nvrtc.nvrtcGetLTOIRSize(program)
        if err != cuda_nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise DSLRuntimeError(
                f"Failed querying NVRTC LTO-IR size for {scope} provider shim."
            )
        if blob_size <= 0:
            raise DSLRuntimeError(
                f"NVRTC produced an empty LTO-IR artifact for {scope} provider shim."
            )
        artifact_blob = bytearray(blob_size)
        err = cuda_nvrtc.nvrtcGetLTOIR(program, artifact_blob)[0]
        if err != cuda_nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise DSLRuntimeError(
                f"Failed retrieving NVRTC LTO-IR for {scope} provider shim."
            )
    finally:
        cuda_nvrtc.nvrtcDestroyProgram(program)

    cached = _CachedBundle(
        path=output_path,
        layouts_by_expression=layouts_by_expression,
        producer_compiler="nvrtc",
        producer_compiler_version=nvrtc_version,
        artifact_size=len(artifact_blob),
        artifact_sha256=hashlib.sha256(artifact_blob).hexdigest(),
    )
    write_binary_atomic(output_path, artifact_blob, scope=scope)
    _write_bundle_metadata(
        output_path,
        artifact_blob,
        cached,
        cache_identity,
        scope=scope,
    )
    return cached


def reset_compile_state() -> None:
    global _COMPILE_PROGRAM_COUNTER
    with _cache_support._STATE_LOCK:
        _COMPILE_PROGRAM_COUNTER = 0


def get_compile_program_counter() -> int:
    with _cache_support._STATE_LOCK:
        return _COMPILE_PROGRAM_COUNTER
