# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile CUTLASS provider bundles with NVRTC."""

from __future__ import annotations

from typing import Any

from cutlass.base_dsl.common import DSLRuntimeError

import cuda.bindings.nvrtc as cuda_nvrtc

from ._bundle_contract import (
    BundleCacheIdentity,
    _decode_layout_probe_name,
    _PreparedLayoutProbes,
)
from ._cache import (
    _STATE_LOCK,
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


def get_version() -> str | None:
    version = getattr(cuda_nvrtc, "nvrtcVersion", None)
    if not callable(version):
        return None
    try:
        err, major, minor = version()
    except (TypeError, ValueError):
        return None
    if err != cuda_nvrtc.nvrtcResult.NVRTC_SUCCESS:
        return None
    return f"{major}.{minor}"


def include_options(include_dirs: list[str]) -> list[bytes]:
    return [f"-I{path}".encode() for path in include_dirs]


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
        with _STATE_LOCK:
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
    with _STATE_LOCK:
        _COMPILE_PROGRAM_COUNTER = 0


def get_compile_program_counter() -> int:
    with _STATE_LOCK:
        return _COMPILE_PROGRAM_COUNTER
