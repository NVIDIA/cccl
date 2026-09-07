# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Preload and compile CUTLASS provider bundles with NVRTC.

The module owns the NVRTC program lifecycle, PCH retry path, and LTO-IR
artifact production. Bundle orchestration decides when this backend runs.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Iterable
from typing import Any

from cutlass.base_dsl.common import DSLRuntimeError

import cuda.bindings.nvrtc as cuda_nvrtc
from cuda.coop._headers._toolkit import (
    preload_toolkit_compiler_libraries,
    validate_nvrtc_version,
)

# PCH must register its child reset before cache registers its fork handlers.
# isort: off
from . import (
    _pch as _provider_pch,
    _cache as _cache_support,
)

# isort: on
from ._bundle_contract import (
    BundleCacheIdentity,
    _decode_layout_probe_name,
    _PreparedLayoutProbes,
)

_COMPILE_PROGRAM_COUNTER = 0


def program_log(nvrtc: Any, program: Any) -> str:
    """Return a decoded NVRTC program log when the runtime exposes it."""

    get_log_size = getattr(nvrtc, "nvrtcGetProgramLogSize", None)
    get_log = getattr(nvrtc, "nvrtcGetProgramLog", None)
    if not callable(get_log_size) or not callable(get_log):
        return ""
    err, log_size = get_log_size(program)
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS or log_size <= 0:
        return ""
    log = bytearray(log_size)
    err = get_log(program, log)[0]
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
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


def get_program_log(program: Any) -> str:
    return program_log(cuda_nvrtc, program)


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


def _create_program(
    source_bytes: bytes,
    encoded_expressions: tuple[bytes, ...],
    *,
    scope: str,
) -> Any:
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
        for expression in encoded_expressions:
            result = cuda_nvrtc.nvrtcAddNameExpression(program, expression)
            err = result[0] if isinstance(result, tuple) else result
            if err != cuda_nvrtc.nvrtcResult.NVRTC_SUCCESS:
                raise DSLRuntimeError(
                    f"Failed registering an NVRTC layout probe for {scope} "
                    "provider bundle."
                )
    except BaseException:
        cuda_nvrtc.nvrtcDestroyProgram(program)
        raise
    return program


def _compile_program(
    program: Any,
    options: list[bytes],
) -> tuple[Any, int]:
    global _COMPILE_PROGRAM_COUNTER

    started_ns = time.perf_counter_ns()
    with _cache_support._STATE_LOCK:
        _COMPILE_PROGRAM_COUNTER += 1
    result = cuda_nvrtc.nvrtcCompileProgram(program, len(options), options)
    duration_ns = max(0, time.perf_counter_ns() - started_ns)
    err = result[0] if isinstance(result, tuple) else result
    return err, duration_ns


def compile_bundle(
    source: str,
    *,
    output_path: str,
    cache_identity: BundleCacheIdentity,
    compiler_options: tuple[str, ...],
    include_dirs: list[str],
    prepared_probes: _PreparedLayoutProbes,
    nvrtc_version: str | None,
    nvrtc_version_tuple: tuple[int, int] | None,
    bundle_arch: str,
    bundle_sm_arch: str,
    header_identity: str,
    phase_timings_ns: dict[str, int],
    scope: str,
) -> _cache_support._CachedBundle:
    """Compile one provider source to cached NVRTC LTO-IR."""

    source_bytes = source.encode("utf-8")
    base_options = [
        *(option.encode("ascii") for option in compiler_options),
        *include_options(include_dirs),
    ]
    encoded_expressions = tuple(
        expression.encode("utf-8") for expression in prepared_probes.expressions
    )
    pch_session: _provider_pch.PCHSession | None = None
    program = None
    try:
        with _provider_pch.pch_session(
            nvrtc_version=nvrtc_version_tuple,
            bundle_arch=bundle_arch,
            bundle_sm_arch=bundle_sm_arch,
            compiler_options=compiler_options,
            include_dirs=tuple(include_dirs),
            header_identity=header_identity,
            preamble_identity=_provider_pch.provider_preamble_identity(source),
        ) as pch_session:
            program = _create_program(
                source_bytes,
                encoded_expressions,
                scope=scope,
            )
            options = [*base_options, *pch_session.options]
            err, compile_duration_ns = _compile_program(program, options)
            pch_fallback_log = ""
            used_pch = pch_session.enabled
            if err != cuda_nvrtc.nvrtcResult.NVRTC_SUCCESS and pch_session.enabled:
                pch_fallback_log = get_program_log(program)
                pch_session.disable_after_failure(compile_duration_ns)
                cuda_nvrtc.nvrtcDestroyProgram(program)
                program = None
                program = _create_program(
                    source_bytes,
                    encoded_expressions,
                    scope=scope,
                )
                err, retry_duration_ns = _compile_program(program, base_options)
                pch_session.phase_timings_ns["pch_fallback"] += retry_duration_ns
                used_pch = False
            if err != cuda_nvrtc.nvrtcResult.NVRTC_SUCCESS:
                compiler_log = get_program_log(program)
                if pch_fallback_log:
                    compiler_log = (
                        "PCH-enabled compilation failed:\n"
                        f"{pch_fallback_log}\n"
                        "Retry without PCH failed:\n"
                        f"{compiler_log}"
                    )
                raise DSLRuntimeError(
                    f"Failed compiling {scope} provider shim to LTO-IR.",
                    cause=RuntimeError(compiler_log),
                )
            if used_pch:
                pch_session.record_success(
                    cuda_nvrtc,
                    program,
                    compile_duration_ns=compile_duration_ns,
                    program_log=get_program_log(program),
                )
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
                    f"NVRTC produced an empty LTO-IR artifact for {scope} "
                    "provider shim."
                )
            artifact_blob = bytearray(blob_size)
            err = cuda_nvrtc.nvrtcGetLTOIR(program, artifact_blob)[0]
            if err != cuda_nvrtc.nvrtcResult.NVRTC_SUCCESS:
                raise DSLRuntimeError(
                    f"Failed retrieving NVRTC LTO-IR for {scope} provider shim."
                )
    except _provider_pch.PCHConfigurationError as exc:
        raise DSLRuntimeError(
            f"Invalid {scope} provider PCH configuration.",
            cause=exc,
        ) from exc
    finally:
        if pch_session is not None:
            for phase, duration_ns in pch_session.phase_timings_ns.items():
                phase_timings_ns[phase] = phase_timings_ns.get(phase, 0) + duration_ns
        if program is not None:
            cuda_nvrtc.nvrtcDestroyProgram(program)

    cached = _cache_support._CachedBundle(
        path=output_path,
        layouts_by_expression=layouts_by_expression,
        producer_compiler="nvrtc",
        producer_compiler_version=nvrtc_version,
        artifact_size=len(artifact_blob),
        artifact_sha256=hashlib.sha256(artifact_blob).hexdigest(),
    )
    _cache_support.write_binary_atomic(output_path, artifact_blob, scope=scope)
    _cache_support._write_bundle_metadata(
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
