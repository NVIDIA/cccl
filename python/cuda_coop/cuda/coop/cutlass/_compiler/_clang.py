# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


"""Host Clang discovery for the optional LLVM-bitcode route."""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from collections.abc import Callable

from cutlass.base_dsl.common import DSLRuntimeError

from ._bundle_contract import BundleCacheIdentity
from ._cache import (
    _CachedBundle,
    _write_bundle_metadata,
    write_binary_atomic,
    write_text_atomic,
)

CLANGXX_ENV = "CUDA_COOP_CUTLASS_PROVIDER_CLANGXX"

CLANGXX_TIMEOUT_SECONDS = 60


def resolve_clang_compiler(
    which: Callable[[str], str | None],
) -> tuple[str | None, str | None, str]:
    clangxx = os.environ.get(CLANGXX_ENV) or which("clang++")
    if not clangxx:
        return None, None, "clang-not-found"
    real_clangxx = os.path.realpath(clangxx)
    try:
        completed = subprocess.run(
            [real_clangxx, "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
        )
        version = completed.stdout.splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        version = None
    digest = hashlib.sha256()
    digest.update(real_clangxx.encode("utf-8", errors="surrogateescape"))
    digest.update(b"\0")
    digest.update((version or "unknown").encode("utf-8", errors="replace"))
    return real_clangxx, version, f"clang-{digest.hexdigest()[:16]}"


def compile_bundle(
    source: str,
    *,
    output_path: str,
    cache_dir: str,
    cache_identity: BundleCacheIdentity,
    compiler_options: tuple[str, ...],
    include_dirs: list[str],
    scope: str,
    clangxx: str | None,
    clang_version: str | None,
) -> _CachedBundle:
    """Compile one provider source to cached LLVM bitcode."""

    if not clangxx:
        raise DSLRuntimeError(
            f"Failed compiling {scope} provider shim to LLVM bitcode.",
            cause=RuntimeError("clang++ not found in PATH"),
        )
    try:
        with tempfile.TemporaryDirectory(
            dir=cache_dir,
            prefix=".bundle-",
        ) as temp_dir:
            source_path = os.path.join(temp_dir, "bundle.cpp")
            write_text_atomic(source_path, source, scope=scope)
            temporary_output_path = os.path.join(temp_dir, "bundle.bc")
            cmd = [
                clangxx,
                *compiler_options,
                *[f"-I{path}" for path in include_dirs],
                source_path,
                "-o",
                temporary_output_path,
            ]
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=CLANGXX_TIMEOUT_SECONDS,
            )
            with open(temporary_output_path, "rb") as artifact_file:
                artifact_blob = artifact_file.read()
    except subprocess.CalledProcessError as exc:
        raise DSLRuntimeError(
            f"Failed compiling {scope} provider shim to LLVM bitcode.",
            cause=RuntimeError(exc.stderr.strip()),
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise DSLRuntimeError(
            f"Timed out compiling {scope} provider shim to LLVM bitcode.",
            cause=exc,
        ) from exc
    except OSError as exc:
        raise DSLRuntimeError(
            f"Failed writing {scope} provider LLVM bitcode artifact.",
            cause=exc,
        ) from exc

    cached = _CachedBundle(
        path=output_path,
        layouts_by_expression={},
        producer_compiler="clang++",
        producer_compiler_version=clang_version,
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
