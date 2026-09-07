# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Process-private NVRTC automatic-PCH lifecycle management."""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import secrets
import shutil
import stat
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

PCH_ENV = "CUDA_COOP_CUTLASS_PROVIDER_PCH"
PCH_MODE_AUTO = "auto"
PCH_MODE_OFF = "off"
PCH_MINIMUM_VERSION = (12, 8)


class PCHConfigurationError(ValueError):
    """Raised for an unsupported process-private PCH configuration."""


class PCHUnavailableError(RuntimeError):
    """Raised when the process-private PCH storage cannot be used safely."""


@dataclass
class PCHSession:
    """One serialized automatic-PCH decision and its telemetry."""

    domain_key: str | None
    directory: str | None
    options: tuple[bytes, ...]
    state: str
    phase_timings_ns: dict[str, int] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return bool(self.options)

    def disable_after_failure(self, duration_ns: int) -> None:
        self._disable_domain()
        self.phase_timings_ns["pch_fallback"] = max(0, duration_ns)

    def _disable_domain(self) -> None:
        if self.domain_key is not None:
            with _PCH_STATE_LOCK:
                _PCH_DISABLED_DOMAINS.add(self.domain_key)

    def record_success(
        self,
        nvrtc: Any,
        program: Any,
        *,
        compile_duration_ns: int,
        program_log: str,
    ) -> None:
        if not self.enabled:
            return
        status_function = getattr(nvrtc, "nvrtcGetPCHCreateStatus", None)
        if not callable(status_function):
            return
        result = getattr(nvrtc, "nvrtcResult", None)
        success = getattr(result, "NVRTC_SUCCESS", object())
        try:
            status_result = status_function(program)
        except Exception:  # noqa: BLE001
            # PCH telemetry must never invalidate an otherwise valid artifact.
            return
        if isinstance(status_result, tuple):
            if not status_result:
                return
            # The current Python binding returns ``(pch_create_status,)``.
            # Accommodate wrappers that expose an API result followed by the
            # creation status without mistaking the former for the latter.
            if len(status_result) > 1:
                if status_result[0] != success:
                    return
                status = status_result[-1]
            else:
                status = status_result[0]
        else:
            status = status_result
        no_attempt = getattr(
            result,
            "NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED",
            object(),
        )
        heap_exhausted = getattr(
            result,
            "NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED",
            object(),
        )
        create_error = getattr(result, "NVRTC_ERROR_PCH_CREATE", object())
        if status == success:
            self.phase_timings_ns["pch_create"] = max(0, compile_duration_ns)
            return
        if status in (heap_exhausted, create_error):
            # PCH creation status does not invalidate a successful compilation.
            # In particular, do not resize the global heap: that would invalidate
            # every PCH file created with its previous base address.
            self._disable_domain()
            self.phase_timings_ns["pch_create"] = max(0, compile_duration_ns)
            self.phase_timings_ns["pch_create_warning"] = 0
            return
        if status == no_attempt and "using precompiled header" in program_log.lower():
            self.phase_timings_ns["pch_hit"] = max(0, compile_duration_ns)


_PCH_STATE_LOCK = threading.RLock()
_PCH_PID = os.getpid()
_PCH_POOL_PATH: str | None = None
_PCH_DISABLED_DOMAINS: set[str] = set()
_PCH_DOMAIN_LOCKS: dict[str, threading.RLock] = {}


def configured_pch_mode() -> str:
    mode = os.environ.get(PCH_ENV, PCH_MODE_AUTO).strip().lower()
    if mode not in (PCH_MODE_AUTO, PCH_MODE_OFF):
        raise PCHConfigurationError(
            f"Unsupported {PCH_ENV} value: {mode!r}. Use auto/off."
        )
    return mode


def _cleanup_pool() -> None:
    with _PCH_STATE_LOCK:
        if _PCH_POOL_PATH is None or _PCH_PID != os.getpid():
            return
        shutil.rmtree(_PCH_POOL_PATH, ignore_errors=True)


def _reset_after_fork() -> None:
    global _PCH_DISABLED_DOMAINS, _PCH_DOMAIN_LOCKS
    global _PCH_PID, _PCH_POOL_PATH, _PCH_STATE_LOCK
    _PCH_STATE_LOCK = threading.RLock()
    _PCH_PID = os.getpid()
    _PCH_POOL_PATH = None
    _PCH_DISABLED_DOMAINS = set()
    _PCH_DOMAIN_LOCKS = {}


def _ensure_process_state() -> None:
    if _PCH_PID != os.getpid():
        _reset_after_fork()


def _ensure_pool() -> str:
    global _PCH_POOL_PATH
    _ensure_process_state()
    try:
        if _PCH_POOL_PATH is None:
            nonce = secrets.token_hex(8)
            _PCH_POOL_PATH = tempfile.mkdtemp(
                prefix=f"cuda-coop-cutlass-pch-{os.getpid()}-{nonce}-"
            )
            os.chmod(_PCH_POOL_PATH, 0o700)
        pool_stat = os.lstat(_PCH_POOL_PATH)
    except OSError as exc:
        raise PCHUnavailableError(
            "Failed preparing the process-private NVRTC PCH pool."
        ) from exc
    if (
        not stat.S_ISDIR(pool_stat.st_mode)
        or stat.S_ISLNK(pool_stat.st_mode)
        or stat.S_IMODE(pool_stat.st_mode) != 0o700
    ):
        raise PCHUnavailableError(
            "The process-private NVRTC PCH pool is not a secure directory."
        )
    return _PCH_POOL_PATH


def make_pch_domain_key(
    *,
    nvrtc_version: tuple[int, int],
    bundle_arch: str,
    bundle_sm_arch: str,
    compiler_options: tuple[str, ...],
    include_dirs: tuple[str, ...],
    header_identity: str,
    preamble_identity: str,
) -> str:
    payload = {
        "nvrtc_version": list(nvrtc_version),
        "bundle_arch": bundle_arch,
        "bundle_sm_arch": bundle_sm_arch,
        "compiler_options": list(compiler_options),
        "include_dirs": list(include_dirs),
        "header_identity": header_identity,
        "preamble_identity": preamble_identity,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _starts_character_literal(line: str, index: int) -> bool:
    if index > 0 and index + 1 < len(line):
        previous = line[index - 1]
        following = line[index + 1]
        if (previous.isalnum() or previous == "_") and (
            following.isalnum() or following == "_"
        ):
            prefix_start = index - 1
            while prefix_start > 0 and (
                line[prefix_start - 1].isalnum() or line[prefix_start - 1] == "_"
            ):
                prefix_start -= 1
            if line[prefix_start:index] not in {"L", "U", "u", "u8"}:
                return False

    escaped = False
    for character in line[index + 1 :]:
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == "'":
            return True
    return False


def _block_comment_state(line: str, in_block_comment: bool) -> bool:
    """Return whether a C block comment remains open after one physical line."""

    quote: str | None = None
    escaped = False
    index = 0
    while index < len(line):
        if in_block_comment:
            comment_end = line.find("*/", index)
            if comment_end < 0:
                return True
            index = comment_end + 2
            in_block_comment = False
            continue

        character = line[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            index += 1
            continue
        if character == "'" and not _starts_character_literal(line, index):
            index += 1
            continue
        if character in {'"', "'"}:
            quote = character
            index += 1
            continue
        if line.startswith("//", index):
            return False
        if line.startswith("/*", index):
            in_block_comment = True
            index += 2
            continue
        index += 1
    return in_block_comment


def provider_preamble_identity(source: str) -> str:
    """Hash the reusable leading preprocessor sequence, excluding request bodies."""

    directives: list[str] = []
    in_block_comment = False
    in_continuation = False
    for line in source.splitlines():
        if in_continuation:
            directives.append(line)
            in_block_comment = _block_comment_state(line, in_block_comment)
            in_continuation = line.rstrip().endswith("\\")
            continue

        remaining = line.lstrip()
        while True:
            if in_block_comment:
                comment_end = remaining.find("*/")
                if comment_end < 0:
                    remaining = ""
                    break
                remaining = remaining[comment_end + 2 :].lstrip()
                in_block_comment = False
                continue
            if remaining.startswith("//") or not remaining:
                remaining = ""
                break
            if remaining.startswith("/*"):
                in_block_comment = True
                remaining = remaining[2:]
                continue
            break

        if not remaining:
            continue
        if not remaining.startswith("#"):
            break
        directives.append(remaining)
        in_block_comment = _block_comment_state(remaining, in_block_comment)
        in_continuation = remaining.rstrip().endswith("\\")

    payload = {
        "directives": directives,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _domain_directory(domain_key: str) -> str:
    directory = os.path.join(_ensure_pool(), f"domain-{domain_key}")
    try:
        try:
            os.mkdir(directory, mode=0o700)
        except FileExistsError:
            pass
        directory_stat = os.lstat(directory)
    except OSError as exc:
        raise PCHUnavailableError(
            "Failed preparing an NVRTC PCH compatibility domain."
        ) from exc
    if (
        not stat.S_ISDIR(directory_stat.st_mode)
        or stat.S_ISLNK(directory_stat.st_mode)
        or stat.S_IMODE(directory_stat.st_mode) != 0o700
    ):
        raise PCHUnavailableError(
            "The NVRTC PCH compatibility domain is not a secure directory."
        )
    return directory


@contextmanager
def pch_session(
    *,
    nvrtc_version: tuple[int, int] | None,
    bundle_arch: str,
    bundle_sm_arch: str,
    compiler_options: tuple[str, ...],
    include_dirs: tuple[str, ...],
    header_identity: str,
    preamble_identity: str,
):
    """Serialize and configure one process-private automatic-PCH attempt."""

    _ensure_process_state()
    lookup_started_ns = time.perf_counter_ns()
    mode = configured_pch_mode()
    if mode == PCH_MODE_OFF:
        session = PCHSession(
            domain_key=None,
            directory=None,
            options=(),
            state="off",
        )
        session.phase_timings_ns["pch_off"] = 0
        session.phase_timings_ns["pch_lookup"] = max(
            0,
            time.perf_counter_ns() - lookup_started_ns,
        )
        yield session
        return
    if nvrtc_version is None or nvrtc_version < PCH_MINIMUM_VERSION:
        session = PCHSession(
            domain_key=None,
            directory=None,
            options=(),
            state="unsupported",
        )
        session.phase_timings_ns["pch_unsupported"] = 0
        session.phase_timings_ns["pch_lookup"] = max(
            0,
            time.perf_counter_ns() - lookup_started_ns,
        )
        yield session
        return

    domain_key = make_pch_domain_key(
        nvrtc_version=nvrtc_version,
        bundle_arch=bundle_arch,
        bundle_sm_arch=bundle_sm_arch,
        compiler_options=compiler_options,
        include_dirs=include_dirs,
        header_identity=header_identity,
        preamble_identity=preamble_identity,
    )
    with _PCH_STATE_LOCK:
        domain_lock = _PCH_DOMAIN_LOCKS.setdefault(
            domain_key,
            threading.RLock(),
        )

    domain_lock.acquire()
    hold_domain_lock = True
    try:
        with _PCH_STATE_LOCK:
            if domain_key in _PCH_DISABLED_DOMAINS:
                session = PCHSession(
                    domain_key=domain_key,
                    directory=None,
                    options=(),
                    state="disabled",
                )
                session.phase_timings_ns["pch_disabled"] = 0
                domain_lock.release()
                hold_domain_lock = False
            else:
                try:
                    directory = _domain_directory(domain_key)
                except PCHUnavailableError:
                    _PCH_DISABLED_DOMAINS.add(domain_key)
                    session = PCHSession(
                        domain_key=domain_key,
                        directory=None,
                        options=(),
                        state="unavailable",
                    )
                    session.phase_timings_ns["pch_unavailable"] = 0
                    domain_lock.release()
                    hold_domain_lock = False
                else:
                    session = PCHSession(
                        domain_key=domain_key,
                        directory=directory,
                        options=(
                            b"--pch",
                            f"--pch-dir={directory}".encode(),
                            b"--pch-messages=true",
                        ),
                        state="enabled",
                    )
        session.phase_timings_ns["pch_lookup"] = max(
            0,
            time.perf_counter_ns() - lookup_started_ns,
        )
        yield session
    finally:
        if hold_domain_lock:
            domain_lock.release()


atexit.register(_cleanup_pool)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


__all__ = [
    "PCH_ENV",
    "PCHConfigurationError",
    "PCHSession",
    "configured_pch_mode",
    "make_pch_domain_key",
    "pch_session",
    "provider_preamble_identity",
]
