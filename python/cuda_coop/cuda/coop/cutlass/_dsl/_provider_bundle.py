# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Provider bundle compilation and link-attribute helpers for CuTe coop shims."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Hashable, Iterable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cutlass._mlir import ir
from cutlass.base_dsl.common import DSLRuntimeError
from cutlass.base_dsl.compiler import GPUArch

import cuda.bindings.nvrtc as cuda_nvrtc
from cuda.coop._headers import HeaderResolutionError, resolve_include_paths
from cuda.coop._headers._identity import (
    HeaderIdentityError,
    IncludeDirsIdentity,
)
from cuda.coop._headers._identity import (
    include_dirs_identity as resolve_include_dirs_identity,
)

from . import _provider_pch

if os.name == "nt":
    import msvcrt
else:
    import fcntl

DUMP_DIR_ENV = "CUDA_COOP_CUTLASS_PROVIDER_DUMP_DIR"
BUNDLE_FORMAT_ENV = "CUDA_COOP_CUTLASS_PROVIDER_BUNDLE_FORMAT"
CACHE_DIR_ENV = "CUDA_COOP_CUTLASS_PROVIDER_CACHE_DIR"
CCCL_ROOT_ENV = "CUDA_COOP_CUTLASS_PROVIDER_CCCL_ROOT"
COMMON_CCCL_ROOT_ENV = "CUDA_COOP_CCCL_ROOT"
CLANGXX_ENV = "CUDA_COOP_CUTLASS_PROVIDER_CLANGXX"
EXTERNAL_LINK_FILES_ATTR = "external-link-files"
LINK_LIBRARIES_ATTR = "link-libraries"
NVVM_VERSION_OPT = b"-nvvm-version=nvvm-latest"
CLANGXX_TIMEOUT_SECONDS = 60
BUNDLE_IDENTITY_VERSION = 1
BUNDLE_CACHE_SCHEMA_VERSION = 3
PROVIDER_BUNDLE_ABI_VERSION = 1
BUNDLE_METADATA_VERSION = 3
LAYOUT_METADATA_VERSION = BUNDLE_METADATA_VERSION
RESOLUTION_ROUTE_PRECOMPILED = "precompiled"
RESOLUTION_ROUTE_MEMORY = "memory"
RESOLUTION_ROUTE_DISK = "disk"
RESOLUTION_ROUTE_CLANG = "clang"
RESOLUTION_ROUTE_NVRTC = "nvrtc"


@dataclass(frozen=True)
class LayoutProbe:
    """C++ constant expressions whose values describe one storage layout."""

    key: Hashable
    size_expression: str
    alignment_expression: str


@dataclass(frozen=True)
class StorageLayout:
    size_in_bytes: int
    alignment: int


@dataclass(frozen=True)
class BundleCompilation:
    path: str
    layouts: dict[Hashable, StorageLayout]


@dataclass(frozen=True)
class BundleIdentity:
    """Portable identity for one provider source and compilation contract."""

    version: int
    provider_abi_version: int
    source_hash: str
    bundle_format: str
    bundle_arch: str
    bundle_sm_arch: str
    compiler_options: tuple[str, ...]
    layout_expressions: tuple[str, ...]


@dataclass(frozen=True)
class BundleCacheIdentity:
    """Mutable-header identity used only by the existing JIT cache."""

    schema_version: int
    bundle: BundleIdentity
    include_key: str
    producer_compiler_version: str

    @property
    def contract_digest(self) -> str:
        payload = {
            "bundle": asdict(self.bundle),
            "include_key": self.include_key,
            "producer_compiler_version": self.producer_compiler_version,
            "schema_version": self.schema_version,
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    @property
    def cache_key(self) -> str:
        return f"v{self.schema_version}:{self.contract_digest}"

    @property
    def artifact_stem(self) -> str:
        return f"bundle_v{self.schema_version}_{self.contract_digest}"


@dataclass(frozen=True)
class BundleResolutionRequest:
    """Canonical source and layout metadata presented to bundle resolvers."""

    identity: BundleIdentity
    source: str
    symbols: tuple[str, ...]


@dataclass(frozen=True)
class BundleResolution:
    """One resolved provider artifact and its exact layout metadata."""

    request: BundleResolutionRequest
    path: str
    layouts_by_expression: Mapping[str, StorageLayout]
    route: str
    producer_compiler: str | None
    producer_compiler_version: str | None
    producer_toolkit_version: str | None
    phase_timings_ns: Mapping[str, int]


@dataclass(frozen=True)
class BundleTelemetry:
    """Internal snapshot of provider resolution counts and phase timings."""

    route_counts: Mapping[str, int]
    phase_counts: Mapping[str, int]
    phase_timings_ns: Mapping[str, int]


@dataclass(frozen=True)
class _BundlePrecompileResolver:
    """One typed resolver registration and its exact resolution contract."""

    callback: Callable[[BundleResolutionRequest], BundleResolution | None]
    route: str
    phase: str


@dataclass(frozen=True)
class _PreparedLayoutProbes:
    source: str
    expressions: tuple[str, ...]
    key_to_expression: dict[Hashable, str]
    symbol: str


@dataclass(frozen=True)
class _CachedBundle:
    path: str
    layouts_by_expression: dict[str, StorageLayout]
    producer_compiler: str | None = None
    producer_compiler_version: str | None = None
    producer_toolkit_version: str | None = None


_BUNDLE_COMPILE_COUNTER = 0
_NVRTC_COMPILE_PROGRAM_COUNTER = 0
_SOURCE_CACHE: dict[str, _CachedBundle] = {}
_MANAGED_BUNDLE_PATHS: set[str] = set()
_ROUTE_COUNTS: dict[str, int] = {}
_PHASE_COUNTS: dict[str, int] = {}
_PHASE_TIMINGS_NS: dict[str, int] = {}
_STATE_LOCK = threading.RLock()
_ARTIFACT_LOCKS: dict[str, threading.RLock] = {}
_ACTIVE_ARTIFACT_LOCK_FDS: set[int] = set()
_UNKNOWN_COMPILER_PROCESS_TOKEN = f"unknown-{os.getpid()}-{os.urandom(8).hex()}"
_ACTIVE_PRECOMPILE_RESOLVERS: ContextVar[tuple[_BundlePrecompileResolver, ...]] = (
    ContextVar(
        "cuda_coop_cutlass_active_precompile_resolvers",
        default=(),
    )
)
_ACTIVE_POST_RESOLUTION_OBSERVERS: ContextVar[
    tuple[Callable[[BundleResolution], None], ...]
] = ContextVar(
    "cuda_coop_cutlass_active_post_resolution_observers",
    default=(),
)


def _acquire_state_lock_before_fork() -> None:
    _STATE_LOCK.acquire()


def _release_state_lock_after_fork() -> None:
    _STATE_LOCK.release()


def _reset_locks_after_fork() -> None:
    global _ACTIVE_ARTIFACT_LOCK_FDS, _ARTIFACT_LOCKS, _STATE_LOCK
    for descriptor in _ACTIVE_ARTIFACT_LOCK_FDS:
        try:
            os.close(descriptor)
        except OSError:
            pass
    _ACTIVE_ARTIFACT_LOCK_FDS = set()
    _STATE_LOCK = threading.RLock()
    _ARTIFACT_LOCKS = {}


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_acquire_state_lock_before_fork,
        after_in_parent=_release_state_lock_after_fork,
        after_in_child=_reset_locks_after_fork,
    )


def _cache_dir_name() -> str:
    getuid = getattr(os, "getuid", None)
    if getuid is None:
        return "cuda_coop_cutlass_provider"
    return f"cuda_coop_cutlass_provider_{getuid()}"


_CACHE_DIR = os.path.join(tempfile.gettempdir(), _cache_dir_name())


def _bundle_precompile_resolver(
    callback: Callable[[BundleResolutionRequest], BundleResolution | None],
    *,
    route: str = RESOLUTION_ROUTE_PRECOMPILED,
    phase: str = "precompile_resolvers",
) -> _BundlePrecompileResolver:
    if not callable(callback):
        raise TypeError("bundle precompile resolver must be callable")
    if route not in {RESOLUTION_ROUTE_PRECOMPILED}:
        raise ValueError(f"unsupported bundle precompile resolver route {route!r}")
    if not isinstance(phase, str) or not phase:
        raise ValueError("bundle precompile resolver phase must be a nonempty string")
    return _BundlePrecompileResolver(
        callback=callback,
        route=route,
        phase=phase,
    )


@contextmanager
def activate_bundle_precompile_resolver(
    resolver: (
        _BundlePrecompileResolver
        | Callable[[BundleResolutionRequest], BundleResolution | None]
    ),
):
    """Push one context-local resolver, tried before enclosing resolvers."""

    if isinstance(resolver, _BundlePrecompileResolver):
        resolver = _bundle_precompile_resolver(
            resolver.callback,
            route=resolver.route,
            phase=resolver.phase,
        )
    else:
        resolver = _bundle_precompile_resolver(resolver)
    resolvers = _ACTIVE_PRECOMPILE_RESOLVERS.get()
    token = _ACTIVE_PRECOMPILE_RESOLVERS.set((*resolvers, resolver))
    try:
        yield
    finally:
        _ACTIVE_PRECOMPILE_RESOLVERS.reset(token)


@contextmanager
def activate_bundle_resolution_observer(
    observer: Callable[[BundleResolution], None],
):
    """Push one context-local observer, invoked after enclosing observers."""

    if not callable(observer):
        raise TypeError("bundle resolution observer must be callable")
    observers = _ACTIVE_POST_RESOLUTION_OBSERVERS.get()
    token = _ACTIVE_POST_RESOLUTION_OBSERVERS.set((*observers, observer))
    try:
        yield
    finally:
        _ACTIVE_POST_RESOLUTION_OBSERVERS.reset(token)


def configured_cache_dir() -> str:
    cache_dir = os.environ.get(CACHE_DIR_ENV)
    if cache_dir:
        return os.path.abspath(os.path.expanduser(cache_dir))
    return _CACHE_DIR


def ensure_cache_dir(scope: str) -> str:
    cache_dir = configured_cache_dir()
    try:
        try:
            os.mkdir(cache_dir, mode=0o700)
        except FileExistsError:
            pass
        cache_stat = os.lstat(cache_dir)
        if stat.S_ISLNK(cache_stat.st_mode) or not stat.S_ISDIR(cache_stat.st_mode):
            raise DSLRuntimeError(
                f"{scope} provider cache path is not a real directory."
            )
        getuid = getattr(os, "getuid", None)
        if getuid is not None and cache_stat.st_uid != getuid():
            raise DSLRuntimeError(
                f"{scope} provider cache directory is not owned by this user."
            )
        if stat.S_IMODE(cache_stat.st_mode) != 0o700:
            os.chmod(cache_dir, 0o700)
    except OSError as exc:
        raise DSLRuntimeError(
            f"Failed preparing {scope} provider cache directory.",
            cause=exc,
        ) from exc
    return cache_dir


def write_binary_atomic(path: str, blob: bytes | bytearray, *, scope: str) -> None:
    cache_dir = ensure_cache_dir(scope)
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=cache_dir,
            prefix=".bundle-",
            delete=False,
        ) as f:
            temp_path = f.name
            f.write(blob)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    except OSError as exc:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        raise DSLRuntimeError(
            f"Failed writing {scope} provider cache artifact.",
            cause=exc,
        ) from exc


def write_text_atomic(path: str, text: str, *, scope: str) -> None:
    write_binary_atomic(path, text.encode("utf-8"), scope=scope)


def _local_artifact_lock(path: str) -> threading.RLock:
    real_path = os.path.realpath(path)
    with _STATE_LOCK:
        return _ARTIFACT_LOCKS.setdefault(real_path, threading.RLock())


def _close_artifact_lock_descriptor(descriptor: int) -> None:
    with _STATE_LOCK:
        try:
            os.close(descriptor)
        except OSError:
            pass
        finally:
            _ACTIVE_ARTIFACT_LOCK_FDS.discard(descriptor)


@contextmanager
def artifact_lock(path: str, *, scope: str):
    """Serialize one cache artifact across threads and processes."""

    lock_path = f"{path}.lock"
    local_lock = _local_artifact_lock(path)
    with local_lock:
        flags = os.O_CREAT | os.O_RDWR
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor_owner_pid = os.getpid()
        descriptor: int | None = None
        try:
            with _STATE_LOCK:
                descriptor = os.open(lock_path, flags, 0o600)
                _ACTIVE_ARTIFACT_LOCK_FDS.add(descriptor)
            lock_stat = os.fstat(descriptor)
            if not stat.S_ISREG(lock_stat.st_mode):
                raise OSError("provider artifact lock is not a regular file")
            getuid = getattr(os, "getuid", None)
            if getuid is not None and lock_stat.st_uid != getuid():
                raise OSError("provider artifact lock is not owned by this user")
            if os.name == "nt":
                if lock_stat.st_size == 0:
                    os.write(descriptor, b"\0")
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
            else:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
        except OSError as exc:
            if descriptor is not None and os.getpid() == descriptor_owner_pid:
                _close_artifact_lock_descriptor(descriptor)
            raise DSLRuntimeError(
                f"Failed locking {scope} provider cache artifact.",
                cause=exc,
            ) from exc
        assert descriptor is not None
        try:
            yield
        finally:
            if os.getpid() == descriptor_owner_pid:
                try:
                    if os.name == "nt":
                        os.lseek(descriptor, 0, os.SEEK_SET)
                        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                    else:
                        fcntl.flock(descriptor, fcntl.LOCK_UN)
                except OSError:
                    pass
                _close_artifact_lock_descriptor(descriptor)


def append_external_link_file_attr(module: ir.Module, path: str) -> None:
    for op in module.body.operations:
        if op.name != "gpu.module":
            continue
        existing: set[str] = set()
        if EXTERNAL_LINK_FILES_ATTR in op.attributes:
            existing.update(
                attr.value
                for attr in op.attributes[EXTERNAL_LINK_FILES_ATTR]
                if getattr(attr, "value", "")
            )
        existing.add(path)
        op.attributes[EXTERNAL_LINK_FILES_ATTR] = ir.ArrayAttr.get(
            [ir.StringAttr.get(x) for x in sorted(existing)]
        )


def append_link_library_attr(module: ir.Module, path: str) -> None:
    for op in module.body.operations:
        if op.name != "gpu.module":
            continue
        existing: set[str] = set()
        if LINK_LIBRARIES_ATTR in op.attributes:
            existing.update(
                attr.value
                for attr in op.attributes[LINK_LIBRARIES_ATTR]
                if getattr(attr, "value", "")
            )
        existing.add(path)
        op.attributes[LINK_LIBRARIES_ATTR] = ir.ArrayAttr.get(
            [ir.StringAttr.get(x) for x in sorted(existing)]
        )


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
    arch = str(getattr(option, "value", "")).strip()
    if arch:
        return arch
    return str(getattr(getattr(dsl, "envar", None), "arch", "")).strip()


def _strip_arch_suffix_for_compute(arch: str) -> str:
    return arch


def _is_numeric_arch(arch: str) -> bool:
    if arch.isdigit():
        return True
    return bool(arch and arch[-1] in ("a", "f") and arch[:-1].isdigit())


def resolve_nvrtc_arch(
    scope: str,
    configured_gpu_arch: Callable[[], str],
) -> str:
    """Returns a compute_* arch for NVRTC LTO-IR compilation."""
    arch = configured_gpu_arch()
    if arch:
        if arch.startswith("compute_"):
            return f"compute_{_strip_arch_suffix_for_compute(arch[8:])}"
        if arch.startswith("compute"):
            return f"compute_{_strip_arch_suffix_for_compute(arch[7:])}"

        sm_arch = arch
        if sm_arch.startswith("sm_"):
            sm_arch = sm_arch[3:]
        elif sm_arch.startswith("sm"):
            sm_arch = sm_arch[2:]
        sm_arch = _strip_arch_suffix_for_compute(sm_arch)
        if _is_numeric_arch(sm_arch):
            return f"compute_{sm_arch}"

    from cutlass.base_dsl.runtime import cuda as cuda_runtime

    major, minor = cuda_runtime.get_compute_capability_major_minor()
    if major is None or minor is None:
        raise DSLRuntimeError(
            f"Unable to resolve CUDA arch for {scope} provider NVRTC bundle compilation."
        )
    return f"compute_{major}{minor}"


def resolve_nvrtc_sm_arch(
    scope: str,
    configured_gpu_arch: Callable[[], str],
) -> str:
    """Returns an sm_* arch for the external link target."""
    arch = configured_gpu_arch()
    if arch:
        if arch.startswith("sm_"):
            return arch
        if arch.startswith("sm"):
            return f"sm_{arch[2:]}"
        if arch.startswith("compute_"):
            return f"sm_{arch[8:]}"
        if arch.startswith("compute"):
            return f"sm_{arch[7:]}"
        if _is_numeric_arch(arch):
            return f"sm_{arch}"

    from cutlass.base_dsl.runtime import cuda as cuda_runtime

    major, minor = cuda_runtime.get_compute_capability_major_minor()
    if major is None or minor is None:
        raise DSLRuntimeError(
            f"Unable to resolve CUDA arch for {scope} provider NVRTC bundle compilation."
        )
    return f"sm_{major}{minor}"


def select_bundle_format(scope: str) -> str:
    """
    Returns one of: "bc" or "ltoir".

    The default `auto` path chooses NVRTC-backed LTO-IR. Users can select
    LLVM bitcode or LTO-IR explicitly through
    CUDA_COOP_CUTLASS_PROVIDER_BUNDLE_FORMAT.
    """
    fmt = os.environ.get(BUNDLE_FORMAT_ENV, "auto").strip().lower()
    if fmt == "bc":
        return "bc"
    if fmt in ("lto", "ltoir"):
        return "ltoir"
    if fmt == "cubin":
        raise DSLRuntimeError(
            f"Unsupported {BUNDLE_FORMAT_ENV} value: {fmt!r}. "
            f"{scope} external provider shims require LTO-IR or LLVM bitcode."
        )
    if fmt != "auto":
        raise DSLRuntimeError(
            f"Unsupported {BUNDLE_FORMAT_ENV} value: {fmt!r}. Use auto/bc/ltoir."
        )
    return "ltoir"


def get_nvrtc_program_log(prog: Any) -> str:
    get_log_size = getattr(cuda_nvrtc, "nvrtcGetProgramLogSize", None)
    get_log = getattr(cuda_nvrtc, "nvrtcGetProgramLog", None)
    if not callable(get_log_size) or not callable(get_log):
        return ""
    err, log_size = get_log_size(prog)
    if err != cuda_nvrtc.nvrtcResult.NVRTC_SUCCESS or log_size <= 0:
        return ""
    log = bytearray(log_size)
    err = get_log(prog, log)[0]
    if err != cuda_nvrtc.nvrtcResult.NVRTC_SUCCESS:
        return ""
    return bytes(log).decode("utf-8", errors="replace").strip("\x00").strip()


def get_nvrtc_version_tuple() -> tuple[int, int] | None:
    version = getattr(cuda_nvrtc, "nvrtcVersion", None)
    if not callable(version):
        return None
    try:
        err, major, minor = version()
        parsed_version = int(major), int(minor)
    except (TypeError, ValueError):
        return None
    if err != cuda_nvrtc.nvrtcResult.NVRTC_SUCCESS:
        return None
    return parsed_version


def get_nvrtc_version() -> str | None:
    version = get_nvrtc_version_tuple()
    if version is None:
        return None
    return f"{version[0]}.{version[1]}"


_PRELOADED_NVRTC_LIB_DIRS: set[str] = set()


def preload_toolkit_nvrtc(include_dirs: Iterable[str]) -> None:
    """Load the resolved CUDA toolkit's NVRTC before the first NVRTC call.

    ``cuda.bindings`` adopts whichever NVRTC library is already loaded in the
    process. A framework built for another CUDA major (for example a cu12
    Torch wheel) can load its own NVRTC first, which then mismatches the CUDA
    headers these bundles compile against. Loading the toolkit's own NVRTC
    eagerly keeps the binding consistent with the resolved include roots.
    """

    import ctypes
    import glob as _glob

    lib_dirs = [
        os.path.join(os.path.dirname(directory), name)
        for directory in include_dirs
        for name in ("lib", "lib64")
    ]
    for lib_dir in lib_dirs:
        if lib_dir in _PRELOADED_NVRTC_LIB_DIRS:
            continue
        candidates = [
            candidate
            for pattern in ("libnvrtc.so.*", "libnvJitLink.so.*")
            for candidate in sorted(_glob.glob(os.path.join(lib_dir, pattern)))
            if re.fullmatch(r"libnv[A-Za-z]+\.so\.\d+", os.path.basename(candidate))
        ]
        loaded = False
        for candidate in candidates:
            try:
                ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue
            loaded = True
        if loaded:
            _PRELOADED_NVRTC_LIB_DIRS.add(lib_dir)
            return


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


def maybe_dump_source(source: str, source_hash: str) -> None:
    dump_dir = os.environ.get(DUMP_DIR_ENV)
    if not dump_dir:
        return
    dump_dir = os.path.abspath(os.path.expanduser(dump_dir))
    os.makedirs(dump_dir, exist_ok=True)
    path = os.path.join(dump_dir, f"cuda_coop_cutlass_bundle_{source_hash}.cpp")
    with open(path, "w", encoding="utf-8") as f:
        f.write(source)


def required_cccl_headers(
    source: str,
    *,
    registered_headers: Callable[[], dict[str, str]],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                relative_path
                for include, relative_path in registered_headers().items()
                if include in source
            }
        )
    )


def cccl_include_dirs(
    required_headers: tuple[str, ...],
    *,
    scope: str,
    provider_dir: str,
) -> list[str]:
    if not required_headers:
        return []

    try:
        paths = resolve_include_paths(
            start=Path(provider_dir),
            configured_roots=(
                os.environ.get(CCCL_ROOT_ENV),
                os.environ.get(COMMON_CCCL_ROOT_ENV),
            ),
            required_headers=required_headers,
        )
    except HeaderResolutionError as exc:
        raise DSLRuntimeError(
            f"{scope} provider could not resolve its CCCL headers.",
            cause=exc,
        ) from exc
    return [str(path) for path in paths.as_tuple()]


def cccl_include_options(include_dirs: list[str]) -> list[bytes]:
    return [f"-I{path}".encode("utf-8") for path in include_dirs]


def include_dirs_identity(include_dirs: list[str]) -> IncludeDirsIdentity:
    try:
        return resolve_include_dirs_identity(include_dirs)
    except HeaderIdentityError as exc:
        raise DSLRuntimeError(
            "Failed fingerprinting provider include paths.",
            cause=exc,
        ) from exc


def include_dirs_cache_key(include_dirs: list[str]) -> str:
    return include_dirs_identity(include_dirs).digest


def bundle_compiler_options(
    bundle_format: str,
    bundle_arch: str,
) -> tuple[str, ...]:
    if bundle_format == "bc":
        return (
            "--target=nvptx64-nvidia-cuda",
            "-std=c++17",
            "-O3",
            "-emit-llvm",
            "-c",
        )
    return (
        "--std=c++17",
        "--relocatable-device-code=true",
        "-default-device",
        NVVM_VERSION_OPT.decode("ascii"),
        f"--gpu-architecture={bundle_arch}",
        "-dlto",
        # Work around an NVRTC 13.3 diagnostic pragma bug in CUDA Toolkit
        # headers that leaves deprecated vector-type markers unexpanded when
        # headers such as cuda_fp8.h are reached without vector_types.h.
        "-D__NV_NO_VECTOR_DEPRECATION_DIAG",
    )


def make_bundle_identity(
    *,
    source_hash: str,
    bundle_format: str,
    bundle_arch: str,
    bundle_sm_arch: str,
    compiler_options: tuple[str, ...],
    layout_expressions: tuple[str, ...],
) -> BundleIdentity:
    return BundleIdentity(
        version=BUNDLE_IDENTITY_VERSION,
        provider_abi_version=PROVIDER_BUNDLE_ABI_VERSION,
        source_hash=source_hash,
        bundle_format=bundle_format,
        bundle_arch=bundle_arch,
        bundle_sm_arch=bundle_sm_arch,
        compiler_options=compiler_options,
        layout_expressions=layout_expressions,
    )


def make_bundle_cache_identity(
    bundle: BundleIdentity,
    *,
    include_key: str,
    producer_compiler_version: str,
) -> BundleCacheIdentity:
    return BundleCacheIdentity(
        schema_version=BUNDLE_CACHE_SCHEMA_VERSION,
        bundle=bundle,
        include_key=include_key,
        producer_compiler_version=producer_compiler_version,
    )


def _prepare_layout_probes(
    source: str,
    layout_probes: Iterable[LayoutProbe],
) -> _PreparedLayoutProbes:
    probes_by_key: dict[Hashable, tuple[str, str]] = {}
    for probe in layout_probes:
        if not isinstance(probe, LayoutProbe):
            raise TypeError("layout_probes must contain LayoutProbe values")
        try:
            hash(probe.key)
        except TypeError as exc:
            raise TypeError("layout probe keys must be hashable") from exc
        size_expression = probe.size_expression.strip()
        alignment_expression = probe.alignment_expression.strip()
        if not size_expression or not alignment_expression:
            raise ValueError("layout probe expressions must be non-empty")
        expressions = (size_expression, alignment_expression)
        existing = probes_by_key.get(probe.key)
        if existing is not None and existing != expressions:
            raise ValueError(f"conflicting layout probes for key {probe.key!r}")
        probes_by_key[probe.key] = expressions

    if not probes_by_key:
        return _PreparedLayoutProbes(
            source=source,
            expressions=(),
            key_to_expression={},
            symbol="",
        )

    unique_probes = sorted(set(probes_by_key.values()))
    probe_digest = hashlib.sha256(
        json.dumps(
            {
                "version": LAYOUT_METADATA_VERSION,
                "probes": unique_probes,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    symbol = f"cuda_coop_layout_probe_{probe_digest}"
    source = (
        f"{source.rstrip()}\n\n"
        "template <unsigned long long Size, unsigned long long Alignment>\n"
        f"__device__ unsigned char {symbol} = 0;\n"
    )
    expression_by_probe = {
        probe: f"&{symbol}<({probe[0]}), ({probe[1]})>" for probe in unique_probes
    }
    key_to_expression = {
        key: expression_by_probe[probe] for key, probe in probes_by_key.items()
    }
    return _PreparedLayoutProbes(
        source=source,
        expressions=tuple(sorted(expression_by_probe.values())),
        key_to_expression=key_to_expression,
        symbol=symbol,
    )


def _validate_storage_layout(
    size_in_bytes: Any,
    alignment: Any,
    *,
    description: str,
) -> StorageLayout:
    if (
        not isinstance(size_in_bytes, int)
        or isinstance(size_in_bytes, bool)
        or not isinstance(alignment, int)
        or isinstance(alignment, bool)
        or size_in_bytes <= 0
        or alignment <= 0
        or alignment & (alignment - 1)
        or size_in_bytes % alignment != 0
    ):
        raise ValueError(
            f"Invalid storage layout for {description}: "
            f"size={size_in_bytes!r}, alignment={alignment!r}."
        )
    return StorageLayout(size_in_bytes=size_in_bytes, alignment=alignment)


def _decode_layout_probe_name(
    lowered_name: bytes | str,
    *,
    symbol: str,
    expression: str,
) -> StorageLayout:
    if isinstance(lowered_name, bytes):
        lowered_name = lowered_name.decode("utf-8", errors="strict")
    lowered_name = lowered_name.rstrip("\0")
    match = re.fullmatch(
        rf"_Z{len(symbol)}{re.escape(symbol)}ILy([0-9]+)ELy([0-9]+)EE",
        lowered_name,
    )
    if match is None:
        raise ValueError(
            "NVRTC returned an unexpected lowered layout-probe name for "
            f"{expression!r}: {lowered_name!r}."
        )
    return _validate_storage_layout(
        int(match.group(1)),
        int(match.group(2)),
        description=expression,
    )


def _layout_metadata_path(output_path: str) -> str:
    return f"{output_path}.layouts.json"


def _optional_metadata_string(value: Any) -> str | None:
    if value is None or isinstance(value, str):
        return value
    raise ValueError("invalid provider producer metadata")


def _load_bundle_metadata(
    output_path: str,
    expressions: tuple[str, ...],
    cache_identity: BundleCacheIdentity,
) -> _CachedBundle | None:
    metadata_path = _layout_metadata_path(output_path)
    try:
        metadata_stat = os.lstat(metadata_path)
        if not stat.S_ISREG(metadata_stat.st_mode):
            return None
        with open(metadata_path, encoding="utf-8") as metadata_file:
            payload = json.load(metadata_file)
        if (
            not isinstance(payload, dict)
            or payload.get("version") != BUNDLE_METADATA_VERSION
            or payload.get("cache_key") != cache_identity.cache_key
            or payload.get("contract_digest") != cache_identity.contract_digest
        ):
            return None

        artifact = payload.get("artifact")
        if not isinstance(artifact, dict):
            return None
        artifact_size = artifact.get("size")
        artifact_sha256 = artifact.get("sha256")
        if (
            not isinstance(artifact_size, int)
            or isinstance(artifact_size, bool)
            or artifact_size < 0
            or not isinstance(artifact_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", artifact_sha256) is None
        ):
            return None
        output_stat = os.lstat(output_path)
        if (
            not stat.S_ISREG(output_stat.st_mode)
            or output_stat.st_size != artifact_size
        ):
            return None
        artifact_digest = hashlib.sha256()
        with open(output_path, "rb") as artifact_file:
            while chunk := artifact_file.read(1024 * 1024):
                artifact_digest.update(chunk)
        if artifact_digest.hexdigest() != artifact_sha256:
            return None

        serialized_layouts = payload.get("layouts")
        if not isinstance(serialized_layouts, dict) or set(serialized_layouts) != set(
            expressions
        ):
            return None
        layouts = {}
        for expression in expressions:
            serialized_layout = serialized_layouts[expression]
            if not isinstance(serialized_layout, list) or len(serialized_layout) != 2:
                return None
            layouts[expression] = _validate_storage_layout(
                serialized_layout[0],
                serialized_layout[1],
                description=expression,
            )

        producer = payload.get("producer")
        if not isinstance(producer, dict):
            return None
        return _CachedBundle(
            path=output_path,
            layouts_by_expression=layouts,
            producer_compiler=_optional_metadata_string(producer.get("compiler")),
            producer_compiler_version=_optional_metadata_string(
                producer.get("compiler_version")
            ),
            producer_toolkit_version=_optional_metadata_string(
                producer.get("toolkit_version")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return None


def _revalidate_memory_cached_bundle(
    cache_identity: BundleCacheIdentity,
    expressions: tuple[str, ...],
) -> _CachedBundle | None:
    with _STATE_LOCK:
        cached = _SOURCE_CACHE.get(cache_identity.cache_key)
    if cached is None:
        return None

    validated = _load_bundle_metadata(
        cached.path,
        expressions,
        cache_identity,
    )
    if validated == cached:
        return validated

    with _STATE_LOCK:
        if _SOURCE_CACHE.get(cache_identity.cache_key) is cached:
            _SOURCE_CACHE.pop(cache_identity.cache_key, None)
    return None


def _write_bundle_metadata(
    output_path: str,
    artifact_blob: bytes | bytearray,
    cached: _CachedBundle,
    cache_identity: BundleCacheIdentity,
    *,
    scope: str,
) -> None:
    payload = {
        "version": BUNDLE_METADATA_VERSION,
        "cache_key": cache_identity.cache_key,
        "contract_digest": cache_identity.contract_digest,
        "artifact": {
            "size": len(artifact_blob),
            "sha256": hashlib.sha256(artifact_blob).hexdigest(),
        },
        "producer": {
            "compiler": cached.producer_compiler,
            "compiler_version": cached.producer_compiler_version,
            "toolkit_version": cached.producer_toolkit_version,
        },
        "layouts": {
            expression: [layout.size_in_bytes, layout.alignment]
            for expression, layout in cached.layouts_by_expression.items()
        },
    }
    write_text_atomic(
        _layout_metadata_path(output_path),
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        scope=scope,
    )


def _bundle_compilation(
    cached: _CachedBundle,
    key_to_expression: dict[Hashable, str],
) -> BundleCompilation:
    with _STATE_LOCK:
        _MANAGED_BUNDLE_PATHS.add(os.path.realpath(cached.path))
    return BundleCompilation(
        path=cached.path,
        layouts={
            key: cached.layouts_by_expression[expression]
            for key, expression in key_to_expression.items()
        },
    )


def _finish_phase(
    phase_timings_ns: dict[str, int],
    phase: str,
    started_ns: int,
) -> None:
    elapsed_ns = max(0, time.perf_counter_ns() - started_ns)
    _add_phase_duration(phase_timings_ns, phase, elapsed_ns)


def _add_phase_duration(
    phase_timings_ns: dict[str, int],
    phase: str,
    duration_ns: int,
) -> None:
    phase_timings_ns[phase] = phase_timings_ns.get(phase, 0) + max(0, duration_ns)


def _validated_precompiled_bundle(
    resolution: BundleResolution,
    request: BundleResolutionRequest,
    *,
    route: str,
) -> _CachedBundle:
    if not isinstance(resolution, BundleResolution):
        raise TypeError(
            "bundle precompile resolver must return BundleResolution or None"
        )
    if resolution.request != request:
        raise ValueError("precompiled bundle resolution does not match its request")
    if resolution.route != route:
        raise ValueError("precompiled bundle resolution has an invalid route")
    if not isinstance(resolution.path, str) or not resolution.path:
        raise ValueError("precompiled bundle resolution requires a non-empty path")
    if not os.path.isfile(resolution.path):
        raise ValueError("precompiled bundle resolution path is not a file")
    layout_expressions = request.identity.layout_expressions
    if set(resolution.layouts_by_expression) != set(layout_expressions):
        raise ValueError(
            "precompiled bundle resolution has incompatible layout metadata"
        )
    layouts_by_expression = {}
    for expression in layout_expressions:
        layout = resolution.layouts_by_expression[expression]
        if not isinstance(layout, StorageLayout):
            raise TypeError(
                "precompiled bundle layout metadata must contain StorageLayout values"
            )
        layouts_by_expression[expression] = _validate_storage_layout(
            layout.size_in_bytes,
            layout.alignment,
            description=expression,
        )
    return _CachedBundle(
        path=resolution.path,
        layouts_by_expression=layouts_by_expression,
        producer_compiler=resolution.producer_compiler,
        producer_compiler_version=resolution.producer_compiler_version,
        producer_toolkit_version=resolution.producer_toolkit_version,
    )


def _finish_bundle_resolution(
    *,
    request: BundleResolutionRequest,
    cached: _CachedBundle,
    route: str,
    key_to_expression: dict[Hashable, str],
    phase_timings_ns: dict[str, int],
    resolution_started_ns: int,
) -> BundleCompilation:
    _finish_phase(phase_timings_ns, "total", resolution_started_ns)
    resolution = BundleResolution(
        request=request,
        path=cached.path,
        layouts_by_expression=dict(cached.layouts_by_expression),
        route=route,
        producer_compiler=cached.producer_compiler,
        producer_compiler_version=cached.producer_compiler_version,
        producer_toolkit_version=cached.producer_toolkit_version,
        phase_timings_ns=dict(phase_timings_ns),
    )
    with _STATE_LOCK:
        _ROUTE_COUNTS[route] = _ROUTE_COUNTS.get(route, 0) + 1
        for phase, duration_ns in phase_timings_ns.items():
            _PHASE_COUNTS[phase] = _PHASE_COUNTS.get(phase, 0) + 1
            _PHASE_TIMINGS_NS[phase] = _PHASE_TIMINGS_NS.get(phase, 0) + duration_ns
    for observer in _ACTIVE_POST_RESOLUTION_OBSERVERS.get():
        observer(resolution)
    return _bundle_compilation(cached, key_to_expression)


def _compile_clang_bundle(
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
    phase_timings_ns: dict[str, int],
) -> _CachedBundle:
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
    )
    phase_started_ns = time.perf_counter_ns()
    try:
        write_binary_atomic(output_path, artifact_blob, scope=scope)
    finally:
        _finish_phase(phase_timings_ns, "artifact_io", phase_started_ns)
    phase_started_ns = time.perf_counter_ns()
    try:
        _write_bundle_metadata(
            output_path,
            artifact_blob,
            cached,
            cache_identity,
            scope=scope,
        )
    finally:
        _finish_phase(phase_timings_ns, "metadata_io", phase_started_ns)
    return cached


def _create_nvrtc_program(
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


def _compile_nvrtc_program(
    program: Any,
    options: list[bytes],
) -> tuple[Any, int]:
    global _NVRTC_COMPILE_PROGRAM_COUNTER

    started_ns = time.perf_counter_ns()
    with _STATE_LOCK:
        _NVRTC_COMPILE_PROGRAM_COUNTER += 1
    result = cuda_nvrtc.nvrtcCompileProgram(program, len(options), options)
    duration_ns = max(0, time.perf_counter_ns() - started_ns)
    err = result[0] if isinstance(result, tuple) else result
    return err, duration_ns


def _compile_nvrtc_bundle(
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
) -> _CachedBundle:
    source_bytes = source.encode("utf-8")
    base_options = [
        *(option.encode("ascii") for option in compiler_options),
        *cccl_include_options(include_dirs),
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
            program = _create_nvrtc_program(
                source_bytes,
                encoded_expressions,
                scope=scope,
            )
            options = [*base_options, *pch_session.options]
            err, compile_duration_ns = _compile_nvrtc_program(program, options)
            _add_phase_duration(
                phase_timings_ns,
                "nvrtc_compile",
                compile_duration_ns,
            )
            pch_fallback_log = ""
            used_pch = pch_session.enabled
            if err != cuda_nvrtc.nvrtcResult.NVRTC_SUCCESS and pch_session.enabled:
                pch_fallback_log = get_nvrtc_program_log(program)
                pch_session.disable_after_failure(compile_duration_ns)
                cuda_nvrtc.nvrtcDestroyProgram(program)
                program = None
                program = _create_nvrtc_program(
                    source_bytes,
                    encoded_expressions,
                    scope=scope,
                )
                err, retry_duration_ns = _compile_nvrtc_program(
                    program,
                    base_options,
                )
                _add_phase_duration(
                    phase_timings_ns,
                    "nvrtc_compile",
                    retry_duration_ns,
                )
                pch_session.phase_timings_ns["pch_fallback"] += retry_duration_ns
                used_pch = False
            if err != cuda_nvrtc.nvrtcResult.NVRTC_SUCCESS:
                program_log = get_nvrtc_program_log(program)
                if pch_fallback_log:
                    program_log = (
                        "PCH-enabled compilation failed:\n"
                        f"{pch_fallback_log}\n"
                        "Retry without PCH failed:\n"
                        f"{program_log}"
                    )
                raise DSLRuntimeError(
                    f"Failed compiling {scope} provider shim to LTO-IR.",
                    cause=RuntimeError(program_log),
                )
            if used_pch:
                pch_session.record_success(
                    cuda_nvrtc,
                    program,
                    compile_duration_ns=compile_duration_ns,
                    program_log=get_nvrtc_program_log(program),
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

            phase_started_ns = time.perf_counter_ns()
            try:
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
                _finish_phase(
                    phase_timings_ns,
                    "lto_retrieval",
                    phase_started_ns,
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

    cached = _CachedBundle(
        path=output_path,
        layouts_by_expression=layouts_by_expression,
        producer_compiler="nvrtc",
        producer_compiler_version=nvrtc_version,
    )
    phase_started_ns = time.perf_counter_ns()
    try:
        write_binary_atomic(output_path, artifact_blob, scope=scope)
    finally:
        _finish_phase(phase_timings_ns, "artifact_io", phase_started_ns)
    phase_started_ns = time.perf_counter_ns()
    try:
        _write_bundle_metadata(
            output_path,
            artifact_blob,
            cached,
            cache_identity,
            scope=scope,
        )
    finally:
        _finish_phase(phase_timings_ns, "metadata_io", phase_started_ns)
    return cached


def _compile_bundle_source(
    source: str,
    *,
    layout_probes: Iterable[LayoutProbe],
    scope: str,
    provider_dir: str,
    registered_headers: Callable[[], dict[str, str]],
    select_bundle_format: Callable[[], str],
    resolve_nvrtc_sm_arch: Callable[[], str],
    resolve_nvrtc_arch: Callable[[], str],
    symbols: Iterable[str],
    which: Callable[[str], str | None] = shutil.which,
    initial_phase_timings_ns: Mapping[str, int] | None = None,
    resolution_started_ns: int | None = None,
) -> BundleCompilation:
    global _BUNDLE_COMPILE_COUNTER

    if resolution_started_ns is None:
        resolution_started_ns = time.perf_counter_ns()
    phase_timings_ns = dict(initial_phase_timings_ns or {})
    if "total" in phase_timings_ns:
        raise ValueError("initial provider phase timings cannot include total")
    for phase, duration_ns in phase_timings_ns.items():
        if (
            not isinstance(phase, str)
            or not phase
            or isinstance(duration_ns, bool)
            or not isinstance(duration_ns, int)
            or duration_ns < 0
        ):
            raise ValueError(
                "initial provider phase timings require nonempty string phases "
                "and nonnegative integer nanoseconds"
            )
    phase_started_ns = time.perf_counter_ns()
    prepared_probes = _prepare_layout_probes(source, layout_probes)
    source = prepared_probes.source
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    _finish_phase(phase_timings_ns, "prepare", phase_started_ns)

    phase_started_ns = time.perf_counter_ns()
    bundle_format = select_bundle_format()
    if prepared_probes.expressions and bundle_format != "ltoir":
        raise DSLRuntimeError(
            f"{scope} provider layout metadata requires an NVRTC LTO-IR bundle; "
            f"set {BUNDLE_FORMAT_ENV}=ltoir."
        )
    bundle_sm_arch = "none"
    if bundle_format == "bc":
        bundle_arch = "nvptx64"
    else:
        bundle_sm_arch = resolve_nvrtc_sm_arch()
        bundle_arch = resolve_nvrtc_arch()
    _finish_phase(phase_timings_ns, "target", phase_started_ns)

    compiler_options = bundle_compiler_options(bundle_format, bundle_arch)
    identity = make_bundle_identity(
        source_hash=source_hash,
        bundle_format=bundle_format,
        bundle_arch=bundle_arch,
        bundle_sm_arch=bundle_sm_arch,
        compiler_options=compiler_options,
        layout_expressions=prepared_probes.expressions,
    )
    bundle_symbols = tuple(sorted(set(symbols)))
    request = BundleResolutionRequest(
        identity=identity,
        source=source,
        symbols=bundle_symbols,
    )

    resolvers = _ACTIVE_PRECOMPILE_RESOLVERS.get()
    if resolvers:
        for resolver in reversed(resolvers):
            phase_started_ns = time.perf_counter_ns()
            try:
                resolution = resolver.callback(request)
            finally:
                _finish_phase(
                    phase_timings_ns,
                    resolver.phase,
                    phase_started_ns,
                )
            if resolution is None:
                continue
            return _finish_bundle_resolution(
                request=request,
                cached=_validated_precompiled_bundle(
                    resolution,
                    request,
                    route=resolver.route,
                ),
                route=resolver.route,
                key_to_expression=prepared_probes.key_to_expression,
                phase_timings_ns=phase_timings_ns,
                resolution_started_ns=resolution_started_ns,
            )

    phase_started_ns = time.perf_counter_ns()
    include_dirs = cccl_include_dirs(
        required_cccl_headers(source, registered_headers=registered_headers),
        scope=scope,
        provider_dir=provider_dir,
    )
    _finish_phase(phase_timings_ns, "header_resolution", phase_started_ns)
    if bundle_format == "ltoir":
        preload_toolkit_nvrtc(include_dirs)

    phase_started_ns = time.perf_counter_ns()
    include_identity = include_dirs_identity(include_dirs)
    include_key = include_identity.digest
    _finish_phase(phase_timings_ns, "header_fingerprint", phase_started_ns)
    for root_identity in include_identity.roots:
        phase = f"header_identity_{root_identity.method.replace('-', '_')}"
        phase_timings_ns[phase] = (
            phase_timings_ns.get(phase, 0) + root_identity.duration_ns
        )
    clangxx: str | None = None
    clang_version: str | None = None
    if bundle_format == "ltoir":
        nvrtc_version_tuple = get_nvrtc_version_tuple()
        nvrtc_version = (
            None
            if nvrtc_version_tuple is None
            else f"{nvrtc_version_tuple[0]}.{nvrtc_version_tuple[1]}"
        )
        cache_compiler_version = (
            nvrtc_version
            if nvrtc_version is not None
            else _UNKNOWN_COMPILER_PROCESS_TOKEN
        )
    else:
        nvrtc_version_tuple = None
        nvrtc_version = None
        clangxx, clang_version, cache_compiler_version = resolve_clang_compiler(which)
    cache_identity = make_bundle_cache_identity(
        identity,
        include_key=include_key,
        producer_compiler_version=cache_compiler_version,
    )

    # Source inspection is independent of JIT compilation. Keep the dump useful
    # even when this process or a previous process already populated the cache.
    phase_started_ns = time.perf_counter_ns()
    maybe_dump_source(source, source_hash)
    _finish_phase(phase_timings_ns, "source_dump", phase_started_ns)

    phase_started_ns = time.perf_counter_ns()
    cached = _revalidate_memory_cached_bundle(
        cache_identity,
        prepared_probes.expressions,
    )
    _finish_phase(phase_timings_ns, "memory_cache", phase_started_ns)
    if cached is not None:
        return _finish_bundle_resolution(
            request=request,
            cached=cached,
            route=RESOLUTION_ROUTE_MEMORY,
            key_to_expression=prepared_probes.key_to_expression,
            phase_timings_ns=phase_timings_ns,
            resolution_started_ns=resolution_started_ns,
        )

    phase_started_ns = time.perf_counter_ns()
    cache_dir = ensure_cache_dir(scope)
    output_path = os.path.join(
        cache_dir,
        f"{cache_identity.artifact_stem}.{bundle_format}",
    )
    lock_started_ns = time.perf_counter_ns()
    with artifact_lock(output_path, scope=scope):
        _finish_phase(phase_timings_ns, "artifact_lock", lock_started_ns)

        # A thread or process may have populated the artifact while this caller
        # waited for the per-artifact lock.
        phase_started_ns = time.perf_counter_ns()
        cached = _revalidate_memory_cached_bundle(
            cache_identity,
            prepared_probes.expressions,
        )
        _finish_phase(phase_timings_ns, "memory_cache_after_lock", phase_started_ns)
        if cached is not None:
            return _finish_bundle_resolution(
                request=request,
                cached=cached,
                route=RESOLUTION_ROUTE_MEMORY,
                key_to_expression=prepared_probes.key_to_expression,
                phase_timings_ns=phase_timings_ns,
                resolution_started_ns=resolution_started_ns,
            )

        phase_started_ns = time.perf_counter_ns()
        cached = None
        if os.path.exists(output_path):
            cached = _load_bundle_metadata(
                output_path,
                prepared_probes.expressions,
                cache_identity,
            )
        _finish_phase(phase_timings_ns, "disk_cache", phase_started_ns)
        if cached is not None:
            with _STATE_LOCK:
                _SOURCE_CACHE[cache_identity.cache_key] = cached
            return _finish_bundle_resolution(
                request=request,
                cached=cached,
                route=RESOLUTION_ROUTE_DISK,
                key_to_expression=prepared_probes.key_to_expression,
                phase_timings_ns=phase_timings_ns,
                resolution_started_ns=resolution_started_ns,
            )

        phase_started_ns = time.perf_counter_ns()
        if bundle_format == "bc":
            cached = _compile_clang_bundle(
                source,
                output_path=output_path,
                cache_dir=cache_dir,
                cache_identity=cache_identity,
                compiler_options=compiler_options,
                include_dirs=include_dirs,
                scope=scope,
                clangxx=clangxx,
                clang_version=clang_version,
                phase_timings_ns=phase_timings_ns,
            )
            route = RESOLUTION_ROUTE_CLANG
        else:
            cached = _compile_nvrtc_bundle(
                source,
                output_path=output_path,
                cache_identity=cache_identity,
                compiler_options=compiler_options,
                include_dirs=include_dirs,
                prepared_probes=prepared_probes,
                nvrtc_version=nvrtc_version,
                nvrtc_version_tuple=nvrtc_version_tuple,
                bundle_arch=bundle_arch,
                bundle_sm_arch=bundle_sm_arch,
                header_identity=include_identity.digest,
                phase_timings_ns=phase_timings_ns,
                scope=scope,
            )
            route = RESOLUTION_ROUTE_NVRTC
        with _STATE_LOCK:
            _SOURCE_CACHE[cache_identity.cache_key] = cached
            _BUNDLE_COMPILE_COUNTER += 1
    _finish_phase(phase_timings_ns, "compiler", phase_started_ns)
    return _finish_bundle_resolution(
        request=request,
        cached=cached,
        route=route,
        key_to_expression=prepared_probes.key_to_expression,
        phase_timings_ns=phase_timings_ns,
        resolution_started_ns=resolution_started_ns,
    )


def compile_bundle_source(
    source: str,
    *,
    scope: str,
    provider_dir: str,
    registered_headers: Callable[[], dict[str, str]],
    select_bundle_format: Callable[[], str],
    resolve_nvrtc_sm_arch: Callable[[], str],
    resolve_nvrtc_arch: Callable[[], str],
    symbols: Iterable[str] = (),
    which: Callable[[str], str | None] = shutil.which,
    initial_phase_timings_ns: Mapping[str, int] | None = None,
    resolution_started_ns: int | None = None,
) -> str:
    """Compile a provider bundle and return its linkable artifact path."""

    return _compile_bundle_source(
        source,
        layout_probes=(),
        scope=scope,
        provider_dir=provider_dir,
        registered_headers=registered_headers,
        select_bundle_format=select_bundle_format,
        resolve_nvrtc_sm_arch=resolve_nvrtc_sm_arch,
        resolve_nvrtc_arch=resolve_nvrtc_arch,
        symbols=symbols,
        which=which,
        initial_phase_timings_ns=initial_phase_timings_ns,
        resolution_started_ns=resolution_started_ns,
    ).path


def compile_bundle_source_with_layouts(
    source: str,
    *,
    layout_probes: Iterable[LayoutProbe],
    scope: str,
    provider_dir: str,
    registered_headers: Callable[[], dict[str, str]],
    select_bundle_format: Callable[[], str],
    resolve_nvrtc_sm_arch: Callable[[], str],
    resolve_nvrtc_arch: Callable[[], str],
    symbols: Iterable[str] = (),
    which: Callable[[str], str | None] = shutil.which,
    initial_phase_timings_ns: Mapping[str, int] | None = None,
    resolution_started_ns: int | None = None,
) -> BundleCompilation:
    """Compile one LTO-IR bundle and recover exact layouts from that program."""

    return _compile_bundle_source(
        source,
        layout_probes=layout_probes,
        scope=scope,
        provider_dir=provider_dir,
        registered_headers=registered_headers,
        select_bundle_format=select_bundle_format,
        resolve_nvrtc_sm_arch=resolve_nvrtc_sm_arch,
        resolve_nvrtc_arch=resolve_nvrtc_arch,
        symbols=symbols,
        which=which,
        initial_phase_timings_ns=initial_phase_timings_ns,
        resolution_started_ns=resolution_started_ns,
    )


def reset_compile_state() -> None:
    global _BUNDLE_COMPILE_COUNTER, _NVRTC_COMPILE_PROGRAM_COUNTER
    with _STATE_LOCK:
        _BUNDLE_COMPILE_COUNTER = 0
        _NVRTC_COMPILE_PROGRAM_COUNTER = 0
        _SOURCE_CACHE.clear()
        _ROUTE_COUNTS.clear()
        _PHASE_COUNTS.clear()
        _PHASE_TIMINGS_NS.clear()


def get_compile_counter() -> int:
    with _STATE_LOCK:
        return _BUNDLE_COMPILE_COUNTER


def get_nvrtc_compile_program_counter() -> int:
    with _STATE_LOCK:
        return _NVRTC_COMPILE_PROGRAM_COUNTER


def get_bundle_telemetry() -> BundleTelemetry:
    with _STATE_LOCK:
        return BundleTelemetry(
            route_counts=dict(_ROUTE_COUNTS),
            phase_counts=dict(_PHASE_COUNTS),
            phase_timings_ns=dict(_PHASE_TIMINGS_NS),
        )


def managed_bundle_paths() -> frozenset[str]:
    """Return bundle paths this process added to CUTLASS link options."""

    with _STATE_LOCK:
        return frozenset(_MANAGED_BUNDLE_PATHS)
