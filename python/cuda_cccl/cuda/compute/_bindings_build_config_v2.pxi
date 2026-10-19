# v2 (cccl.c.parallel.v2, HostJIT) — the build configuration every build passes.
# Selected at CMake configure time and configure_file'd to the build dir as
# `_bindings_build_config.pxi`.
#
# Precompiled headers are enabled once a cache directory is supplied. Parsing
# the CUB / libcudacxx / Thrust bundle dominates a build, and one cached header
# pair serves every algorithm. Leaving them on is safe: an unusable cache or a
# stale entry falls back to building without one.
#
# The directory is resolved lazily, not here: on the first build,
# `cuda.compute._pch.ensure_configured()` (invoked from `_cccl_interop`'s
# `call_build` / `call_compile`) resolves it and calls `set_pch_cache_dir()`.
# Importing this extension has no filesystem side effects.

cdef extern from "cccl/c/types.h":
    cdef struct cccl_build_config:
        const char** extra_compile_flags
        size_t num_extra_compile_flags
        const char** extra_include_dirs
        size_t num_extra_include_dirs
        int enable_pch
        int verbose
        const char* pch_cache_dir


# The current cache directory as UTF-8 bytes (b"" = PCH off), rebound by
# set_pch_cache_dir(). It is read only through _get_build_config(), which
# snapshots the reference into a fresh per-build object -- so a build never
# shares mutable C state with a concurrent reconfigure().
cdef bytes _pch_cache_dir_bytes = b""


def set_pch_cache_dir(path):
    """Point new builds at the precompiled-header cache at `path`.

    Passing None or an empty path disables precompiled headers, which is also
    the state before this is called.
    """
    global _pch_cache_dir_bytes
    _pch_cache_dir_bytes = b"" if not path else str(path).encode("utf-8")


cdef class _BuildConfig:
    # A fresh, self-contained config per build. It owns the bytes buffer that
    # config.pch_cache_dir points into, so a concurrent set_pch_cache_dir() can
    # neither free nor tear anything this build is still reading under nogil.
    cdef cccl_build_config config
    cdef bytes _path

    cdef cccl_build_config* ptr(self) noexcept nogil:
        return &self.config


cdef _BuildConfig _get_build_config():
    cdef _BuildConfig bc = _BuildConfig.__new__(_BuildConfig)
    bc._path = _pch_cache_dir_bytes
    bc.config.extra_compile_flags = NULL
    bc.config.num_extra_compile_flags = 0
    bc.config.extra_include_dirs = NULL
    bc.config.num_extra_include_dirs = 0
    bc.config.verbose = 0
    if bc._path:
        bc.config.pch_cache_dir = <const char*>bc._path
        bc.config.enable_pch = 1
    else:
        bc.config.pch_cache_dir = NULL
        bc.config.enable_pch = 0
    return bc


cdef inline str _pch_cache_dir_impl():
    return _pch_cache_dir_bytes.decode("utf-8") if _pch_cache_dir_bytes else None
