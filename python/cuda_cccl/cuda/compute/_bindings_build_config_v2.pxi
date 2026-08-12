# v2 (cccl.c.parallel.v2, HostJIT) — the build configuration every build passes.
# Selected at CMake configure time and configure_file'd to the build dir as
# `_bindings_build_config.pxi`.
#
# Precompiled headers are enabled once a cache directory is supplied. Parsing
# the CUB / libcudacxx / Thrust bundle dominates a build, and one cached header
# pair serves every algorithm. Leaving them on is safe: an unusable cache or a
# stale entry falls back to building without one.
#
# The directory is pushed in by `_bindings.py` once this extension has finished
# importing, rather than resolved here. Resolving it during module
# initialization would mean importing `cuda.compute._pch` while `cuda.compute`
# is still initializing, and the surrounding package is not yet importable at
# that point.

cdef extern from "cccl/c/types.h":
    cdef struct cccl_build_config:
        const char** extra_compile_flags
        size_t num_extra_compile_flags
        const char** extra_include_dirs
        size_t num_extra_include_dirs
        int enable_pch
        const char* pch_cache_dir
        int verbose


# One config shared by every build. Its contents are copied during the call and
# no pointer into it is retained, so a single module-level instance suffices and
# there is nothing to keep alive per build.
cdef cccl_build_config _shared_build_config

# Keeps the encoded cache path alive for as long as the config points at it.
cdef bytes _pch_cache_dir_bytes = b""

_shared_build_config.extra_compile_flags = NULL
_shared_build_config.num_extra_compile_flags = 0
_shared_build_config.extra_include_dirs = NULL
_shared_build_config.num_extra_include_dirs = 0
_shared_build_config.pch_cache_dir = NULL
_shared_build_config.enable_pch = 0
_shared_build_config.verbose = 0


def set_pch_cache_dir(path):
    """Build against the precompiled-header cache at `path`.

    Passing None or an empty path disables precompiled headers, which is also
    the state before this is called.
    """
    global _pch_cache_dir_bytes
    _pch_cache_dir_bytes = b"" if not path else str(path).encode("utf-8")
    if _pch_cache_dir_bytes:
        _shared_build_config.pch_cache_dir = <const char*>_pch_cache_dir_bytes
        _shared_build_config.enable_pch = 1
    else:
        _shared_build_config.pch_cache_dir = NULL
        _shared_build_config.enable_pch = 0


cdef inline cccl_build_config* _get_build_config() noexcept nogil:
    return &_shared_build_config


cdef inline str _pch_cache_dir_impl():
    return _pch_cache_dir_bytes.decode("utf-8") if _pch_cache_dir_bytes else None
