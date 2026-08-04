# v2 (cccl.c.parallel.v2, HostJIT) — the cccl_build_config every build passes,
# plus the PCH cache-directory accessor. Selected at CMake configure time and
# configure_file'd to the build dir as `_bindings_build_config.pxi`.
#
# v2's struct carries two fields v1's does not: enable_pch and verbose.
#
# Precompiled headers are turned on here, for every build. Parsing the CUB /
# libcudacxx / Thrust bundle dominates a HostJIT build, and one cached PCH pair
# serves all twelve algorithms, so this is the difference between a ~3.4s and a
# ~1.1s cold build. It is safe to leave on unconditionally: when the cache is
# unusable, or an entry is stale, the C layer builds without a PCH instead of
# failing. Set CCCL_ENABLE_PCH=0 to disable — the C layer treats that as a kill
# switch that beats this value.

cdef extern from "cccl/c/types.h":
    cdef struct cccl_build_config:
        const char** extra_compile_flags
        size_t num_extra_compile_flags
        const char** extra_include_dirs
        size_t num_extra_include_dirs
        int enable_pch
        int verbose


cdef extern from "cccl/c/pch.h":
    cdef size_t cccl_hostjit_pch_cache_dir(char*, size_t) nogil


# One config shared by every build. The C layer copies what it needs during the
# call and retains no pointer into it, so a single module-level instance is
# enough and there is nothing to keep alive per build.
cdef cccl_build_config _shared_build_config

_shared_build_config.extra_compile_flags = NULL
_shared_build_config.num_extra_compile_flags = 0
_shared_build_config.extra_include_dirs = NULL
_shared_build_config.num_extra_include_dirs = 0
_shared_build_config.enable_pch = 1
_shared_build_config.verbose = 0


cdef inline cccl_build_config* _get_build_config() noexcept nogil:
    return &_shared_build_config




cdef inline str _pch_cache_dir_impl():
    # Two-call snprintf idiom: ask for the size, then fill. The path comes from
    # an environment-dependent chain resolved in the C layer, so there is no
    # bounded length to assume here.
    cdef size_t needed = cccl_hostjit_pch_cache_dir(NULL, 0)
    if needed == 0:
        return None

    cdef char* buf = <char*>malloc(needed)
    if buf == NULL:
        raise MemoryError()
    try:
        if cccl_hostjit_pch_cache_dir(buf, needed) == 0:
            return None
        return buf[:needed - 1].decode("utf-8")
    finally:
        free(buf)
