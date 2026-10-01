# v1 (cccl.c.parallel, NVRTC) — the cccl_build_config every build passes.
# Selected at CMake configure time and configure_file'd to the build dir as
# `_bindings_build_config.pxi`.
#
# v1's struct has four fields; it carries no enable_pch, because NVRTC has no
# precompiled-header cache to enable. cuda.compute needs none of the four, so a
# fresh zeroed config is equivalent to the NULL v1 passed before. Routing all
# twelve call sites in _bindings_impl.pyx through the same per-build
# _BuildConfig keeps them backend-agnostic; the difference between v1 and v2
# lives here.

cdef extern from "cccl/c/types.h":
    cdef struct cccl_build_config:
        const char** extra_compile_flags
        size_t num_extra_compile_flags
        const char** extra_include_dirs
        size_t num_extra_include_dirs


def set_pch_cache_dir(path):
    # NVRTC has no precompiled-header cache, so there is nothing to point at.
    pass


cdef class _BuildConfig:
    cdef cccl_build_config config

    cdef cccl_build_config* ptr(self) noexcept nogil:
        return &self.config


cdef _BuildConfig _get_build_config():
    cdef _BuildConfig bc = _BuildConfig.__new__(_BuildConfig)
    bc.config.extra_compile_flags = NULL
    bc.config.num_extra_compile_flags = 0
    bc.config.extra_include_dirs = NULL
    bc.config.num_extra_include_dirs = 0
    return bc


cdef inline str _pch_cache_dir_impl():
    # No PCH cache exists on this backend, so there is no directory to report.
    return None
