# v1 (cccl.c.parallel, NVRTC) — the cccl_build_config every build passes.
# Selected at CMake configure time and configure_file'd to the build dir as
# `_bindings_build_config.pxi`.
#
# v1's struct has four fields; it carries no enable_pch, because NVRTC has no
# precompiled-header cache to enable. cuda.compute needs none of the four, so
# every build passes NULL — exactly what it passed before the `_ex` entry
# points were adopted. Adopting them uniformly keeps the twelve call sites in
# _bindings_impl.pyx backend-agnostic; the difference between v1 and v2 lives
# here.

cdef extern from "cccl/c/types.h":
    cdef struct cccl_build_config:
        const char** extra_compile_flags
        size_t num_extra_compile_flags
        const char** extra_include_dirs
        size_t num_extra_include_dirs


cdef inline cccl_build_config* _get_build_config() noexcept nogil:
    return NULL




def set_pch_cache_dir(path):
    # NVRTC has no precompiled-header cache, so there is nothing to point at.
    pass


cdef inline str _pch_cache_dir_impl():
    # No PCH cache exists on this backend, so there is no directory to report.
    return None
