# Sweep the STF and places headers for exceptions escaping destructors and noexcept
# functions.
#
# bugprone-exception-escape is enabled repo-wide in .clang-tidy, but the clang-tidy CI job
# analyzes translation units compiled by clang's CUDA front end, which STF and places are
# excluded from (see ci/build_tidy.sh). Neither is checked there as a result.
#
# This sweep parses each header as host C++ instead. An escaping exception is a host-side
# property, so nothing is lost by ignoring device code, and a host C++ parse needs no CUDA
# support at all. That is what lets the check cover STF even where STF compilation is
# disabled, so this file is deliberately included unconditionally.

cccl_get_cudatoolkit()

cccl_tidy_add_header_sweep(
  exception_escape
  cudax/include
  CHECKS "-*,bugprone-exception-escape"
  GLOBS
    "cuda/experimental/stf.cuh"
    "cuda/experimental/__stf/*.cuh"
    "cuda/experimental/places.cuh"
    "cuda/experimental/__places/*.cuh"
  EXCLUDES
    # These headers use device intrinsics (atomicCAS, threadIdx, ...) that a host C++ parse
    # cannot resolve, so they can only be analyzed by a CUDA compilation.
    "cuda/experimental/__stf/places/exec/host/callback_queues.cuh"
    "cuda/experimental/__stf/stream/interfaces/slice_reduction_ops.cuh"
    # Not self-contained in a host C++ parse, but reached through stf.cuh, hence still
    # analyzed.
    "cuda/experimental/__stf/stackable/stackable_task_dep.cuh"
  INCLUDE_DIRECTORIES
    "${CCCL_SOURCE_DIR}/cudax/include"
    "${CCCL_SOURCE_DIR}/libcudacxx/include"
  SYSTEM_INCLUDE_DIRECTORIES "${CUDAToolkit_INCLUDE_DIRS}"
)
