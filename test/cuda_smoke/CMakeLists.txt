# Smoke tests verifying the CUDA runtime is functional (device visible,
# kernel launch, memory allocation round-trips, etc). Invoked explicitly from
# CI via ci/build_common.sh before GPU test presets run. Not registered in
# CTest (GPU-only; CPU test presets such as thrust-cpu would pick it up
# otherwise).
#
# Each check is built as its own executable so that a crash / corrupted CUDA
# context in one check cannot prevent the others from running and reporting
# their own pass/fail status.
#
# Declare CUDA inside this subproject only so the root project() can stay
# CXX-only (external consumers via add_subdirectory(cccl) see no change).
project(CCCL_CUDA_SMOKE LANGUAGES CUDA)

cccl_get_catch2()

set(
  cuda_smoke_checks
  device_available
  managed_memory
  device_memory
  async_mempool
  pinned_memory
  pinned_memory_mapped
)

foreach (check IN LISTS cuda_smoke_checks)
  set(target cccl.test.cuda_smoke.${check})
  cccl_add_executable(${target} SOURCES ${check}.cu)
  target_link_libraries(${target} PRIVATE Catch2::Catch2WithMain)
endforeach()
