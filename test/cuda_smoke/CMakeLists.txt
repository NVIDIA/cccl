# Simple smoke test verifying the CUDA runtime is functional (device visible,
# kernel launch + managed-memory round-trip work). Registered in CTest with
# the "cuda_smoke" label so CI can run it separately before the main test
# suite via `ctest -L cuda_smoke`.
#
# Declare CUDA inside this subproject only so the root project() can stay
# CXX-only (external consumers via add_subdirectory(cccl) see no change).
project(CCCL_CUDA_SMOKE LANGUAGES CUDA)

cccl_get_catch2()

cccl_add_executable(
  ${CCCL_CUDA_SMOKE_TARGET}
  ADD_CTEST
  SOURCES cuda_runtime_smoke.cu
)

target_link_libraries(${CCCL_CUDA_SMOKE_TARGET} PRIVATE Catch2::Catch2WithMain)
set_tests_properties(${CCCL_CUDA_SMOKE_TARGET} PROPERTIES LABELS "cuda_smoke")
