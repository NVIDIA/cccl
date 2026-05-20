# Simple smoke test verifying the CUDA runtime is functional (device visible,
# kernel launch + managed-memory round-trip work). Invoked explicitly from CI
# via ci/build_common.sh before running GPU tests to fail fast on broken CUDA
# installs. Also registered as a CTest test for local test runs.
#
# Declare CUDA inside this subproject only so the root project() can stay
# CXX-only (external consumers via add_subdirectory(cccl) see no change).
project(CCCL_CUDA_SMOKE LANGUAGES CUDA)

cccl_get_catch2()

cccl_add_executable(${CCCL_CUDA_SMOKE_TARGET} SOURCES cuda_runtime_smoke.cu)

target_link_libraries(${CCCL_CUDA_SMOKE_TARGET} PRIVATE Catch2::Catch2WithMain)

add_test(
  NAME ${CCCL_CUDA_SMOKE_TARGET}
  COMMAND "$<TARGET_FILE:${CCCL_CUDA_SMOKE_TARGET}>"
)
