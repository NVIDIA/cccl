# Simple smoke test verifying the CUDA runtime is functional (device visible,
# kernel launch + managed-memory round-trip work). Invoked explicitly from CI
# via ci/build_common.sh before GPU test presets run. Not registered in CTest
# (GPU-only; CPU test presets such as thrust-cpu would pick it up otherwise).
#
# Declare CUDA inside this subproject only so the root project() can stay
# CXX-only (external consumers via add_subdirectory(cccl) see no change).
project(CCCL_CUDA_SMOKE LANGUAGES CUDA)

cccl_get_catch2()

cccl_add_executable(${CCCL_CUDA_SMOKE_TARGET} SOURCES cuda_runtime_smoke.cu)

target_link_libraries(${CCCL_CUDA_SMOKE_TARGET} PRIVATE Catch2::Catch2WithMain)
