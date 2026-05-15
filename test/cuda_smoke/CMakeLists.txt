# Registered as a CTest setup fixture (${CCCL_CUDA_SMOKE_FIXTURE}); library
# tests get FIXTURES_REQUIRED via the deferred walker in
# cmake/CCCLCudaSmokeFixture.cmake, which emits ${CMAKE_BINARY_DIR}/CTestCustom.cmake.
#
# Declare CUDA inside this subproject only so the root project() can stay
# CXX-only (external consumers via add_subdirectory(cccl) see no change).
project(CCCL_CUDA_SMOKE LANGUAGES CXX CUDA)

cccl_get_catch2()

cccl_add_executable(${CCCL_CUDA_SMOKE_TARGET} SOURCES cuda_runtime_smoke.cu)

target_link_libraries(${CCCL_CUDA_SMOKE_TARGET} PRIVATE Catch2::Catch2WithMain)

# The smoke TEST_CASEs are tagged [.smoke] (leading dot hides them from the
# default Catch2 listing when the binary is run by hand). Pass the tag
# explicitly so CTest actually selects them; without it, Catch2 v3 exits
# non-zero with "No tests ran" and the fixture poisons every downstream test.
add_test(
  NAME
  ${CCCL_CUDA_SMOKE_TARGET}
  COMMAND
  "$<TARGET_FILE:${CCCL_CUDA_SMOKE_TARGET}>"
  "[.smoke]"
)

set_tests_properties(
  ${CCCL_CUDA_SMOKE_TARGET}
  PROPERTIES FIXTURES_SETUP ${CCCL_CUDA_SMOKE_FIXTURE}
)
