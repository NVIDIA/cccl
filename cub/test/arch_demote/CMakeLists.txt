# Regression test for NVIDIA/cccl#11403 and NVIDIA/cccl#11440: cub::detail::ptx_compute_cap() must report the
# architecture of the device code that is actually executed. This includes the case where device-linking against a
# translation unit compiled for a lower architecture clamps the virtual architecture metadata of every kernel in the
# module (-rdc=true).

if (NOT CUB_ENABLE_RDC_TESTS)
  return()
endif()

# The lower architecture is requested via a separate target with its own CUDA_ARCHITECTURES, which needs separable
# compilation. This is not supported with clang-cuda (it requires the architectures of all targets to be known).
if (NOT "${CMAKE_CUDA_COMPILER_ID}" STREQUAL "NVIDIA")
  return()
endif()

set(arch_demote_target cub.test.arch_demote)
set(arch_demote_other_target cub.test.arch_demote.other)

# Compiled for the lowest virtual architecture only. The test never runs its code, but device-linking it into the test
# executable, which is compiled for the architectures of the test suite, lowers the virtual architecture of the module.
add_library(${arch_demote_other_target} OBJECT other.cu)
cccl_configure_target(${arch_demote_other_target})
target_link_libraries(
  ${arch_demote_other_target}
  PRIVATE cub.compiler_interface
)
set_target_properties(
  ${arch_demote_other_target}
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES 75-virtual
)

cccl_add_executable(
  ${arch_demote_target}
  SOURCES main.cu $<TARGET_OBJECTS:${arch_demote_other_target}>
  METATARGET_PATH cub.test.device.arch_demote
)
target_link_libraries(${arch_demote_target} PRIVATE cub.compiler_interface)
set_target_properties(
  ${arch_demote_target}
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)
target_compile_definitions(${arch_demote_target} PRIVATE CCCL_ENABLE_ASSERTIONS)

rapids_test_add(
  NAME ${arch_demote_target}
  # gersemi: off
  COMMAND
    "${CMAKE_COMMAND}"
       "-DCCCL_SOURCE_DIR=${CCCL_SOURCE_DIR}"
       "-DTEST=$<TARGET_FILE:${arch_demote_target}>"
       -P "${CUB_SOURCE_DIR}/test/run_test.cmake"
  # gersemi: on
  GPUS 1
  PERCENT ${CCCL_TEST_GPU_PERCENT_SMALL}
)
