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

# Compiled for the lowest virtual architecture only. The test never runs its code, but device-linking it into the test
# executable, which is compiled for the architectures of the test suite, lowers the virtual architecture of the module.
set(arch_demote_other_target cub.test.arch_demote.other)
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

cub_add_test(arch_demote_target arch_demote "" main.cu 0)
cub_configure_cuda_target(${arch_demote_target} RDC ON)
target_sources(
  ${arch_demote_target}
  PRIVATE $<TARGET_OBJECTS:${arch_demote_other_target}>
)
