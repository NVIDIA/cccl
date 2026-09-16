# Regression test for NVIDIA/cccl#11403: -rdc=true device-linking against a translation unit compiled for a
# different architecture clamps the virtual architecture metadata of every kernel in the module.

if (NOT CUB_ENABLE_RDC_TESTS)
  return()
endif()

set(arch_demote_target cub.test.arch_demote)

cccl_add_executable(
  ${arch_demote_target}
  SOURCES main.cu other.cu
  METATARGET_PATH cub.test.device.arch_demote
)
target_link_libraries(${arch_demote_target} PRIVATE cub.compiler_interface)

# Compile each translation unit for a different, fixed architecture, independent of CUDA_ARCHITECTURES, to force
# nvlink to device-link mismatched architectures together.
set_target_properties(
  ${arch_demote_target}
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES OFF
)
set_source_files_properties(
  main.cu
  PROPERTIES
    COMPILE_OPTIONS "--generate-code=arch=compute_80,code=[compute_80,sm_80]"
)
set_source_files_properties(
  other.cu
  PROPERTIES
    COMPILE_OPTIONS "--generate-code=arch=compute_75,code=[compute_75,sm_75]"
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

# Regression test for https://github.com/NVIDIA/cccl/pull/11440#issuecomment-5699615518, a counterexample to the fix
# for NVIDIA/cccl#11403.
set(arch_native_mismatch_target cub.test.arch_native_mismatch)

cccl_add_executable(
  ${arch_native_mismatch_target}
  SOURCES native_mismatch.cu
  METATARGET_PATH cub.test.arch_native_mismatch
)
target_link_libraries(
  ${arch_native_mismatch_target}
  PRIVATE cub.compiler_interface
)

# Compile a native SASS binary for a lower virtual architecture (Ampere: compute_80 -> sm_86) alongside a separate,
# higher virtual-only PTX (compute_86) in the same fatbin. On an sm_86 GPU, CUDA runs the native sm_86 binary
# (compiled with __CUDA_ARCH__ == 800) rather than JIT-compiling the compute_86 PTX.
set_target_properties(
  ${arch_native_mismatch_target}
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES OFF
)
set_source_files_properties(
  native_mismatch.cu
  PROPERTIES
    COMPILE_OPTIONS
      "--generate-code=arch=compute_80,code=sm_86;--generate-code=arch=compute_86,code=compute_86"
)
target_compile_definitions(
  ${arch_native_mismatch_target}
  PRIVATE CCCL_ENABLE_ASSERTIONS
)

rapids_test_add(
  NAME ${arch_native_mismatch_target}
  # gersemi: off
  COMMAND
    "${CMAKE_COMMAND}"
       "-DCCCL_SOURCE_DIR=${CCCL_SOURCE_DIR}"
       "-DTEST=$<TARGET_FILE:${arch_native_mismatch_target}>"
       -P "${CUB_SOURCE_DIR}/test/run_test.cmake"
  # gersemi: on
  GPUS 1
  PERCENT ${CCCL_TEST_GPU_PERCENT_SMALL}
)
