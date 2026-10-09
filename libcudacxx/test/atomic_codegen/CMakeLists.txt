if (libcudacxx_codegen_filecheck_atomics_sass)
  add_subdirectory(sass)
endif()

if (NOT libcudacxx_codegen_filecheck_atomics_ptx)
  return()
endif()

add_custom_target(libcudacxx.test.atomics.ptx)

if ("NVHPC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  return()
endif()

file(GLOB libcudacxx_atomic_codegen_tests CONFIGURE_DEPENDS "*.cu")
if (NOT libcudacxx_atomic_codegen_tests)
  return()
endif()

libcudacxx_codegen_check_tools(libcudacxx_codegen_tests_enabled)
if (NOT libcudacxx_codegen_tests_enabled)
  return()
endif()

libcudacxx_codegen_get_cuda_architectures(atomic_codegen_cuda_archs PTX)
if (NOT atomic_codegen_cuda_archs)
  message(
    STATUS
    "No compatible CUDA architectures requested; skipping atomic PTX codegen tests"
  )
  return()
endif()

foreach (arch IN LISTS atomic_codegen_cuda_archs)
  libcudacxx_codegen_add_ptx_tests(
    AGGREGATE_TARGET libcudacxx.test.atomics.ptx
    TARGET_PREFIX atomic_codegen
    ARCH "${arch}"
    CHECK_PREFIXES SMXX
    TESTS ${libcudacxx_atomic_codegen_tests}
    COMPILE_DEFINITIONS _CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE=1
  )
endforeach()
