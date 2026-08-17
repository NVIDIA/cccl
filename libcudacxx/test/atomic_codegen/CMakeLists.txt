if (libcudacxx_codegen_filecheck_atomics_ptx)
  add_custom_target(libcudacxx.test.atomics.ptx)

  set(libcudacxx_atomic_codegen_tests)
  if (NOT "NVHPC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
    file(GLOB libcudacxx_atomic_codegen_tests CONFIGURE_DEPENDS "*.cu")
  endif()

  if (libcudacxx_atomic_codegen_tests)
    libcudacxx_codegen_check_tools(libcudacxx_codegen_tests_enabled)
    if (libcudacxx_codegen_tests_enabled)
      libcudacxx_codegen_get_cuda_architectures(atomic_codegen_cuda_archs)
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
    endif()
  endif()
endif()

if (libcudacxx_codegen_filecheck_atomics_sass)
  add_subdirectory(sass)
endif()
