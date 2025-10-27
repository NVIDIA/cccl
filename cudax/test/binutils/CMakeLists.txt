find_package(cudax REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Find the cu++filt library.
find_library(cudax_cufilt_lib "cufilt" PATHS "${CUDAToolkit_LIBRARY_DIR}" NO_DEFAULT_PATH)
if (NOT cudax_cufilt_lib)
  message(FATAL_ERROR "cudax: cu++filt library (libcufilt.a) not found.")
endif()

foreach(cn_target IN LISTS cudax_TARGETS)
  cudax_get_target_property(config_prefix ${cn_target} PREFIX)
  set(config_meta_target ${config_prefix}.tests)

  # 1. positive case (pass)
  set(test_target ${config_prefix}.test.binutils_demangle)
  add_executable(${test_target} demangle.pass.cpp)
  target_compile_options(${test_target} PRIVATE "$<$<CXX_COMPILER_ID:MSVC>:/MT>")
  target_link_libraries(${test_target} PRIVATE
    ${cn_target}
    CUDA::cudart_static
    ${cudax_cufilt_lib}
  )
  cudax_clone_target_properties(${test_target} ${cn_target})
  add_dependencies(${config_meta_target} ${test_target})
  add_test(NAME ${test_target} COMMAND "$<TARGET_FILE:${test_target}>")

  # 2. missing <nvdecode.h> include but the function is not used (pass)
  set(test_target ${config_prefix}.test.binutils_demangle_no_nvdecode_pass)
  add_executable(${test_target} demangle_no_nvdecode.pass.cpp)
  target_compile_options(${test_target} PRIVATE "$<$<CXX_COMPILER_ID:MSVC>:/MT>")
  target_link_libraries(${test_target} PRIVATE ${cn_target})
  cudax_clone_target_properties(${test_target} ${cn_target})
  add_dependencies(${config_meta_target} ${test_target})
  add_test(NAME ${test_target} COMMAND "$<TARGET_FILE:${test_target}>")

  # 3. missing <nvdecode.h> and the function is used (fail)
  set(test_target ${config_prefix}.test.binutils_demangle_no_nvdecode_fail)
  add_test(NAME ${test_target}
    COMMAND ${CMAKE_CTEST_COMMAND}
            --build-and-test
                ${CMAKE_CURRENT_LIST_DIR}/binutils_demangle_no_nvdecode_fail
                ${CMAKE_CURRENT_BINARY_DIR}/binutils_demangle_no_nvdecode_fail
            --build-generator ${CMAKE_GENERATOR}
            --test-command ${CMAKE_CTEST_COMMAND}
  )
  set_tests_properties(${test_target} PROPERTIES WILL_FAIL true)
endforeach()
