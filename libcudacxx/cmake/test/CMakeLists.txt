if ("MSVC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  # There's a bug that prevents build-and-test from working on MSVC.
  # See NVIDIA/nvbench#43.
  return()
endif()

set(cmake_opts
  -D "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
  -D "CMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
  -D "CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
)

# Temporary installation prefix for tests against installed project:
set(tmp_install_prefix "${CMAKE_CURRENT_BINARY_DIR}/test_install")

# Add a build-and-test CTest.
# - full_test_name_var will be set to the full name of the test.
# - subdir is the relative path to the test project directory.
# - test_id is used to generate a unique name for this test, allowing the
#   subdir to be reused.
# - Any additional args will be passed to the project configure step.
function(libcudacxx_add_compile_test full_test_name_var subdir test_id)
  set(test_name libcudacxx.test.cmake.${subdir}.${test_id})
  set(src_dir "${CMAKE_CURRENT_SOURCE_DIR}/${subdir}")
  set(build_dir "${CMAKE_CURRENT_BINARY_DIR}/${subdir}/${test_id}")
  add_test(NAME ${test_name}
    COMMAND "${CMAKE_CTEST_COMMAND}"
      --build-and-test "${src_dir}" "${build_dir}"
      --build-generator "${CMAKE_GENERATOR}"
      --build-options
        ${cmake_opts}
        ${ARGN}
      --test-command "${CMAKE_CTEST_COMMAND}" --output-on-failure
  )
  set(${full_test_name_var} ${test_name} PARENT_SCOPE)
endfunction()

################################################################################
# Test against source dir

libcudacxx_add_compile_test(test_name
  test_export
  source_tree
  -D "libcudacxx_DIR=${libcudacxx_SOURCE_DIR}/lib/cmake/libcudacxx/"
  -D TEST_TYPE=SOURCE_TREE
)

################################################################################
# Test against install tree

libcudacxx_add_compile_test(test_name
  test_export
  install_tree
  -D "libcudacxx_DIR=${tmp_install_prefix}/lib/cmake/libcudacxx/"
  -D TEST_TYPE=INSTALL_TREE
)
set_tests_properties(${test_name} PROPERTIES FIXTURES_REQUIRED install_tree)

################################################################################
# Install tree fixtures
add_test(NAME libcudacxx.test.cmake.install_tree.install
  COMMAND "${CMAKE_COMMAND}"
    --install "${libcudacxx_BINARY_DIR}"
    --prefix "${tmp_install_prefix}"
)
set_tests_properties(libcudacxx.test.cmake.install_tree.install PROPERTIES
  FIXTURES_SETUP install_tree
)

add_test(NAME libcudacxx.test.cmake.install_tree.cleanup
  COMMAND "${CMAKE_COMMAND}" -E rm -rf "${tmp_install_prefix}"
)
set_tests_properties(libcudacxx.test.cmake.install_tree.cleanup PROPERTIES
  FIXTURES_CLEANUP install_tree
)
